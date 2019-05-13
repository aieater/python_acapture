from __future__ import print_function
import os
DEBUG = False
def to_bool(s): return s in [1,'True','TRUE','true','1','yes','Yes','Y','y','t']
USE_CONFIG = "ACAPTURE_CONFIG" in os.environ
if "DEBUG" in os.environ and to_bool(os.environ["DEBUG"]):
    DEBUG = True
    try: import __builtin__
    except ImportError: import builtins as __builtin__
    import inspect
    def lpad(s,c): return s[0:c].ljust(c)
    def rpad(s,c):
        if len(s) > c: return s[len(s)-c:]
        else: return s.rjust(c)
    def print(*args, **kwargs):
        s = inspect.stack()
        __builtin__.print("\033[47m%s@%s(%s):\033[0m "%(rpad(s[1][1],20), lpad(str(s[1][3]),10), rpad(str(s[1][2]),4)),end="")
        return __builtin__.print(*args, **kwargs)
_g_open = open
import platform
import subprocess
import sys
import time
import glob
import numpy as np
import multiprocessing
import threading
import re
try:
    import queue
except ImportError:
    import Queue as queue
from io import BytesIO
import configparser
import traceback

config = configparser.ConfigParser()
ini_file_name = os.path.basename(__file__.split(".")[0]+".ini")
if os.path.exists(ini_file_name) and USE_CONFIG:
    config.read(ini_file_name)
else:
    config["AsyncCamera"] = {"camera":0,"fps":60,"width":1280,"height":720,"format":"MJPG"}
    config["AsyncVideo"] = {"file":"video.mp4","loop":True,"frame_capture":False,"sound":True, "sound_volume":0.3}
    if USE_CONFIG: config.write(_g_open(ini_file_name,"w"))

def get_config(): return {section: dict(config[section]) for section in config.sections()}

try:
    from mss import mss
except:
    print("Error: Does not exist screen capture library.")
    print("   > pip3 install mss")


try:
    import cv2
except:
    print("Error: Does not exist OpenCV library.")
    print("   > curl -sL http://install.aieater.com/setup_opencv | bash -")
    print("   or")
    print("   > pip3 install opencv-python")

try:
    import os, sys
    with _g_open(os.devnull, 'w') as f:
        oldstdout = sys.stdout
        sys.stdout = f
        import pygame
        sys.stdout = oldstdout
except:
    traceback.print_exc()
    print("Error: Does not exist sound mixer library.")
    print("   > pip3 install pygame contextlib")



def which(program):
    if platform.uname()[0] == "Darwin":
        try:
            cmd = subprocess.check_output("which "+program,shell=True)
            cmd = cmd.decode("utf8").strip()
            return cmd
        except:
            return None
    else:
        def is_exe(fpath):
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
        fpath, fname = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file
    return None

FFMPEG = which('ffmpeg')
if FFMPEG is None:
    print("Error: Does not exist ffmpeg.")
    print("   > brew install ffmpeg # on MacOSX")
    print("   or")
    print("   > sudo apt install -y ffmpeg # on Ubuntu")


class BaseCapture(object):
    def keyboard_listener(self,key,x,y): pass
class AsyncCamera(BaseCapture):
    # format:YUYV/MJPG
    def __init__(self,fd=None,**kwargs):
        self.conf = config["AsyncCamera"]

        for k in self.conf: setattr(self,k,self.conf[k])
        for k in kwargs: setattr(self,k,kwargs[k])
        def s_bool(s,k): setattr(s,k,to_bool(getattr(s,k)))
        def s_int(s,k): setattr(s,k,int(getattr(s,k)))
        def s_float(s,k): setattr(s,k,float(getattr(s,k)))

        s_int(self,"fps")
        s_int(self,"width")
        s_int(self,"height")

        if fd == None:
            if re.match(r"\d",self.camera) is not None:
                fd = int(self.camera)
            else:
                raise "not available camera number"

        self.q = queue.Queue()
        self.q2 = queue.Queue()
        self.t = threading.Thread(target=AsyncCamera.func,args=(self.q,self.q2,fd,{"fps":self.fps,"width":self.width,"height":self.height,"format":self.format}))
        self.t.setName("AsyncCamera")
        self.t.setDaemon(True)
        self.t.start()
        self.current = None

    def destroy(self): self.q2.put(0)

    def is_ended(self): return False


    def func(q,q2,fd,opt):
        import cv2
        import time
        v = cv2.VideoCapture(fd)
        buffers = []
        v.set(cv2.CAP_PROP_FOURCC,(ord(opt['format'][0])<<0) + (ord(opt['format'][1])<<8) + (ord(opt['format'][2])<<16) + (ord(opt['format'][3])<<24))
        v.set(cv2.CAP_PROP_FPS,opt["fps"])
        v.set(cv2.CAP_PROP_FRAME_WIDTH,opt["width"])
        v.set(cv2.CAP_PROP_FRAME_HEIGHT,opt["height"])
        cnt = 0
        tm = time.time()
        while v.isOpened():
            stat, src = v.read()
            if stat:
                cnt+=1
                if time.time()-tm > 1.0:
                    if DEBUG: print("CameraFPS:",cnt, src.shape)
                    cnt = 0
                    tm = time.time()
                if q.qsize()<1:
                    src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
                    q.put((time.time(),src))
                if q2.qsize()>0:
                    print("Kill camera thread")
                    return
    def read(self):
        while self.current is None:
            if self.q.empty() == False:
                o = self.q.get()
                self.current = o
        if self.q.empty() == False:
            o = self.q.get()
            self.current = o
        return self.current

class AsyncVideo(BaseCapture):
    def __init__(self,fd=None,**kwargs):
        self.conf = config["AsyncVideo"]
        self.lock = threading.Lock()
        self.frame_capture = False
        self.reset = 0
        self.key_queue = []
        self.queue = queue.Queue()

        for k in self.conf: setattr(self,k,self.conf[k])
        for k in kwargs: setattr(self,k,kwargs[k])
        def s_bool(s,k): setattr(s,k,to_bool(getattr(s,k)))
        def s_int(s,k): setattr(s,k,int(getattr(s,k)))
        def s_float(s,k): setattr(s,k,float(getattr(s,k)))

        s_bool(self,"loop")
        s_bool(self,"frame_capture")
        s_bool(self,"sound")
        s_float(self,"sound_volume")

        if fd is None:
            fd = self.conf["file"]

        v = cv2.VideoCapture(fd,cv2.CAP_FFMPEG)
        self.start_time = 0
        self.offset = 0
        self.seq = v.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = v.get(cv2.CAP_PROP_FPS)
        sound = None
        sound_fd = None
        if self.frame_capture == False and self.sound and self.sound_volume > 0:
            FFMPEG = which('ffmpeg') #"/usr/local/bin/ffmpeg"
            fd = fd.replace("\\","")
            sound = fd + ".mp3"
            cmd = FFMPEG+" -i \""+fd+"\" -ab 192 -ar 44100 \""+fd+".mp3\""
            subprocess.call("/bin/bash -c \"if [ ! -e \""+sound+"\" ]; then "+cmd+"; fi\"", shell=True, stdout=subprocess.PIPE)
            if sound is not None:
                try:
                    sound_fd = _g_open(sound,"rb")
                    pygame.mixer.init()
                    pygame.mixer.music.load(sound_fd)
                except:
                    sound = None
                    sound_fd = None

        self.seq_is_ended = False
        self.need_to_close = False
        self.framecount = 0
        self.cnt = 0
        self.tm = time.time()
        self.v = v
        self.fd = fd
        self.sound = sound
        self.sound_fd = sound_fd
        self.f = 0
        self.current = (False,None)
        self.previous_frame = -1
        self.t = threading.Thread(target=self.func,args=())
        self.t.setName("AsyncVideo")
        self.t.setDaemon(True)
        self.t.start()

    def keyboard_listener(self,key,x,y):
        self.key_queue.append(key)

    def is_ended(self): return self.seq_is_ended

    def func(self):
        tm = time.time()
        cnt = 0
        tm2 = time.time()
        lock = self.lock
        while True:
            if time.time()-tm2 > 1.0:
                if threading.main_thread().is_alive() == False:
                    self.destroy()
                    if DEBUG: print("Leave from AsyncVideo thread")
                    return
                tm2 = time.time()
            if self.need_to_close: return

            #############################################################################
            # Extractor
            if self.frame_capture:
                if self.queue.qsize() > 60*60:
                    time.sleep(0.008)
                    continue
                check,frame = self.v.read()
                if check:
                    self.queue.put((check,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                    cnt+=1
                    if time.time() - tm > 1.0:
                        tm = time.time()
                        if DEBUG: print("VideoFPS:",cnt)
                        cnt = 0
                else:
                    if self.loop:
                        self.v.set(cv2.CAP_PROP_POS_FRAMES,0)
            #############################################################################
            # Player
            else:
                lock.acquire()
                if self.previous_frame != self.f:
                    current_frame = int(self.f)
                    if self.framecount > current_frame:
                        lock.release()
                        time.sleep(0.008)
                        self.previous_frame = current_frame

                        continue
                    self.framecount += 1
                    if current_frame-self.framecount > 10 or self.reset:
                        #print("SET",current_frame,self.framecount,self.reset)
                        self.v.set(cv2.CAP_PROP_POS_FRAMES,current_frame)
                        self.reset = 0
                        self.framecount = current_frame
                    lock.release()
                    check,frame = self.v.read()
                    lock.acquire()
                    if check:
                        self.current = (check,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    self.previous_frame = current_frame
                    cnt+=1
                    if time.time() - tm > 1.0:
                        tm = time.time()
                        if DEBUG: print("VideoFPS:",cnt)
                        cnt = 0
                # print(self.framecount,self.seq,self.f,self.offset,time.time()-self.start_time)
                if self.seq <= self.f:
                    if DEBUG: print("End")
                    if self.loop:
                        if DEBUG: print("Loop")
                        self.v.set(cv2.CAP_PROP_POS_FRAMES,0)
                        self.start_time = 0
                        self.framecount = 0
                        self.previous_frame = -1
                        self.offset = 0
                        self.f = 0
                    else:
                        self.seq_is_ended = True
                else:
                    # pass
                    time.sleep(0.008)
                lock.release()

    def destroy(self):
        self.need_to_close = True
        if self.sound is not None:
            pygame.mixer.music.stop()
            sound_fd.close()
        try:
            self.v.release()
            if DEBUG: print("Released-AsyncVideo")
        except:
            traceback.print_exc()

    def read(self):
        tm = time.time()
        #############################################################################
        # Extractor
        if self.frame_capture:
            if self.queue.qsize() > 0:
                return self.queue.get()
            return (False,None)

        #############################################################################
        # Player
        lock = self.lock
        lock.acquire()

        key = 0
        reset = 0
        if len(self.key_queue)>0:
            key = self.key_queue.pop(0)
        if key:
            if (key&0x0100):
                if (key&0xFF) == 100:#left
                    self.offset -= 5
                    self.reset = 1
                if (key&0xFF) == 101:#top
                    self.offset += 30
                    self.reset = 1
                if (key&0xFF) == 102:#right
                    self.offset += 5
                    self.reset = 1
                if (key&0xFF) == 103:#bottom
                    self.offset -= 30
                    self.reset = 1
                pf = ((tm-self.start_time+self.offset)*self.fps)

                self.framecount = 0
                if pf < 0:
                    self.start_time = tm
                    self.offset = 0
                if self.seq <= pf:
                    self.start_time = 0
                    self.offset = 0
                #print(key&0xFF)
            else:
                if (key&0xFF) == ord(b'q'):
                    return
                if (key&0xFF) == ord(b'/'): # dec vol
                    if self.sound is not None:
                        self.sound_volume -= 0.1
                        if self.sound_volume < 0: self.sound_volume = 0
                        pygame.mixer.music.set_volume(self.sound_volume)
                if (key&0xFF) == ord(b'*'): # inc vol
                    if self.sound is not None:
                        self.sound_volume += 0.1
                        if self.sound_volume > 1.0: self.sound_volume = 1.0
                        pygame.mixer.music.set_volume(self.sound_volume)



        f = ((tm-self.start_time+self.offset)*self.fps)
        if self.reset:
            if self.sound:
                p = 0
                if tm-self.start_time+self.offset > 0:
                    p = tm-self.start_time+self.offset
                pygame.mixer.music.play(0,p)

        # pygame.mixer.music.stop()
        # pygame.mixer.music.play(0,self.start_time*1)
        # # pygame.mixer.music.set_pos(0)
        # pygame.mixer.music.set_volume(self.sound_volume)
        if self.start_time == 0:
            if self.sound is not None:
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.play(0)
                    pygame.mixer.music.set_volume(self.sound_volume)
                except:
                    self.sound = None
            self.framecount = 0
            self.previous_frame = -1
            self.offset = 0
            self.start_time = tm
            self.offset = 0
            f = 0
        self.f = f
        lock.release()
        return self.current

class ImgFileStub(BaseCapture):
    def __init__(self,fd):
        self.f = cv2.imread(fd,cv2.IMREAD_COLOR)
        self.f = cv2.cvtColor(self.f,cv2.COLOR_BGR2RGB)
    def is_ended(self): return False
    def destroy(self): pass

    def read(self):
        return (True,self.f)

class DirImgFileStub(BaseCapture):
    def __init__(self,fd):
        self.f = fd
        if self.f[-1] != os.sep:
            self.f += os.sep

        self.f += "**"+os.sep+"*"
        files = glob.glob(self.f,recursive=True)
        self.flist = []
        for f in files:
            filename, ext = os.path.splitext(f)
            ext = ext.lower()
            if ext == ".png" or ext == ".jpg" or ext == ".jpeg" or ext == ".tiff" or ext == ".psd" or ext == ".gif" or ext == ".bmp":
                f = os.path.join(self.f, f)
                self.flist += [f]
    def is_ended(self): return len(self.flist) == 0
    def destroy(self): pass
    def read(self):
        while len(self.flist) > 0:
            ff = self.flist.pop(0)
            img = cv2.imread(ff,cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                return (True,img)
        return (False,None)

class ScreenCapture(BaseCapture):
    def __init__(self): self.need_to_close = False
    def is_ended(self): return False
    def destroy(self): self.need_to_close = True
    def read(self):
        with mss() as sct:
            monitor = sct.monitors[1]
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return (True,img)
        return (False,None)


def open(f,**kwargs):
    if type(f) == str:
        if re.match(r"\d",f) is not None: f = int(f)
        if f == "-1": f = -1
    if isinstance(f, (int,)):
        if f == -1:
            return ScreenCapture()
        return AsyncCamera(f,**kwargs)
    return AsyncVideo(f,**kwargs)
    if os.path.exists(f):
        if os.path.isdir(f):
            return DirImgFileStub(f)
        else:
            filename, ext = os.path.splitext(f)
            ext = ext.lower()
            if ext == ".png" or ext == ".jpg" or ext == ".jpeg" or ext == ".tiff" or ext == ".psd" or ext == ".gif" or ext == ".bmp":
                return ImgFileStub(f)
            else:
                return AsyncVideo(f,**kwargs)
    else:
        print("Does not exist",f)
    return None

def camera_info():
    if platform.uname()[0] == "Linux":
        exe_file = which("v4l2-ctl")
        subprocess.check_call(exe_file + " -d /dev/video0 --list-formats-ext",shell=True)
    else:
        print("This is available on Linux")
    # v4l2-ctl -d /dev/video0 --list-formats-ext
    # sudo apt-get install uvccapture guvcview uvcdynctrl
    # luvcview
    # sudo apt-get -y install uvccapture
    pass







def extract_video2images(f,**kwargs):
    FFMPEG = which('ffmpeg')
    f = f.strip()
    dr = os.path.join(os.path.dirname(f),os.path.basename(f).split(".")[0])
    mkdir = "mkdir -p \"" + dr +"\""

    quality = 2
    if "quality" in kwargs:
        quality = int(kwargs["quality"])
    format = "jpg"
    if "format" in kwargs:
        format = kwargs["format"]
    cmd = "%s -i \"%s\" -qscale:v %d  \"%s/image_%%05d.jpg\"" % (FFMPEG,f,quality,dr,)
    if format == "png":
        cmd = "%s -i \"%s\" -vcodec png \"%s/image_%%05d.png\"" % (FFMPEG,f,dr,)
    print(mkdir)
    subprocess.call(mkdir,shell=True)
    print(cmd)
    subprocess.call(cmd,shell=True)

def compress_images2video(f,**kwargs):
    FFMPEG = which('ffmpeg')
    f = os.path.abspath(f)
    format = "jpg"
    if "format" in kwargs:
        format = kwargs["format"]
    cmd = "%s -framerate 30 -i \"%s/image_%%05d.jpg\" -vcodec libx264 -pix_fmt yuv420p -r 60 \"%s.out.mp4\"" % (FFMPEG,f,f,)
    if format == "png":
        cmd = "%s -framerate 30 -i \"%s/image_%%05d.png\" -vcodec libx264 -pix_fmt yuv420p -r 60 \"%s.out.mp4\"" % (FFMPEG,f,f,)
    print(cmd)
    subprocess.call(cmd,shell=True)

def extract_video2audio(f):
    FFMPEG = which('ffmpeg')
    f = os.path.abspath(f)
    cmd = "%s -i \"%s\" -ab 192 -ar 44100 \"%s.out.mp3\"" % (FFMPEG,f,f,)
    print(cmd)
    subprocess.call(cmd,shell=True)

def join_audio_with_video(f,sf):
    FFMPEG = which('ffmpeg')
    f = os.path.abspath(f)
    sf = os.path.abspath(sf)
    print(f,sf)
    cmd = "%s -i \"%s\" -i \"%s\"  -map 0:v -map 1:a -c copy -shortest \"%s.mkv\"" % (FFMPEG,f,sf,f,)
    print(cmd)
    subprocess.call(cmd,shell=True)

def convert(f,func):
    f = os.path.abspath(f)
    video = Video(f,frame_capture=True)
    count = 0
    dr = f.split(".")[0]
    mkdir = "mkdir -p \"" + dr +"\""
    print(mkdir)
    subprocess.call(mkdir,shell=True)
    while True:
        img = video.read_frame()
        if img is not None:
            img = std_resize(img)
            imgs = func(img)
            for im in imgs:
                fname = "%s/image_%05d.png" % (dr,count,)
                cv2.imwrite(fname,im)
                print(count)
                count += 1
        else:
            break
    print("Done")
    print("Clear")
    subprocess.call("rm -f \"%s.mp3\""%(f,),shell=True)
    subprocess.call("rm -f \"%s.out.mp3\""%(f,),shell=True)
    subprocess.call("rm -f \"%s.out.mp4\""%(dr,),shell=True)
    subprocess.call("rm -f \"%s.out.mp4.mkv\""%(dr,),shell=True)
    print("Create an audio")
    toaudio(f)
    print("Archiving....")
    compress(dr)
    print("Attach audio to video.")
    joinaudio(dr+".out.mp4",f+".out.mp3")
    print("Completed.")
    
def gamma(img,g):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, g) * 255.0, 0, 255)
    img = cv2.LUT(img, lookUpTable)
    return img


if __name__ == '__main__':
    import acapture
    import pyglview
    import sys
    # cap = acapture.open(-1)
    cap = None
    if len(sys.argv)>1:
        cap = acapture.open(sys.argv[1])
    else:
        cap = acapture.open(os.path.join(os.path.expanduser('~'),"test.mp4"))
    view = pyglview.Viewer(keyboard_listener=cap.keyboard_listener)
    def loop():
        try:
            check,frame = cap.read()
            if check:
                view.set_image(gamma(frame,0.6))
        except:
            traceback.print_exc()
            exit(9)
        pass
    view.set_loop(loop)
    view.start()
    print("Main thread ended")
