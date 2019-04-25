# acapture (async capture python library)

## Description
acapture is a python camera/video capturing library for realtime.
When python apps implement video, web camera and screenshot capturing, it is too slow FPS and suffering with that performance.
In addition, python is non event driven architecture and always be I/O blocking  problem, and that will be imped parallelism.

acapture(AsynchronusCapture) library provides async video/camera capturing implementation and can solve that blocking and performance problems.


acapture library is useful instead of OpenCV VideoCapture API.

#### OpenCV has blocking problem.
```
import cv2
cap = cv2.VideoCapture(0)
check,frame = cap.read() # blocking!! and depends on camera FPS.
```

#### acapture library can solve that blocking problem in realtime apps.
```
import acapture
cap = acapture.open(0)
check,frame = cap.read() # non-blocking
```



### Also see 'pyglview' package.

OpenCV3 renderer is too slow due to cv2.waitKey(1).
If you want to more performance, you should use OpenCV4+ or 'pyglview' package.

https://github.com/aieater/python_glview.git

This package is supported fastest OpenGL direct viewer and OpenCV renderer both.
If your environment was not supported OpenGL, it will be switched to CPU renderer(OpenCV) automatically and also available remote desktop(Xserver) like VNC.



## Getting Started

##### Base libraries on Ubuntu16.04
|  Library  | installation  |
| ---- | ---- |
| Camera  | sudo apt install -y libv4l-dev libdc1394-22 libdc1394-22-dev v4l-utils |
| Video  | sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libfaac-dev libmp3lame-dev mplayer |

##### Base libraries on MacOSX
|  Library  | installation  |
| ---- | ---- |
| Brew | /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" |
| Camera  | - |
| Video  | brew install ffmpeg mplayer  |


##### Python package dependencies
|  Version  |  Library  | installation  |
| ---- | ---- | ---- |
|  v3.x/v4.x  |  OpenCV  | pip3 install opencv-python  |
|  v4.x  |  mss  | pip3 install mss  |
|  v1.1x.x  |  numpy  | pip3 install numpy  |
|  v1.9.x  |  pygame  | pip3 install pygame  |
|  v3.7.x  |  configparser  | pip3 install configparser  |


#### Finally, install acapture.

```
pip3 install acapture
```

-----

### Examples


#### Video stream (Async)
```
import acapture
import cv2

cap = acapture.open("test.mp4")
while True:
    check,frame = cap.read() # non-blocking
    if check:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow("test",frame)
        cv2.waitKey(1)
```

#### Video frames (Async)

```
cap = acapture.open("test.mp4",frame_capture=True)
```

#### Camera stream (Async)
```
import acapture
import cv2

cap = acapture.open(0) # /dev/video0
while True:
    check,frame = cap.read() # non-blocking
    if check:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow("test",frame)
        cv2.waitKey(1)
```

#### Screenshot stream (Sync)
```
import acapture
import cv2

cap = acapture.open(-1)
while True:
    check,frame = cap.read() # blocking
    if check:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow("test",frame)
        cv2.waitKey(1)
```

#### Directory images (Sync)
```
import acapture
import cv2

cap = acapture.open("images/")
while True:
    check,frame = cap.read() # blocking
    if check:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow("test",frame)
        cv2.waitKey(1)
```

#### Unit image (Preloaded)
```
import acapture
import cv2

cap = acapture.open("images/test.jpg")
while True:
    check,frame = cap.read()
    if check:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow("test",frame)
        cv2.waitKey(1)
```


#### Extract video to jpg images.
```
import acapture

acapture.extract_video2images("test.mp4",format="jpg",quality=2)
```


-----


##### APIs

|  Version  |  Function  | Required | Description  |
| ---- | ---- | ---- | ---- |
|  v1.0  |  open(f,**kargs)  | f |  Open stream. [-1=>screenshot], [0=>camera0], [1=>camera1], [dirpath=>images], [path=>image],[path=>video] |
|  v1.0  |  extract_video2images(path,**kargs)  | path | Extract video to images. |
|  v1.0  |  camera_info()  |  | Display camera information on Ubuntu. |
|  v1.0  |  compress_images2video(path,**kargs)  | path | Make video from images. |
|  v1.0  |  extract_video2audio(f)  | path |  Extract audio file as mp3. |
|  v1.0  |  join_audio_with_video(vf,af)  | vf, af | Join video file and audio file. |


### Also see 'pyglview' package.

OpenCV3 renderer is too slow.
If you want to more performance, you should use OpenCV4 or pyglview package.

https://github.com/aieater/python_glview.git

This package is supported fastest OpenGL viewer and OpenCV renderer both.
If your environment was not supported OpenGL, it will be switched to CPU renderer(OpenCV) automatically and also available remote desktop(Xserver) like VNC.

#### acapture + pyglview + webcamera example.
```
import cv2
import acapture
import pyglview
viewer = pyglview.Viewer()
cap = acapture.open(0) # Camera 0,  /dev/video0
def loop():
    check,frame = cap.read() # non-blocking
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if check:
        viewer.set_image(frame)
viewer.set_loop(loop)
viewer.start()
```
Logicool C922 1280x720(HD) is supported 60FPS.
This camera device and OpenGL direct renderer is best practice.
Logicool BRIO 90FPS camera is also good!, but little bit expensive.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
