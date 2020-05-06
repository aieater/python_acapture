#!/usr/bin/env python3
import os
import cv2
import acapture
cap = acapture.open(os.path.join(os.path.expanduser('~'),"test.mp4"))
while True:
    check,frame = cap.read()
    if check:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow("test",frame)
        cv2.waitKey(1)
