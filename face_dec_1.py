import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt

## REading video frame and convert to rgb
# cap = cv2.VideoCapture(0)

# while(True):
#     ret, frame = cap.read()
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

#     cv2.imshow('frame', rgb)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         out = cv2.imwrite('capture.jpg', frame)
#         break

# cap.release()
# cv2.destroyAllWindows()

# webcam = cv2.VideoCapture(0)
# ret, frame = webcam.read()
# print (ret)
# webcam.release()
# # Open a new thread to manage the external cv2 interaction
# cv2.startWindowThread()

# # Create a window holder to show you image in
# cv2.namedWindow("PyData Tutorial", cv2.WINDOW_NORMAL)
# cv2.imshow("PyData Tutorial", frame)
 
# # Press any key to close external window
# cv2.waitKey()   
# cv2.destroyAllWindows() dwf

# cap = cv2.VideoCapture(0)

# while(True):
#     ret, frame = cap.read()
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

#     cv2.imshow('frame', rgb)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         out = cv2.imwrite('capture.jpg', frame)
#         break

# cap.release()
# cv2.destroyAllWindows()
# Reading from storage and displaying it
# video = cv2.VideoCapture('video/mark.mp4')
# print(video.isOpened())
# cv2.namedWindow("feed",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("feed",850,480)

# while video.isOpened():
# 	ret, frame = video.read()
# 	cv2.imshow("feed",frame)
# 	if cv2.waitKey(27) & 0xFF == ord('q'):
# 	 	break

# video.release()
# cv2.destroyAllWindows()
# ### Recording and writing a video
webcam = cv2.VideoCapture(0)
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'H264')
video = cv2.VideoWriter('video/video_rod.avi',fourcc, 20.0, (640,480))

while webcam.isOpened():
    ret, frame = webcam.read()
    video.write(frame)
    # write/append to video object
    cv2.imshow('PyData Tutorial',frame)
    if cv2.waitKey(40) & 0xFF == 27:
        break
# release both video objects created
webcam.release()
video.release()
cv2.destroyAllWindows()
