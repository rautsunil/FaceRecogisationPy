import numpy as np
import cv2
from face_recognition_system.videocamera import VideoCamera
from face_recognition_system.externalviddeo import ExternalVideo
from face_recognition_system.detectors import FaceDetector
import face_recognition_system.operations as op

cap = cv2.VideoCapture('/Users/srauz/tensorflowworkspace/PyData-master/video/mark.mp4')
out = None
while(cap.isOpened()):
    ret, frame = cap.read()
    if out is None:
        [h, w] = frame.shape[:2]
        out = cv2.VideoWriter("/Users/srauz/tensorflowworkspace/PyData-master/video/test_out_face_ex.avi", 0, 25.0, (w, h))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()
