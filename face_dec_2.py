
# coding: utf-8

# In[ ]:


from IPython.display import YouTubeVideo
import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt
from IPython.display import clear_output # Extra

# Open a new thread to manage the external cv2 interaction
cv2.startWindowThread()



def plt_show(image, title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap="Greys_r")
    plt.show()
    


webcam = cv2.VideoCapture(0)
_, frame = webcam.read()
webcam.release()
plt_show(frame) 
#show_frame()


# ### Try to detect the face. What is the returned object?

# In[ ]:


detector = cv2.CascadeClassifier("xml/frontal_face.xml")

scale_factor = 1.2
min_neighbors = 5
min_size = (30, 30)
biggest_only = True
flags = cv2.CASCADE_FIND_BIGGEST_OBJECT |             cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else             cv2.CASCADE_SCALE_IMAGE
        
faces_coord = detector.detectMultiScale(frame,
                                        scaleFactor=scale_factor,
                                        minNeighbors=min_neighbors,
                                        minSize=min_size,
                                        flags=flags)
print (faces_coord)
print ("Length: " + str(len(faces_coord)) )


# ### Draw a rectangle around the face

# In[ ]:


for (x, y, w, h) in faces_coord:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 0), 8)
plt_show(frame) 

# ### Helpful classes definitions

# In[ ]:


class FaceDetector(object):
   

    def __init__(self, xml_path):
        # Create classifier object
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image, biggest_only=True):
       
        is_color = len(image) == 3
        if is_color:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
       
        scale_factor = 1.2
       
        min_neighbors = 5

        # Sets the min_size of the face we want to detect. Default is 20x20
        min_size = (30, 30)

        # Change to True if we want to detect only one face
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
            cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
            cv2.CASCADE_SCALE_IMAGE

        face_coord = self.classifier.detectMultiScale(
            image_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=flags
        )

        return face_coord


# ### Helpful classes definitions

# In[ ]:


class VideoCamera(object):
    """ A class to handle the video stream.
    """
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self, in_grayscale=False):
        """ Get current frame of a live video.
        :param in_grayscale: Frame captured in color or grayscale [False].
        :type in_grayscale: Logical
        :return: Current video frame
        :rtype: numpy array
        """
        _, frame = self.video.read()

        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    
    def show_frame(self, seconds, in_grayscale=False):
      
    	_, frame = self.video.read()
    	if in_grayscale:
        	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	cv2.imshow('SnapShot', frame)
    	key_pressed = cv2.waitKey(seconds * 1000)

    	return key_pressed & 0xFF
    

# ### Detect Face in a Live Video

# In[ ]:


webcam = VideoCamera()
detector = FaceDetector("xml/frontal_face.xml")


# In[ ]:


try:
    while True:
        frame = webcam.get_frame()
        faces_coord = detector.detect(frame)
        for (x, y, w, h) in faces_coord:
            cv2.rectangle(frame, (x, y), (x + w, y + h), 
                          (150, 150, 0), 8)
        #plt_show(frame)
        webcam.show_frame(1)
 

except KeyboardInterrupt:
     print ("Live Video Interrupted")


# Congrats, you have learned how to detect faces.

# <h2 align="center" style='color: #be2830'>Image Normalization</h2>
# 
# <div style="float: right; width: 30%; margin-right: 100px">
#     <img style="width: 50%" src="https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcQYNNV8kaI_AeCTzO1yZFGLcHJlUU8eiQX78Fa-S39jtC0FyU56">
#     <img src="http://docs.opencv.org/3.1.0/equalization_opencv.jpg">
# </div>
# 
# - Cut the Face
# - Normallize Pixel Intensity
# - Resize Face Image
# - Align Face Image? 
# 
# Before feeding the faces to train the model and before trying to recognize.

# <h2 align="center" style='color: #be2830'>State of the Art - Facebook DeepFace</h2>
# 
# <img style="float:right; width: 50%" src="images/facebook_norm.png">
# <br>
# <p>"DeepFace: Closing the Gap to Human-Level Performance in Face Verification"</p>
# <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf"><p> Y. Taigman et al., 2014.</p></a>
# 

# ### Cut the face
# <img style="width: 40%; float: right; margin-right: 100px" src="images/cut_face.png">
# Only the 70% of the width.
# ```python
# w_rm = int(0.2 * w / 2)
# ```

# In[ ]:


""" This module contains functions to manipulate images. Sush as:
- cut_face
- cut_face_ellipse
- normalize_intensity
- resize
"""

import numpy as np
import cv2

#!/usr/bin/env python
# operations.py

def resize(images, size=(100, 100)):
    """ Function to resize the number of pixels in an image.

    To achieve a standarized pixel number accros different images, it is
    desirable to make every picture of the same pixel size. By using an OpenCV
    method we increase or reduce the number of pixels accordingly.

    :param image: image to be resized.
    :param size: desired size for the output image
    :type image: numpy array of 2 or three dimensions
    :type size: tuple containing the size
    :return: the image with the acoordingly pixel size
    :rtype: numpy array of 2 dimensions
    """
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # using different OpenCV method if enlarging or shrinking
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm


def normalize_intensity(images):
    """ This method normalizes the size and pixel intensity of an image.

    Each image has their own distribution of intensity pixels in grayscale.
    This function normalizes these intensities such that the image uses
    all the range of grayscale values.

    :param image: image to normalize, can be in color or grayscale
    :type image: numpy array of two (graycale) or three (color) dimensions.
    :return: the image in grayscale with the intensities normalized
    :rtype: numpy array of two dimensions.
    """
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm


def cut_face_rectangle(image, face_coord):
    """ Cuts the image to just show the face.

    This function takes the rectangle coordenates around a detected face
    and cuts the original image returning only the detected face image.

    :param image: original image where face was detected
    :param face: rectangle containing (x,y,w,h)
    :type image: numpy array
    :type face: tuple
    :return: image of face only
    :rtype: numpy array
    """
    images_rectangle = []
    for (x, y, w, h) in face_coord:
        images_rectangle.append(image[y: y + h, x: x + w])
    return images_rectangle

def cut_face_ellipse(image, face_coord):
    """ Cuts the image to just show the face in an ellipse.

    This function takes the rectangle coordenates around a detected face
    or faces and cuts the original image with the face coordenates. It also
    surrounds the face with an ellipse making black all the extra
    background.
    or faces

    :param image: original image where faces were detected
    :param faces: object containing the detected face information
    :type image: numpy array
    :type faces: DetectedFace object
    :return: images containing only the face enclose by an ellipse
    :rtype: numpy array
    """
    images_ellipse = []
    for (x, y, w, h) in face_coord:
        center = (x + w / 2, y + h / 2)
        axis_major = h / 2
        axis_minor = w / 2
        mask = np.zeros_like(image)
        # create a white filled ellipse
        mask = cv2.ellipse(mask,
                           center=center,
                           axes=(axis_major, axis_minor),
                           angle=0,
                           startAngle=0,
                           endAngle=360,
                           color=(255, 255, 255),
                           thickness=-1)
        # Bitwise AND operation to black out regions outside the mask
        image_ellipse = np.bitwise_and(image, mask)
        images_ellipse.append(image_ellipse[y: y + h, x: x + w])

    return images_ellipse

def draw_face_rectangle(image, faces_coord):
    """ Draws a rectangle around the face found.
    """
    for (x, y, w, h) in faces_coord:
        cv2.rectangle(image, (x, y), (x + w, y + h), (206, 0, 209), 2)
    return image

def draw_face_ellipse(image, faces_coord):
    """ Draws an ellipse around the face found.
    """
    for (x, y, w, h) in faces_coord:
        center = (x + w / 2, y + h / 2)
        axis_major = h / 2
        axis_minor = w / 2
        cv2.ellipse(image,
                    center=center,
                    axes=(axis_major, axis_minor),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=(206, 0, 209),
                    thickness=2)
    return image

# <h2 align="center" style='color: #be2830'>Build Our Dataset</h2>
# <h4 align="center">
# Detect $\rightarrow$ Cut $\rightarrow$ Normalize $\rightarrow$ Resize $\rightarrow$ Save</h4>
# 

# In[ ]:


folder = "people/" + raw_input('Person: ').lower() # input name
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)
if not os.path.exists(folder):
    os.mkdir(folder)
    counter = 1
    timer = 0
    while counter < 21 : # take 20 pictures
        frame = webcam.get_frame()
        faces_coord = detector.detect(frame) # detect
        if len(faces_coord) and timer % 700 == 50: # every Second or so
            faces = normalize_faces(frame, faces_coord) # norm pipeline
            cv2.imwrite(folder + '/' + str(counter) + '.jpg', faces[0])
            #plt_show(faces[0], "Images Saved:" + str(counter))
            webcam.show_frame(1)
            counter += 1
        draw_rectangle(frame, faces_coord) # rectangle around face
        cv2.imshow("PyData Tutorial", frame) # live feed in external
        cv2.waitKey(50)
        timer += 50
    cv2.destroyAllWindows()
else:
    print ("This name already exists.")


# In[ ]:


del webcam


# ## NEXT
# <ol> 
#     <h2> <li> Manipulation of Images and Videos [DONE]</h2> 
#     <h2> <li> Face Detection and Building the Dataset [DONE]</h2>
#     <h2 style='color: #be2830'><a style='color: #be2830' href="http://localhost:8888/notebooks/03_Building_the_Recognition_Model.ipynb"> <li> Building the Recognition Model.</a></h2>
#     <h2> <li> Recognize Faces in a Live VIdeo Feed</h2>
# <ol>
