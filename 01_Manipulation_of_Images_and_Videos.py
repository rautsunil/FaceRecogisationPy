
# coding: utf-8

# <div style="float: left; width: 50%; height: 200px; padding-bottom:40px">
#     <img src="http://pydata.org/amsterdam2016/static/images/pydata-logo-amsterdam-2016.png" alt="PyData Amsterdam 2016 Logo">
# </div>
# <div style="float: right; width: 50%; height: 200px; padding-bottom:40px">
#     <img style="height: 100%; float:right" src="http://pydata.org/amsterdam2016/media/sponsor_files/qualogy_logo_350px.png" alt="Qualogy Logo">
# </div>
# 
# # Building a Face Recognition System with OpenCV in the blink of an Eye
# 
# This notebook was created for the tutorial during the PyData Meeting:
# - Author: <font color='#be2830'>Rodrigo Agundez from Qualogy</font>
# - Place: Papaverweg 265, Amsterdam
# - Date: <font color='#be2830'>Saturday March 12, 2016</font>
# - Time: <font color='#be2830'>16:15</font>
# - Room 2
# 
# 
# 

# The goal of this tutorial is to build a simple face recognition system with the use of the opencv library. This tutorial is separated in four parts:
# 1. Manipulation of Images and Videoss
# 2. Face Detection and Building the Dataset
# 3. Building the Recognition Model
# 4. Recognize Faces in a Live VIdeo Feed
# <br>Extra: Try to trick the face recognition to classify other types of objects.
# 
# <br>
# <h2 align="center" style='color: white; background-color: #be2830'>Let's Get Started!</h2> 

# <h2 align="center" style='color: #be2830'>Imports</h2>

# In[3]:


import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import cv2


# ### A bit about OpenCV
# OpenCV is an open source computer vision and machine learning software library.
# The library includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms. These algorithms can be used to:
# <div style="float: left; width: 40%; margin-top: 16px; margin-bottom: 16px">
# <ul style="align: left; list-style-type:square">
#   <li>Detect Faces</li>
#   <li>Recognize Faces</li>
#   <li>Identify Objects</li>
#   <li>Classify human actions in videos</li>
#   <li>Track camera movement</li>
#   <li>Track moving objects</li>
# </ul>
# </div>
# <div style="float: right; width: 60%; margin-top: 16px; margin-bottom: 16px">
# <ul style="align: left; list-style-type:square">
#   <li>Extract 3D models of objects</li>
#   <li>Produce 3D point clouds from stereo cameras</li>
#   <li>Stitch images together to produce a high resolution image of an entire scene</li>
#   <li>Find similar images from an image database</li>
#   <li>Remove red eyes from images taken using flash</li>
#   <li>Follow eye movements</li>
# </ul>
# </div>
# 
# It has C++, C, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS. 
# 
# ### Requiered Packages for this tutorial
# <ul style="list-style-type:square">
#   <li>OpenCV (cv2)</li>
#   <li>Numpy</li>
#   <li>matpotlib</li>
# </ul>

# <h2 align="center" style='color: #be2830'>Let's Take  Picture</h2>

# In[5]:


webcam = cv2.VideoCapture(0)
ret, frame = webcam.read()
print (ret)
webcam.release()


# <h2 align="center" style='color: #be2830'>How to show it?</h2>

# ### Resizable Window

# In[1]:


# Open a new thread to manage the external cv2 interaction
cv2.startWindowThread()

# Create a window holder to show you image in
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_NORMAL)
cv2.imshow("PyData Tutorial", frame)
 
# Press any key to close external window
cv2.waitKey()   
cv2.destroyAllWindows() 


# ### Fix Window

# In[ ]:


# Create a window holder to show you image in
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)
cv2.imshow("PyData Tutorial", frame)

# Press any key to close external window
cv2.waitKey() 
cv2.destroyAllWindows()


# ### What about within the notebook?

# <h2 align="center" style='color: #be2830'>Frame is a numpy array!</h2>

# In[ ]:


print (frame)


# ### Inside the notebook

# In[ ]:


plt.imshow(frame)
plt.show()


# ### Looks ugly, what happened?
# <div align="center" style="margin-top:20px"> 
# OpenCV $\rightarrow$ BGR format<br>
# matplotlib $\rightarrow$ RGB
# </div>

# ### From BGR to RGB format

# In[ ]:


# Pixel color conversion
frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.imshow(frame_RGB)

# Let's take those ugly ticks off
plt.axis("off") 
plt.show()  


# <h2 align="center" style='color: #be2830'>Read and Write Images</h2>
# 
# ``` python
# cv2.imwrite(file_path (str), image (numpy.ndarray))
# 
# cv2.imread(file_path (str), read_mode (int))
# ```
# #### Read Modes
# -  ```1 = cv2.IMREAD_COLOR```
# -  ```0 = cv2.IMREAD_GRAYSCALE```
# - ```-1 = cv2.IMREAD_UNCHANGED```

# ### Write in GBR or RGB?

# In[ ]:


cv2.imwrite('images/picture_GBR.jpg',frame)
cv2.imwrite('images/picture_RGB.jpg',frame_RGB)

os.system("nautilus images") 


# ### Read in GBR or RGB?

# In[ ]:


read_mode = 1

picture_GBR = cv2.imread('images/picture_GBR.jpg', read_mode)
picture_RGB = cv2.imread('images/picture_RGB.jpg', read_mode)

# numpy intervention
picture = np.hstack((picture_GBR, picture_RGB))
plt.axis("off")
plt.title("GBR   RGB")
plt.imshow(picture, cmap="Greys_r")
plt.show()


# <h2 align="center" style='color: red'>OpenCV read & write</h2>
# 
# | Image File |$\rightarrow$|OpenCV     |$\rightarrow$|Image File   |
# |:-----------:|:-----------:|:---------:|:-----------:|:-----------:|
# | .jpg .png etc| cv2.imread() | numpy array | cv2.imwrite() | .jpg .png etc |
# |RGB|$\rightarrow$|GBR|$\rightarrow$|RGB
# 

# ### Useful function

# In[ ]:


def plt_show(image, title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off") 
    plt.title(title)
    plt.imshow(image, cmap="Greys_r")
    plt.show()


# <h2 align="center" style='color: #be2830'>Let's Take  Video</h2>

# In[ ]:


# Open connection to camera
webcam = cv2.VideoCapture(0)
print (webcam.isOpened())


# ### How to show it?
# 1. Get a frame from webcam
# 2. Show the frame
# 3. Repeat

# <h2 align="center" style='color: #be2830'>External Window or in Notebook</h2>

# ### External Window

# In[ ]:


cv2.namedWindow("PyData Tutorial", cv2.WINDOW_NORMAL)
 
while True:
      
    _, frame = webcam.read()
    cv2.imshow("PyData Tutorial", frame) 
     
    # code 27 is ESC key
    if cv2.waitKey(20) & 0xFF == 27:
        break
        
cv2.destroyAllWindows() 


# NOTE: Video feed killed but object still exists

# ### In Notebook
# Note: Since the notebook does not run as a tty device type getch() returns an error so we cannot catch a press key with this method. <br>
# `Instead KeyboardInterrupt` exception.

# In[ ]:


# module to allow interactive window inside notebook
from IPython.display import clear_output
try:
    while True:
        _, frame = webcam.read()
        plt_show(frame)
        clear_output(wait=True)
except KeyboardInterrupt:
    print ("Live Video Interrupted")


# ### Slower, why? I don't really know

# ## Video feed killed but live video feed object still exists

# In[ ]:


webcam.release()


# <h2 align="center" style='color: #be2830'>Read and Write Videos</h2>
# 
# ``` python
# cv2.VideoCapture(file_path (str))
# 
# fourcc = cv2.VideoWriter_fourcc(video_codec)
# 
# container = cv2.VideoWriter(file_path (str), fourcc, frames_per_second (float), pixel_size (tuple))
# 
# container.write(frame (np.ndarray))
# ```
# #### Codecs
# -  ```XVID, MJPG, X264, DIVX, H264, etc. ```
# - Complete list of codecs in [www.fourcc.org](http://www.fourcc.org/codecs.php "Wikipedia")

# ### Reading a video - Display in External Window

# In[ ]:


video = cv2.VideoCapture("videos/video_test.avi")
print (video.isOpened())

cv2.namedWindow("PyData Tutorial", cv2.WINDOW_NORMAL)
cv2.resizeWindow("PyData Tutorial", 850, 480)

while video.isOpened():
    
    ret, frame = video.read()  
#     if not ret: 
#         break
    
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.flip(frame, 1)

    cv2.imshow("PyData Tutorial", frame)
     
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
        
video.release()
cv2.destroyAllWindows() 


# ### Reading a video - Display in Notebook

# In[ ]:


video = cv2.VideoCapture("videos/video_test.avi")

try:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        # REMEMBER width ---> columns, height ---> rows
        width = frame.shape[1]
        
        frame = frame[:,width / 3:width,:]
        plt_show(frame)
        clear_output(wait=True)
except KeyboardInterrupt:
    print ("Video Interrupted")
video.release()


# ### Recording and writing a video
# 
# Waiting Time : time involved in cv2.waitKey()
# $$ FPS \approx \frac{1}{Waiting Time [seconds]} \\$$

# ### Using an External Window

# In[ ]:


webcam = cv2.VideoCapture(0)
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('videos/video_rod.avi',fourcc, 20.0, (640,480))

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
os.system("vlc videos/video_rod.avi")


# ### Using the Notebook

# In[ ]:


webcam = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('videos/video_rod.avi',fourcc, 7, (640,480))
try:
    while webcam.isOpened():
        ret, frame = webcam.read()
        video.write(frame)
        plt_show(frame)
        clear_output(wait=True)
except KeyboardInterrupt:
    print ("Video capture stopped")
# release both video objects created
webcam.release()
video.release()
cv2.destroyAllWindows()
os.system("vlc videos/video_rod.avi")


# ### Display video using html
# ```html
# <video src="video_rod.avi"></video>
# ```
# <div style="text-align: center">
#     <video style="float: center" src="videos/video_rod.avi" controls></video>
# </div>
# 
# <img src="images/video_support.png">
# 

# ### Using .mp4 

# In[ ]:


webcam = cv2.VideoCapture(0)
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'H264')
video = cv2.VideoWriter('videos/video_rod.mp4',fourcc, 20.0, (640,480))

while webcam.isOpened():

    ret, frame = webcam.read()
    # write/append to video object
    video.write(frame)
    cv2.imshow('PyData Tutorial',frame)
    if cv2.waitKey(40) & 0xFF == 27:
        break
        
# release both video objects created
webcam.release()
video.release()
cv2.destroyAllWindows()


# ### Display video using html
# ```html
# <video src="videos/video_rod.mp4"></video>
# ```
# <div style="text-align: center">
#     <video style="float: center" src="videos/video_rod.mp4" controls></video>
# </div>

# <h2 align="center" style='color: #be2830'>Drawing and Writing on Images/Videos</h2>
# 
# ``` python
# cv2.line(image, coord_1 (tuple), coord_2 (tuple), color_GBR (tuple), thickness (int))
# 
# cv2.rectangle(image, top_left (tuple), bottom_right (tuple), color_GBR (tuple), thickness (int))
# 
# cv2.circle(image, center (tuple), radius (int), color_GBR (tuple), thickness (int))
# 
# cv2.ellipse(image, center (tuple), axes_length (tuple), angle (int), start_angle (int), end_angle (int), color (tuple), thickness (int))
# 
# cv2.putText(image, text (str), bottom_left (tuple), font, size (float), color (tuple), thickness (int))
# ```
# 

# ### Rectangle and dynamic typing

# In[ ]:


webcam = cv2.VideoCapture(0)
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)
message = ""

while webcam.isOpened():
    
    _, frame = webcam.read()
    
    cv2.rectangle(frame, (100, 100), (530, 400), (150, 150, 0), 3)
    cv2.putText(frame, message, (95, 95), cv2.FONT_HERSHEY_SIMPLEX, .7, 
                (150, 150, 0), 2)
    
    cv2.imshow('PyData Tutorial',frame)
    key = cv2.waitKey(100) & 0xFF
    if key not in [255, 27]:
        message += chr(key)
    elif key == 27:
        break
        
# release both video objects created
webcam.release()
cv2.destroyAllWindows()


# <h2 align="center" style='color: #be2830'>Using a Mask</h2>
# <br>
# <div style="float: left; width: 40%; padding-bottom:40px">
#     <p> Stencil </p>
#     <img src="http://blog.speckproducts.com/blog/wp-content/uploads/2013/12/news_miamiMiniMaker-03_800w.jpg">
# </div>
# <div style="float: right; width: 40%; padding-bottom:40px">
#     <p> Semiconductors </p>
#     <img src="http://willson.cm.utexas.edu/Research/Sub_Files/Immersion/images/immersion_2.jpg">
# </div>
# 

# ### View through a circle

# In[ ]:


webcam = cv2.VideoCapture(0)
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)

while webcam.isOpened():
    
    _, frame = webcam.read()
    mask = np.zeros_like(frame)
    height, width, _ = frame.shape
    
    cv2.circle(mask, (width / 2, height / 2), 200, (255, 255, 255), -1)
    frame = np.bitwise_and(frame, mask)
    
    cv2.imshow('PyData Tutorial', frame)
    if cv2.waitKey(40) & 0xFF == 27:
        break
        
# release both video objects created
webcam.release()
cv2.destroyAllWindows()


# <h2 align="center" style='color: #be2830'>Handeling Mouse Events</h2>
# 
# ``` python
# cv2.setMouseCallback(window (str), callback_function)
# def function(event (int), x (int), y (int), )
# ```
# <div style="float: left; width: 50%">
# Events:
# - `cv2.EVENT_MOUSEMOVE`
# - `cv2.EVENT_LBUTTONDOWN`
# - `cv2.EVENT_RBUTTONDOWN`
# - `cv2.EVENT_MBUTTONDOWN`
# - `cv2.EVENT_LBUTTONUP`
# </div>
# <br>
# - `cv2.EVENT_RBUTTONUP`
# - `cv2.EVENT_MBUTTONUP`
# - `cv2.EVENT_LBUTTONDBLCLK`
# - `cv2.EVENT_RBUTTONDBLCLK`
# - `cv2.EVENT_MBUTTONDBLCLK`

# ### Draw circle with mouse

# In[ ]:


# mouse callback function
def draw_circle(event,x,y,flags,param):
    global x_in, y_in
    if event == cv2.EVENT_LBUTTONDOWN:
        x_in = x 
        y_in = y
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(frame, (int((x + x_in)) / 2, int((y + y_in)/2)), 
                   int(math.sqrt((y - y_in) ** 2 + (x - x_in) ** 2) / 2), (150, 150, 0), -1)
        
cv2.namedWindow('PyData Tutorial')
cv2.setMouseCallback('PyData Tutorial', draw_circle)

webcam = cv2.VideoCapture(0)
_, frame = webcam.read()
webcam.release()

while True:
    cv2.imshow('PyData Tutorial',frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()


# ### Using a mask to uncover image

# In[ ]:


# mouse callback function
def draw_circle(event,x,y,flags,param):
    global x_in, y_in
    if event == cv2.EVENT_LBUTTONDOWN:
        x_in = x 
        y_in = y
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(mask, (int((x + x_in)) / 2, int((y + y_in)/2)), 
                   int(math.sqrt((y - y_in) ** 2 + (x - x_in) ** 2) / 2), 
                   (255, 255, 255), -1)
        
cv2.namedWindow('PyData Tutorial')
cv2.setMouseCallback('PyData Tutorial', draw_circle)

webcam = cv2.VideoCapture(0)
_, frame = webcam.read()
mask = np.zeros_like(frame)

while True:
    _, frame = webcam.read()
    frame = np.bitwise_and(frame, mask)
    cv2.imshow('PyData Tutorial', frame)
    if cv2.waitKey(40) & 0xFF == 27:
        break
webcam.release()
cv2.destroyAllWindows()


# ## NEXT
# <ol> 
#     <h2> <li> Manipulation of Images and Videos. [DONE]</h2> 
#     <h2 style='color: #be2830'><a style='color: #be2830' href="http://localhost:8888/notebooks/02_Face_Detection_and_Building_the_Dataset.ipynb"> <li> Face Detection and Building the Dataset</a></h2>
#     <h2> <li>Building the Recognition Model</h2>
#     <h2> <li> Recognize Faces in a Live VIdeo Feed</h2>
# <ol>
