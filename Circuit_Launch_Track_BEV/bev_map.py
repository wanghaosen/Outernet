import matplotlib.pyplot as plt
import cv2
import numpy as np


fig = plt.figure(0)
front = cv2.imread("george.jpg")
# front = cv2.imread("input.jpg")
# front = cv2.imread("input5.jpg")
plt.imshow(front[:,:,::-1])
plt.show()

# In[56]:


# Pixel coordinates of the outer corner of the white markers in the image
# for george.jpg
imgpts = np.float32([
    [1196, 1380],
    [2668, 1504],
    [3844, 2244],
    [664, 2212.]
])

# #for input.jpg
# imgpts = np.float32([
#     [601, 757],
#     [1137, 791],
#     [804, 528],
#     [277, 319],       
# ])

# #for input5.jpg
# imgpts = np.float32([
#     [155, 655],
#     [1210, 660],
#     [964, 460],
#     [332, 437.]
# ]) 


# These are the desired coordinates in the output image. I'm using a 2cm/pixel scale;
# the markers form a 3.95m x 2.413m rectangle.
pixelSize = 0.02

#for george.jpg
dstpts = np.float32([
    [3.1623, 4.3942],
    [-3.1623, 3.1242],
    [-3.1623, -4.3942],
    [3.1623, -4.3942],
    ]) / pixelSize

# #for our pics
# dstpts = np.float32([
#     [-3.75, 2.83],
#     [3.75, 2.83],
#     [4.05, -2.83],
#     [-3.75, -3.98], 
#     ]) / pixelSize

imagesize = np.float32([1000, 600])
dstpts += np.float32(imagesize / 2)

# In[156]:


# generate the perspective matrix from the above four point pairs
M = cv2.getPerspectiveTransform(imgpts, dstpts)


# In[158]:


# Project the image into a new 900x450 image
# (I adjusted the image size here and offset above so the track
# is nicely centered in the output image)
bg = cv2.warpPerspective(front, M, tuple(imagesize))
cv2.imwrite("birdseye_cl_new.png", bg) # write out png