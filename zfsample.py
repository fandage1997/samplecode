#!/usr/bin/env python
# coding: utf-8

# In[3]:


# %load zfsample.py
#!/usr/bin/env python

# In[1]:


# main part of the code

import torch
import cv2
import random
import numpy as np
from numpy.linalg import inv

img = cv2.imread('traintupian.jpg', 0)
img = cv2.resize(img, (320, 240))

test_image = img.copy()

top_x = random.randint(0+32, 320-32-128)
top_y = random.randint(0+32, 240-32-128)

top_left_point = (top_x, top_y)
top_right_point = (top_x+128, top_y)
bottom_left_point = (top_x, top_y+128)

bottom_right_point = (top_x+128, top_y+128)

four_points = [top_left_point, top_right_point, bottom_left_point, bottom_right_point]

perturbed_four_points = []
for point in four_points:
    perturbed_four_points.append( (point[0] + random.randint(-32,32), point[1] + random.randint(-32,32)) )

H = cv2.getPerspectiveTransform( np.float32(four_points), np.float32(perturbed_four_points) )

H_inverse = inv(H)

warped_image = cv2.warpPerspective(img, H_inverse, (320,240))

Ip1 = test_image[top_left_point[1]:bottom_right_point[1],top_left_point[0]:bottom_right_point[0]]
Ip2 = warped_image[top_left_point[1]:bottom_right_point[1],top_left_point[0]:bottom_right_point[0]]
training_image = np.dstack((Ip1, Ip2))
H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))

datum = (training_image, H_four_points)


# In[2]:


cv2.imshow('Image',img)#run the following 3 lines to show the original image (greyscale), close the window after seeing the image
cv2.waitKey()
cv2.destroyAllWindows()


# In[3]:


cv2.imshow('Image',warped_image)#run the following 3 lines to show the warped image
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




