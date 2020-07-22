#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
from skimage import io
import random
import numpy as np
from numpy.linalg import inv


# In[2]:


i=0
j=0  #after running the script, you could delete this line and then rerun the script, by running it again and again, you could get different images

while i<=23: #23 is based on the the number of images that is included in the file "fiftyimages"
    
    i=i+1
    j=j+1
    
    testimg=cv2.imread('fiftyimages' + '/' + 'fiftyimage (%d).tif'%(i))
    
    testimg=cv2.resize(testimg,(160,120))
    
    y0=random.randint(3,50)
    x0=random.randint(3,90)
    
    cropped=testimg[y0:y0+64,x0:x0+64]
    
    top_x = random.randint(16, 80)     #The structure of this part of the code is cited from  https://github.com/paeccher/Deep-Homography-Estimation-Pytorch/blob/master/dataset.py
    top_y = random.randint(16, 40)
    
    top_left_point = (top_x, top_y)
    top_right_point = (top_x+64, top_y)
    bottom_left_point = (top_x, top_y+64)
    bottom_right_point = (top_x+64, top_y+64)
    
    four_points = [top_left_point, top_right_point, bottom_left_point, bottom_right_point]
    
    perturbed_four_points = []
    
    for point in four_points:
        perturbed_four_points.append( (point[0] + random.randint(-16,16), point[1] + random.randint(-16,16)) )
    
    H = cv2.getPerspectiveTransform( np.float32(four_points), np.float32(perturbed_four_points) )
    H_inverse = inv(H)
    warped_image = cv2.warpPerspective(cropped, H_inverse, (64,64))   #The citation ends here, note that only the structure was cited, the parameters are self-designed
    
    cv2.imwrite(r"D:\summer\rescaledimage\resized%d.tif"%(j),cropped)     #create these files to save the images
    cv2.imwrite(r"D:\summer\randomperspective\warped%d.tif"%(j),warped_image)
    np.savetxt(r'D:\summer\homographyparam\H%d.txt'%(j),H)
    np.savetxt(r'D:\summer\inverse\H_inverse%d.txt'%(j),H_inverse)


# In[3]:


cv2.imshow('cropped',cropped)  #the following lines could show you an example of the images that are generated
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('warped_image',warped_image)
cv2.waitKey()
cv2.destroyAllWindows()

