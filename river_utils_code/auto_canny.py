
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
#import matplotlib.pyplot as plt
import glob

dir_d = "/media/antor/Files/main_projects/finally/*.jpg"
dir_list = glob.glob(dir_d)
dir_list = dir_list[0:2]

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
counter = 1
for i in range(len(dir_list)):
    img = cv2.imread(dir_list[i] ,cv2.IMREAD_GRAYSCALE)
    denoise = cv2.fastNlMeansDenoising(img, h=24, templateWindowSize=7, searchWindowSize=21)
    edges = auto_canny(denoise)
    #denoise = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    cv2.namedWindow('denoise',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('denoise',edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #cv2.imwrite("F:/D/Papers/River/"+str(counter)+".jpg",edges)
    counter+=1

