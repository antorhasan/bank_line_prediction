import rasterio
import cv2
from rasterio.plot import show
from rasterio.plot import show_hist
import numpy as np
import matplotlib.pyplot as plt
from preprocess import *

class viz():
    '''change bgr to rgb of a 3 band image
        input : np.ndarray
        output : np.ndarray'''
    def __init__(self, path):
        self.path = path
        data = rasterio.open(self.path)
        data = data.read()
        img = rasterio.plot.reshape_as_image(data)
        img_np = np.asarray(img)
        img = img_np[:,:,:]
        #print(img.shape)
        '''the int16 format of the input channels needs to be changed into regular int64
        format in order to broadcast properly with int64 numpy array'''
        img = np.divide(np.multiply(np.int64(img), [255]), [3000])   
        
        #conversion is needed to unit8 to keep the range between 0-255
        self.img = np.uint8(img)
        self.img_np = img_np

    def get_array(self):
        return self.img_np

    def cv_view(self):
        '''view raster file's first 3 channel in opencv, input: raster object. 
        output: visualization of the first 3 channel in opencv'''

        '''opencv reads files in BGR channel format. Here this is satisfied and no 
        conversion is needed'''
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def cv_write(self, where, filename):

        onlyname = filename.split('.')
        #print(onlyname[0])
        cv2.imwrite( where + onlyname[0] + '.png', self.img)


""" 
def crop_to_256(array):
    data = []
    coor_list = [516, 491, 516, 450, 396, 355, 325, 277, 310, 400]
    img = self.img_np
    for i in range(len(coor_list)):
        num = coor_list[k]
        crop_img = img[256*k : 256*(k+1),num:num+768]

        for j in range(3):
            crop_img_256 = img_y[:,256*j:256*(j+1)]
            data.append(crop_img_256)
            print(i)
            data = np.asarray(data) """


path_list = path_sort('./data/finaltif/')
path_list = path_list[0:2]
data = []
for i in range(len(path_list)):
    img = viz('./data/finaltif/'+path_list[i])
    img_full = img.get_array()
    coor_list = [516, 491, 516, 450, 396, 355, 325, 277, 310, 400]

    for j in range(len(coor_list)):
        num = coor_list[j]
        crop_img = img_full[256*j : 256*(j+1),num:num+768]

        for k in range(3):
            crop_img_256 = crop_img[:,256*k:256*(k+1)]
            data.append(crop_img_256)
    print(i)

data = np.asarray(data)

print(data.shape)