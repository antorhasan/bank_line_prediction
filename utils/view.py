import rasterio
import cv2
from rasterio.plot import show
from rasterio.plot import show_hist
import numpy as np

class viz():
    '''change bgr to rgb of a 3 band image
        input : np.ndarray
        output : np.ndarray'''
    def __init__(self, raster):
        self.raster = raster
    

    def cv_view(self):
        '''view raster file's first 3 channel in opencv, input: raster object. 
        output: visualization of the first 3 channel in opencv'''

        img = rasterio.plot.reshape_as_image(self.raster)
        img = np.asarray(img)
        img = img[:,:,0:3]

        '''the int16 format of the input channels needs to be changed into regular int64
        format in order to broadcast properly with int64 numpy array'''
        img = np.divide(np.multiply(np.int64(img), [255]), [3000])   
        
        #conversion is needed to unit8 to keep the range between 0-255
        img = np.uint8(img)

        '''opencv reads files in BGR channel format. Here this is satisfied and no 
        conversion is needed'''

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        #cv2.imwrite('./data/try48/',)
    
    def cv_write(self, where, filename):
        img = rasterio.plot.reshape_as_image(self.raster)
        img = np.asarray(img)
        img = img[:,:,0:3]
        
        img = np.divide(np.multiply(np.int64(img), [255]), [3000])   

        img = np.uint8(img)
        onlyname = filename.split('.')
        #print(onlyname[0])
        cv2.imwrite( where + onlyname[0] + '.png', img)