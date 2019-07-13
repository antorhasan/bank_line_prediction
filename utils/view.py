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
    

    def torgb(self):
        orig = self.raster
        temp = self.raster
        print(orig[:,:,0:1])
        orig[:,:,0:1] = temp[:,:,2:3]
        orig[:,:,1:2] = temp[:,:,1:2]
        orig[:,:,2:3] = temp[:,:,2:3]
        print(orig[:,:,0:1])
        return orig

    def cv_view(self):
        img = rasterio.plot.reshape_as_image(self.raster)
        img = np.asarray(img)
        img = img[:,:,0:3]
        print(img.dtype)
        img = np.uint8((img*255)/3000)
        print(img)
        #img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        #print(img.shape)
        #img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
            

