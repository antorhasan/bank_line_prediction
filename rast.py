import rasterio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.plot import show_hist
from utils.view import viz

raster = rasterio.open("./data/drive-download-20190706T155419Z-001/LC08_137044_20130412.tif")

print(raster.meta)
band1 = raster.read()
#img = np.moveaxis(band1, 0, -1)
#img = rasterio.plot.reshape_as_image(band1)
#img = band1[:,:,0:3]
#img = np.moveaxis(band1, 1, -1)
#img = np.moveaxis(band1, 0, 1)
#img = rasterio.plot.reshape_as_image(img)
#rgb = viz(img).torgb()

print(band1.shape)

#print(img.shape)
img = band1[0:3,:,:]
#plt.imshow(rgb)
#plt.show()
#show(img,adjust='linear')
show_hist(img[0:1,:,:],bins=50, histtype='stepfilled',lw=0.0, stacked=False, alpha=0.3)
""" cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', band1)
cv2.waitKey(0)
cv2.destroyAllWindows() """
