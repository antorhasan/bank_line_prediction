import rasterio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.plot import show_hist
from utils.view import *
import glob
from os import listdir
from os.path import isfile, join

path = '/media/antor/Stuff/downloads/lan8m/'

#files = [f for f in glob.glob(path + "*.tif", recursive=True)]
files = [f for f in listdir(path) if isfile(join(path, f))]

#print(files)

for f in files:
    print(f)
    #raster = rasterio.open("./data/l813.tif")
    raster = rasterio.open(path + f)

    img = raster.read()

    vix = viz(img)
    #vix.cv_view()
    vix.cv_write('./data/lan8m/', f)

""" 
raster = rasterio.open("./data/l813.tif")

print(raster.meta)
band1 = raster.read()

vix = viz(band1)
vix.cv_view() """



#img = rasterio.plot.reshape_as_image(band1)
#img = band1[:,:,0:3]
#img = np.moveaxis(band1, 1, -1)
#img = np.moveaxis(band1, 0, 1)
#img = rasterio.plot.reshape_as_image(img)
#rgb = viz(img).torgb()

#print(img.shape)

#print(img.shape)


#plt.imshow(rgb)
#plt.show()
#show(img,adjust='linear')
#show_hist(band1[0:3,:,:],bins=50, histtype='stepfilled',lw=0.0, stacked=False, alpha=0.3)

