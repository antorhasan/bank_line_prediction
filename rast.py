import rasterio
import cv2

raster = rasterio.open("./data/drive-download-20190706T154825Z-001/LC08_138044_20130419.tif")

print(raster.meta)
band1 = raster.read(4)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', band1)
cv2.waitKey(0)
cv2.destroyAllWindows()
