import numpy as np
import cv2

img = np.asarray(cv2.imread('../data/Landsat8/LC08_001004_20140524-0000000000-0000006912.tif'))


print(img)

""" cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows() """