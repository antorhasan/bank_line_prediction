import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

path = './data/label1/'

files = [f for f in listdir(path) if isfile(join(path,f))]

#img = cv2.imread('./houghlines5.jpg')


#print(img.dtype)
#print(img)
for i in files:
    print(i)
    img = cv2.imread(path + i)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()