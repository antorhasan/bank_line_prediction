import cv2
import numpy as np

img = cv2.imread('./data/bankline/198801.png', 0)
print(img.shape)

img = img[121:-1-120,121:-1-120]
img = cv2.resize(img, (img.shape[1]+900,img.shape[0]+1000),interpolation = cv2.INTER_AREA)
img = img[:,175:-1]


scale_percent = 185
for_he = 165
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * for_he / 100)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

cropped = resized[243-40:2881-40,410+0:1813+0]

#print(cropped.dtype)
_ , fin = cv2.threshold(cropped,245,255,cv2.THRESH_BINARY)
print(fin.shape)
print(len(fin))
coor = np.where(img>240)
print(coor)
#print(len(coor[0]),len(coor[1]))
list_coor = list(zip(coor[0],coor[1]))

ori = cv2.imread('./data/finaljan/198801.png', 1)

for i in list_coor:
    ori[i[0],i[1]] = [0,0,255]


#thresh = np.asarray(thresh)
cv2.imwrite('./data/binline/first.png',fin)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', ori)
cv2.waitKey(0)
cv2.destroyAllWindows