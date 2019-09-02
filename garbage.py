import cv2
import numpy as np 

""" img = cv2.imread('./data/exp1/201601.png')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 """
def data_ag():
    """ path = path_sort('./data/exp1/198801.png')
    #path = path[0:2]
    data = []
    for i in range(len(path)):
        print(i) """
    data = []
    img = cv2.imread('./data/exp1/198801.png',0)
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            if img[j,k] == 255 :
                data.append(k)
    return data

data = data_ag()
print(data[0:50])
data = np.asarray(data)

data = np.where(data>505,data-505,505-data)

print(data[0:50])

