import cv2 
import numpy as np 
from preprocess import path_sort

def trip_thin_line():
    '''trim the image 120 pixel on four sides'''
    path = path_sort('./data/bankline/')
    print(path)
    for i in range(len(path)):
        img = cv2.imread('./data/bankline/'+str(path[i])+'.png',0)
        img = img[0:-1,:]
        cv2.imwrite('./data/bankline/'+str(path[i])+'.png',img)

def singl_pix():
    '''makes sure each row of an image has only two valid pixels'''
    path = path_sort('./data/bankline/')
    path = path[0:2]
    print(path)
    for i in range(len(path)):
        print(i)
        img = cv2.imread('./data/bankline/'+str(path[i])+'.png',0)
        coor = []
        for j in range(img.shape[0]):
            lis = []
            for k in range(img.shape[1]):
                if img[j,k] == 255 :
                    lis.append(k)
            if len(lis) != 0 :
                small = min(lis)
                big = max(lis)
                if big-small>5 :
                    coor.append(tuple([j,small]))
                    coor.append(tuple([j,big]))
        new = np.zeros((img.shape[0],img.shape[1]))
        for j in coor:
            new[j[0],j[1]] = 255
        
        cv2.imwrite('./data/exp/'+str(path[i])+'.png', new)

def standar_height():
    '''crop all images so that everyone has same number of 
    valid pixels'''
    path = path_sort('./data/exp/')
    path = path[0:2]
    min_lis = []
    max_lis = []
    for i in range(len(path)):
        img = cv2.imread('./data/exp/'+str(path[i])+'.png',0)
        coor = np.where(img==255)
        min_lis.append(min(coor[0]))
        max_lis.append(max(coor[0]))
    print(min_lis,max_lis)
    print(max(min_lis),min(max_lis))
    for i in range(len(path)):
        img = cv2.imread('./data/exp/'+str(path[i])+'.png',0)
        img = img[max(min_lis):min(max_lis),:]
        cv2.imwrite('./data/exp1/'+str(path[i])+'.png',img)

def las():
    path = path_sort('./data/exp1/')
    path = path[0:2]
    for i in range(len(path)):
        img = cv2.imread('./data/exp1/'+str(path[i])+'.png',0)
        coor = np.where(img == 255)
        lis = list(zip(coor[0],coor[1]))
        for j in range(len(lis)):
            #print(lis[j])
            if j==5:
                break
        print(len(coor[0]),len(coor[1]))
""" 
singl_pix()
standar_height()
las()
 """
def check_number():
    '''check how many valid 1 pixels are in each row of a mask'''
    path = path_sort('./data/exp1/')
    path = path[0:2]

    for i in range(len(path)):
        print(i)
        img = cv2.imread('./data/exp1/'+str(path[i])+'.png',0)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
        coor = []
        cu = 0
        for j in range(img.shape[0]):
            coun = 0
            for k in range(img.shape[1]):
                
                if img[j,k] == 255 :
                    cu += 1
                    
                    coun += 1
                if coun == 3 :
                    coor.append(j)
        print(cu)
        print(coor)
