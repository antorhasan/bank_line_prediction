import cv2
import numpy as np
import os
import glob
from os import listdir
from os.path import isfile, join

path = './data/finaljan/'

files = [f for f in listdir(path) if isfile(join(path, f))]
files = files[0:1]

""" only_need = glob.glob(path)

onlyfiles = [os.path.basename(x) for x in only_need]

onlyn = [os.path.splitext(x)[0] for x in onlyfiles]
 """

coor_list = [510,440,350,275,265,285,250,215,105,125,260,345,
            510,570,640,700,725,660,635,600,530,510,470,495]

print(len(only_need))
img_coun = 0
for i in range(len(only_need)):
    
    img = cv2.imread(only_need[i],cv2.IMREAD_GRAYSCALE)

    for k in range(24):
        num = coor_list[k]
        
    
        if k==23:
            crop_img = img[5833 : 6089,num+(256*0):num+(256*(2+1))]
        else:
            crop_img = img[256*k : 256*(k+1),num+(256*0):num+(256*(2+1))]
        cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/1/"+str(i)+"/"+"20141227 SS_1_"+str(i)+".jpg", new)
    
    img_coun+=1
    print(img_coun)

    