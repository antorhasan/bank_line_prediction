import cv2
import numpy as np
import os
import glob
from os import listdir
from os.path import isfile, join

path = './data/finaljan/'

files = [f for f in listdir(path) if isfile(join(path, f))]
#files = files[0:2]

""" only_need = glob.glob(path)

onlyfiles = [os.path.basename(x) for x in only_need]

onlyn = [os.path.splitext(x)[0] for x in onlyfiles]
 """

coor_list = [516, 491, 516, 450, 396, 355, 325, 277, 310, 400]

#img_coun = 0
for i in range(len(files)):
    print(files[i])
    img = cv2.imread(path + files[i])

    for k in range(len(coor_list)):
        num = coor_list[k]

        crop_img = img[256*k : 256*(k+1),num:num+768]
        cv2.imwrite("./data/crop1/"+files[i].split('.')[0]+str(k)+".png", crop_img)
    
    
    

    