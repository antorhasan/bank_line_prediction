
import cv2
import numpy as np
import os
import glob

path = "/media/antor/Files/main_projects/region_define/*.jpg"


only_need = glob.glob(path)

onlyfiles = [os.path.basename(x) for x in only_need]

onlyn = [os.path.splitext(x)[0] for x in onlyfiles]
#onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

#img_coun = 305
#print(only_need)
#print(onlyfiles)
print(onlyn)
#only_need = only_need[0:1]

coor_list = [510,440,350,275,265,285,250,215,105,125,260,345,
            510,570,640,700,725,660,635,600,530,510,470,495]
'''495'''

img = cv2.imread(only_need[0], cv2.IMREAD_COLOR)

for i in range(len(coor_list)):
    if i==23:
        for j in range(5):
            img[5833+j,coor_list[i]:coor_list[i]+768] = [0,0,255]
            img[6088-j,coor_list[i]:coor_list[i]+768] = [0,0,255]
            img[5833:6088,coor_list[i]+j] = [0,0,255]
            img[5833:6088,coor_list[i]+768-j] = [0,0,255]
    
    else:
        for j in range(5):
            img[(256*i)+j,coor_list[i]:coor_list[i]+768] = [0,0,255]
            img[256*(i+1)-j,coor_list[i]:coor_list[i]+768] = [0,0,255]
            img[256*i:256*(i+1),coor_list[i]+j] = [0,0,255]
            img[256*i:256*(i+1),coor_list[i]+768-j] = [0,0,255]
    
    
cv2.imwrite("/media/antor/Files/ML/Untitled Folder/"+"trial5.jpg", img)
    
    
    
    
    
