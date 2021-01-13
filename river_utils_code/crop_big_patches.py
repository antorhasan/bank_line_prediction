
import cv2
import numpy as np
import os
#import matplotlib.pyplot as plt
#from os import listdir
#from os.path import isfile, join
import glob
#from os import listdir
#from os.path import isfile, join

#path = "/media/antor/Files/ML/Papers/label_gt/*.jpg"
#path = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/Untitled Folder/20111016.jpg"

path = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/only/*.jpg"

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

print(len(only_need))
img_coun = 0
for i in range(len(only_need)):
    
    img = cv2.imread(only_need[i],cv2.IMREAD_GRAYSCALE)

    for k in range(24):
        num = coor_list[k]
        
        
        
       
        #j = 24
        #u = 495

        #for j in range(1):
        #for k in range(0,3,1):
            #if j==11:
            #    crop_img = img[5577:6089, 512*k : 512*(k+1)]
            #else:
        #crop_img = img[256*j : 256*(j+1), u:u+768]
        if k==23:
            crop_img = img[5833 : 6089,num+(256*0):num+(256*(2+1))]
        else:
            crop_img = img[256*k : 256*(k+1),num+(256*0):num+(256*(2+1))]
        #crop_img = img[256*k : 256*(k+1),num:num+768]

        #cv2.namedWindow('denoise',cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('denoise',crop_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #print(i)
        #print(k)
        #cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/Untitled Folder/"+str(k)+"/"+"19890509 S"+"_"+str(k)+".jpg", crop_img)
        #cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/Untitled Folder/"+"20111016"+"_"+str(k)+".jpg", crop_img)
        cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/1/"+str(i)+"/"+"20141227 SS_1_"+str(i)+".jpg", new)
    
    img_coun+=1
    print(img_coun)
            #print(str(img_coun)+'a'+str(j)+'b'+str(k))

    
