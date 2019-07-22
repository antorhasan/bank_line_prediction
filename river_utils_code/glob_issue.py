
import cv2
import numpy as np
import glob
import os

path = "/media/antor/Files/main_projects/final_data/"+"*.jpg"
dir_list = glob.glob(path)
#print(dir_list)
only = [os.path.basename(x) for x in dir_list]
onl = [os.path.splitext(x)[0] for x in only]
#print(onl)
results = list(map(int, onl))
results.sort()
resul = list(map(str, results))
#print(resul)
#resul = resul[0:32]
print(resul)
coun = 0
cou = 0
an = 0
for i in range(len(resul)):
    img = cv2.imread("/media/antor/Files/main_projects/final_data/"+resul[i]+".jpg" ,cv2.IMREAD_GRAYSCALE)
    print(i)
    for f in range(3):
        coun=cou+(30*f)
        crop_img = img[0:256,256*f:256*(f+1)]
        cv2.imwrite("/media/antor/Files/main_projects/finally/"+str(coun)+".jpg", crop_img)
        print(coun)


        #print(coun)
    cou=cou+1
    an = an+1
    if an==30:
        cou=cou+60
        an=0
