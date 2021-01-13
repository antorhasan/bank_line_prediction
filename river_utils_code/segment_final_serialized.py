
import cv2
import numpy as np
import glob
import os



path = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/prep_f/12/"+str(0)+"/"+"*.jpg"
dir_list = glob.glob(path)
print(dir_list)
only = [os.path.basename(x) for x in dir_list]
onl = [os.path.splitext(x)[0] for x in only]
#print(onl)
for j in range(len(onl)):
    onl[j] = onl[j][:4]
print(onl)


counter = 3600
for i in range(24):
    path = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/prep_f/12/"+str(i)+"/"+"*.jpg"
    dir_list = glob.glob(path)
    only = [os.path.basename(x) for x in dir_list]
    onl = [os.path.splitext(x)[0] for x in only]
    #print(onl)
    for j in range(len(onl)):
        onl[j] = onl[j][:4]
    print(onl)
    results = list(map(int, onl))
    results.sort()
    resul = list(map(str, results))

    #print(resul)
    coun = 0
    for i in range(len(resul)):
        resul[i] = str(coun)
        coun+=1
    #print(resul)

    #counter = 0
    for i in range(len(resul)):
        img = cv2.imread(dir_list[i] ,cv2.IMREAD_GRAYSCALE)
        cv2.imwrite("/media/antor/Files/main_projects/final_data/"+str(counter)+".jpg",img)
        counter+=1
