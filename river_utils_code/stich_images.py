
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import os
import glob
def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])


# In[ ]:


path = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/fixed/*.jpg"
only_need = glob.glob(path)
#only_need = only_need[0:3]
#only_need.sort()
#only_need.sort(key=sortKeyFunc)
#sorted(only_need)
only = [os.path.basename(x) for x in only_need]
onl = [os.path.splitext(x)[0] for x in only]
#print(onl)

#sorted(only_need, key=lambda name: int(name))
results = list(map(int, onl))
results.sort()
resul = list(map(str, results))
print(resul)

#only = [os.path.basename(x) for x in only_need]
#onl = [os.path.splitext(x)[0] for x in only]
#print(onl)



path_f = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/*jpg"
n_f = glob.glob(path_f)
onlyfiles = [os.path.basename(x) for x in n_f]

onlyn = [os.path.splitext(x)[0] for x in onlyfiles]
#onlyn = onlyn[0:1]
print(int(len(resul)/3))
#print(onlyn)

coun = 0
nam = 0
for i in range(int(len(resul)/3)):
    
    
    img1 = cv2.imread("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/fixed/"+resul[i*3]+".jpg",cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/fixed/"+resul[(i*3)+1]+".jpg",cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/fixed/"+resul[(i*3)+2]+".jpg",cv2.IMREAD_GRAYSCALE)
    print(resul[i*3],resul[(i*3)+1],resul[(i*3)+2])
    
    con = np.concatenate((img1,img2,img3), axis=1)
    
    #cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/Untitled Folder/"+str(i)+".jpg", con)
    #cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/10/"+str(i)+"/"+"20130315g_"+str(i)+".jpg", con)
    cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/avg_fix/"+"20141211 SS_"+str(i)+".jpg", con)

    #print(coun)

    coun+=1
    if coun==24:
        nam+=1
        coun=0

