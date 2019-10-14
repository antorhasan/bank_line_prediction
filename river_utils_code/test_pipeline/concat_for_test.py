
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import glob


# In[2]:


path = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/ere/*.jpg"
only_need = glob.glob(path)
#only_need = only_need[0:3]
#print(only_need)

path_f = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/*jpg"
n_f = glob.glob(path_f)
onlyfiles = [os.path.basename(x) for x in n_f]

onlyn = [os.path.splitext(x)[0] for x in onlyfiles]
#onlyn = onlyn[0:1]
#print(int(len(only_need)/3))
#print(onlyn)

coun = 0
nam = 0
for i in range(int(len(only_need)/3)):
    
    
    img1 = cv2.imread(only_need[i],cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(only_need[i+6],cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(only_need[i+12],cv2.IMREAD_GRAYSCALE)
    
    con = np.concatenate((img1,img2,img3), axis=1)
    
    #cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/2/"+str(coun)+"/"+str(onlyn[nam])+"_"+str(coun)+".jpg", con)
    cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/Untitled Folder/"+str(onlyn[nam])+"_"+str(coun)+".jpg", con)

    print(coun)

    coun+=1
    if coun==24:
        nam+=1
        coun=0

