
# coding: utf-8

# In[17]:


import cv2
import numpy as np
import glob
import os


# In[16]:


path = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/full/*.jpg"
only_need = glob.glob(path)
print(only_need)

img1 = cv2.imread(only_need[0],cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(only_need[1],cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(only_need[2],cv2.IMREAD_GRAYSCALE)
#img4 = cv2.imread(only_need[0],cv2.IMREAD_GRAYSCALE)


new = np.mean([img1,img2,img3],axis=0)

cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/avg_fix/"+"19920115_g.jpg", new)


# In[ ]:


path = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/full/*.jpg"
only_need = glob.glob(path)
print(only_need)

img1 = cv2.imread(only_need[0],cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(only_need[1],cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(only_need[2],cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread(only_need[3],cv2.IMREAD_GRAYSCALE)


new = np.mean([img1,img2,img3,img4],axis=0)

cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/avg_fix/"+"19920115_g.jpg", new)


# In[ ]:


path = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/Untitled Folder/*.jpg"
only_need = glob.glob(path)
only = [os.path.basename(x) for x in only_need]
onl = [os.path.splitext(x)[0] for x in only]
results = list(map(int, onl))
results.sort()
resul = list(map(str, results))
print(resul)

for i in range(24):
    
    img1 = cv2.imread(only_need[i],cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(only_need[i+23],cv2.IMREAD_GRAYSCALE)
    #img3 = cv2.imread(only_need[0],cv2.IMREAD_GRAYSCALE)
    #img4 = cv2.imread(only_need[0],cv2.IMREAD_GRAYSCALE)


    new = np.mean([img1,img2],axis=0)

    cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/1/"+str(i)+"/"+"20141227 SS_1_"+str(i)+".jpg", new)

