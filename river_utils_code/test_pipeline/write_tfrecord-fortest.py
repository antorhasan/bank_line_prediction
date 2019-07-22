
# coding: utf-8

# In[ ]:


from random import shuffle
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf
import pickle
import os


# In[ ]:


'''trainY = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/mini_gt/*.jpg"
trainY_list = glob.glob(trainY)
onlyfiles = [os.path.basename(x) for x in trainY_list]

onlyn = [os.path.splitext(x)[0] for x in onlyfiles]
print(onlyn)'''


# In[ ]:


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# In[ ]:



def createDataRecord(out_filename, addrs_y, addrs_m):
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs_y)):
        print(i)
       
        img_y = np.asarray(cv2.imread(addrs_y[i], cv2.IMREAD_GRAYSCALE))
        img_m = np.asarray(cv2.imread(addrs_m[i], cv2.IMREAD_GRAYSCALE)) 
        
        last_y = np.reshape(img_y, (256,256,1))
        last_y = last_y/255
       
        last_m = np.reshape(img_m,(256,256,1))
        last_m = np.where(last_m>230,1,0)
        last_m = last_m.astype(float)
        
        #last_x = np.multiply(last_y,last_m)
        
        feature = {
            'image_y': _bytes_feature(last_y.tostring()),
            'image_m': _bytes_feature(last_m.tostring())
            #'image_x': _bytes_feature(last_x.tostring())     
        }
  
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()


# In[ ]:


trainY = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/mini_gt/*.jpg"
trainY_list = glob.glob(trainY)

trainM = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/mini_msk/*.jpg"
trainM_list = glob.glob(trainM)

createDataRecord("/media/antor/Files/ML/Papers/test.tfrecords", trainY_list, trainM_list)
    

