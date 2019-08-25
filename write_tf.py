from random import shuffle
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from os import walk



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def createDataRecord(out_filename, addrs_y, addrs_m):
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs_y)):
        print(addrs_y[i])
        print(addrs_m[i])

        img_y = cv2.imread(trainY + str(addrs_y[i])+'.png')
        #img_y = cv2.fastNlMeansDenoising(img_y, h=24, templateWindowSize=7, searchWindowSize=21)
        
        img_y = np.asarray(img_y)

        #img_y = np.asarray(cv2.imread(addrs_y[i], cv2.IMREAD_GRAYSCALE))
        img_m = np.asarray(cv2.imread(trainM + str(addrs_m[i])+'.png',cv2.IMREAD_GRAYSCALE)) 
        
        for j in range(3):
            imgy = img_y[:,256*j:256*(j+1)]

            imgm = img_m[:,256*j:256*(j+1)]
            #print(imgy.shape)
            imgy = np.reshape(imgy, (256,256,3))
            last_y = imgy/255
        
            imgm = np.reshape(imgm,(256,256,1))
            #imgm = np.where(imgm>230,1,0)
            #last_m = imgm.astype(float)
            last_m = imgm/255
            kernel = np.ones((3,3), np.uint8)
            last_m = cv2.dilate(last_m, kernel, iterations=1)

            """             cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', last_y)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', last_m)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
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



#trainY = "./data/crop1/"
trainY = "./data/crop1/"

""" f = []
for (dirpath, dirnames, filenames) in walk(trainY):
    f.extend(filenames)
    break """

trainY_list = [f for f in listdir(trainY) if isfile(join(trainY, f))]
#trainY_list = glob.glob(trainY)
#trainY_list = trainY_list[0:10]

#print(f)

trainM = "./data/infralabel/"
trainM_list = [f for f in listdir(trainM) if isfile(join(trainM, f))]
#trainM_list = glob.glob(trainM)
#trainM_list = trainM_list[0:10]

#shuffle(trainY_list)
#shuffle(trainM_list)

'''sort the data'''
for i in range(len(trainY_list)):
    trainY_list[i] = int(trainY_list[i].split('.')[0])
trainY_list.sort()

for j in range(len(trainM_list)):
    trainM_list[j] = int(trainM_list[j].split('.')[0])
trainM_list.sort()

train_Y = trainY_list[0:280]
train_M = trainM_list[0:280]
val_Y = trainY_list[280:300]
val_M = trainM_list[280:300]
#train_Y = train_Y

print(train_Y)
print(train_M)
#print(val_M)
#print(train_M)
createDataRecord("./data/record/train_dil_3.tfrecords", train_Y, train_M)
createDataRecord("./data/record/val_dil_3.tfrecords", val_Y, val_M)
    