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

        img_y = np.asarray(img_y)

        img_m = np.asarray(cv2.imread(trainM + str(addrs_m[i])+'.png',cv2.IMREAD_GRAYSCALE)) 
        
        for j in range(3):
            imgy = img_y[:,256*j:256*(j+1)]

            imgm = img_m[:,256*j:256*(j+1)]
            imgy = np.reshape(imgy, (256,256,3))
            last_y = imgy/255
        
            imgm = np.reshape(imgm,(256,256,1))
            last_m = imgm/255
            kernel = np.ones((3,3), np.uint8)
            last_m = cv2.dilate(last_m, kernel, iterations=1)

            feature = {
                'image_y': _bytes_feature(last_y.tostring()),
                'image_m': _bytes_feature(last_m.tostring())
            }
    
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()


def create_data():
    '''create tfrecord from rgb img and canny edged bankline mask '''

    trainY = "./data/crop1/"
    trainY_list = [f for f in listdir(trainY) if isfile(join(trainY, f))]
    trainM = "./data/infralabel/"
    trainM_list = [f for f in listdir(trainM) if isfile(join(trainM, f))]

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

    print(train_Y)
    print(train_M)
    createDataRecord("./data/record/train_dil_3.tfrecords", train_Y, train_M)
    createDataRecord("./data/record/val_dil_3.tfrecords", val_Y, val_M)
    
if __name__ == "__main__":
    
    pass