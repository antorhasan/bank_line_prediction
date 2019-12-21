import tensorflow as tf
import torch.nn as nn
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from os import listdir
from os.path import isfile, join
import sys


""" img = cv2.imread('./data/img/lines/201901.png',0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(asd)
 """
path = './data/img/finaljan/'
data_list = [f for f in listdir(path) if isfile(join(path, f))]
for i in range(len(data_list)):
    data_list[i] = int(data_list[i].split('.')[0])
data_list.sort()
#print(data_list)
#print(asd)
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_data():
    writer = tf.io.TFRecordWriter('./data/tfrecord/'+ 'pix_img' +'.tfrecords')
    for i in range(2047):
        print(i)
        seq_list = []
        msk_list = []
        for j in range(32):
            img = cv2.imread(path + str(data_list[j]) + '.png', 1)
            img = img[1:2048,385:1130,:]
            img = img[i,:,:]
            img = img/255
            seq_list.append(img)

            msk = cv2.imread('./data/img/lines/' + str(data_list[j]) + '.png', 0)
            msk = msk[1:2048,385:1130]
            msk = msk[i,:]
            indx = np.argwhere(msk==255)
            indx = np.reshape(indx, (2,))
            msk_list.append(indx)
            #print(indx)
            #print(asd)

        seq_list = np.asarray(seq_list, dtype=np.float32)
        msk_list = np.asarray(msk_list, dtype=np.float32)

        feature = {
            'image': _bytes_feature(seq_list.tostring()),
            'msk': _bytes_feature(msk_list.tostring())
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


    writer.close()
    sys.stdout.flush()    


write_data()