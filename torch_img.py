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

def write_pix_img(mode):
    '''writes data into a tf record from rgb,infra images along with labels from binary image with
    lines as two pixel for both banks per vertical length'''

    offset_left = 385
    offset_right = 1130

    path = './data/img/finaljan/'
    path_infra = './data/img/infra1/'
    data_list = [f for f in listdir(path) if isfile(join(path, f))]
    for i in range(len(data_list)):
        data_list[i] = int(data_list[i].split('.')[0])
    data_list.sort()
    #print(data_list)
    #print(asd)
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def write_data():
        writer = tf.io.TFRecordWriter('./data/tfrecord/'+ 'pix_img_var' +'.tfrecords')
        for i in range(2047):
            print(i)
            seq_list = []
            msk_list = []
            for j in range(32):
                img = cv2.imread(path + str(data_list[j]) + '.png', 1)
                img = img[1:2048,offset_left:offset_right,:]
                img = img[i,:,:]
                img = img/255

                if mode=='infra':
                    img_i = cv2.imread(path_infra + str(data_list[j]) + '.png', 1)
                    img_i = img_i[1:2048,offset_left:offset_right,:]
                    img_i = img_i[i,:,:]
                    img_i = img_i/255
                
                img = np.concatenate((img,img_i), axis=1)
                
                #seq_list.append(img)

                msk = cv2.imread('./data/img/lines/' + str(data_list[j]) + '.png', 0)
                msk = msk[1:2048,offset_left:offset_right]
                msk = msk[i,:]
                indx = np.argwhere(msk==255)
                indx = np.reshape(indx, (2,))
                #msk_list.append(indx)
                #print(indx)
                #print(asd)

                seq_list = np.asarray(img, dtype=np.float32)
                msk_list = np.asarray(indx, dtype=np.float32)

                
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

def img_bin_lable_to_tf(mode):
    '''writes data into a tf record from rgb,infra images along with labels from binary image with
    lines as two pixel for both banks per vertical length'''


    path = './data/img/final_rgb/'
    path_infra = './data/img/infra/'
    data_list = [f for f in listdir(path) if isfile(join(path, f))]
    data_list = [int(f.split('.')[0]) for f in data_list]
    data_list.sort()
    #data_list = [str(f) for f in data_list]
    print(data_list)

    print(asd)
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def write_data():
        writer = tf.io.TFRecordWriter('./data/tfrecord/'+ 'ulti_mat' +'.tfrecords')
        for i in range(50):
            print(i)
            seq_list = []
            msk_list = []
            for j in range(32):
                img = cv2.imread(path + str(data_list[j]) + '.png', 1)
                img = img[3:2051,:,:]
                img = img[i,:,:]
                img = img/255

                if mode=='infra':
                    img_i = cv2.imread(path_infra + str(data_list[j]) + '.png', 1)
                    img_i = img_i[3:2051,:,:]
                    img_i = img_i[i,:,:]
                    img_i = img_i/255
                
                img = np.concatenate((img,img_i), axis=1)
                
                #seq_list.append(img)

                msk = cv2.imread('./data/img/lines/' + str(data_list[j]) + '.png', 0)
                msk = msk[1:2048,:]
                msk = msk[i,:]
                indx = np.argwhere(msk==255)
                indx = np.reshape(indx, (2,))
                #msk_list.append(indx)
                #print(indx)
                #print(asd)

                seq_list = np.asarray(img, dtype=np.float32)
                msk_list = np.asarray(indx, dtype=np.float32)

                
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

if __name__ == "__main__" :
    #write_pix_img(mode='infra')
    img_bin_lable_to_tf(mode='infra')