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

def cal_mean_torch():
    '''calculate the mean and std of torch attempted dataset'''

    def _parse_function(example_proto):

        features = {
                "image": tf.io.FixedLenFeature((), tf.string),
                "msk": tf.io.FixedLenFeature((), tf.string)
            }

        parsed_features = tf.io.parse_single_example(example_proto, features)

        image = tf.io.decode_raw(parsed_features["image"],  tf.float32)
        msk = tf.io.decode_raw(parsed_features["msk"],  tf.float32)
        
        return image, msk

    dataset = tf.data.TFRecordDataset('./data/tfrecord/pix_img.tfrecords')
    dataset = dataset.map(_parse_function)
    #dataset = dataset.shuffle(200)
    #dataset = dataset.batch(batch_size, drop_remainder=True)
    i = 0
    for img, msk in dataset:
        
        msk = np.reshape(msk, (32,2))   
        if i == 0 :
            concat = msk
        else:
            concat = np.concatenate((concat, msk), axis=0)
        i += 1

    data = np.asarray(concat)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    print(mean,std)
    np.save('./data/np_arr/mean', mean)
    np.save('./data/np_arr/std', std)

arr = np.random.randint(0,500,(4,32,2))
mean = np.load('./data/np_arr/mean.npy')

val = arr-mean
print(val.shape)