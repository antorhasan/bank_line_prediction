import tensorflow as tf
import torch.nn as nn
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from os import listdir
from os.path import isfile, join
import os
import sys
import wandb
from models import CNN_Model
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import itertools
from scipy.signal import savgol_filter


def _parse_function_(example_proto):

    features = {
            'input_tensor': tf.io.FixedLenFeature((), tf.string),
            'reg_coor': tf.io.FixedLenFeature((), tf.string),
            'bin_label': tf.io.FixedLenFeature((), tf.string),
            'year_id': tf.io.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    input_tensor = tf.io.decode_raw(parsed_features["input_tensor"],  tf.float32)
    reg_coor = tf.io.decode_raw(parsed_features["reg_coor"],  tf.float32)
    bin_label = tf.io.decode_raw(parsed_features["bin_label"],  tf.float32)
    year_id = tf.io.decode_raw(parsed_features['year_id'],  tf.float32)
    
    return input_tensor, reg_coor, bin_label, year_id

load_mod = True
save_mod = False
total_window = 30
num_lstm_layers = 1
num_channels = 7
batch_size = 100
EPOCHS = 200
lr_rate = .0001
in_seq_num = 20
output_at = 10
model_type = 'CNN_Model_fix'
drop_rate = 0.25
time_step = 5
val_batch_size = batch_size
total_time_step = 33
data_div_step = 31
num_val_img = 2
log_performance = 15    ###number of epochs after which performance metrics are calculated
model_save_at = 50     ###number of epochs after which to save model
early_stop_thresh = 30
val_img_ids = [201901, 202001]
input_str = '2019'
test_img_dict = {'2021':'34','2020':'33','2019':'32'}
input_ts = int(test_img_dict[input_str])
#print(test_img_dict['2021'])
#print(asd)

dataseti1 = tf.data.TFRecordDataset(os.path.join('./data/tfrecord/comp_tf.tfrecords'))
dataseti1 = dataseti1.window(size=total_time_step, shift=total_time_step, stride=1, drop_remainder=False)
dataseti1 = dataseti1.map(lambda x: x.skip(input_ts-time_step))

if input_ts != (total_time_step+1) :
    #print('hell')
    dataseti1 = dataseti1.map(lambda x: x.take(time_step-1))
 
dataseti1 = dataseti1.flat_map(lambda x: x)
dataseti1 = dataseti1.map(_parse_function_).batch(time_step-1)
dataseti1 = dataseti1.batch(val_batch_size, drop_remainder=True)


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(device)
model = CNN_Model(num_channels, batch_size, val_batch_size,time_step, num_lstm_layers, drop_rate)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

if load_mod == True:
    checkpoint = torch.load('./data/model/ser_mod.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

msk_mean = np.load(os.path.join('./data/mean_img/line_mean.npy'))
msk_std = np.load(os.path.join('./data/mean_img/line_std.npy'))
#print(asd)

model.eval()
with torch.no_grad():
    pred_list = []
    prev_actual_list = []

    #counter = 0
    for input_tensor, reg_coor, _ , year_id in dataseti1:
        
        input_tensor = np.reshape(input_tensor, (batch_size,time_step-1,745,num_channels))
        reg_coor = np.reshape(reg_coor, (batch_size,time_step-1,2))
        prev_time_step = reg_coor[:,time_step-2:time_step-1,:]

        input_tensor = torch.Tensor(input_tensor).cpu()
        #print(input_tensor.size())
        pred = model(input_tensor)
        prev_actual_list.append(prev_time_step)
        pred_list.append(pred.numpy())
        #print(asd)
        #counter+= 1
    
    pred_list = np.asarray(pred_list)
    total_smpls = int(pred_list.shape[0])*int(pred_list.shape[1])
    #val_num_rows = int(total_smpls/num_val_img)
    pred_list = np.resize(pred_list, (total_smpls,2))
    pred_list = (msk_std * pred_list) + msk_mean
    
    prev_actual_list = np.asarray(prev_actual_list)
    prev_actual_list = np.resize(prev_actual_list, (total_smpls,2))
    prev_actual_list = (msk_std * prev_actual_list) + msk_mean


    prev_left = prev_actual_list[:, 0]
    prev_right = prev_actual_list[:, 1]

    pred_left = pred_list[:, 0]
    pred_right = pred_list[:, 1]

    window = 99
    poly = 2
    pred_left = savgol_filter(pred_left, window, poly)
    pred_right = savgol_filter(pred_right, window, poly)

    pred_ers_lft = np.reshape(np.where(pred_left<prev_left, 1, 0),(pred_left.shape[0],1))
    pred_ers_rht = np.reshape(np.where(pred_right>prev_right, 1, 0),(pred_right.shape[0],1))

    #print(pred_list.shape)
    #print(prev_actual_list.shape)
    #print(counter)
    #print(asd)
    #num_rows = int(pred_list.shape[0])

    img = cv2.imread(os.path.join('./data/img/up_rgb/'+str(int(input_str)-1)+'01.png'), 1)
    #print(img.shape)
    for i in range(total_smpls):
        #img[i,int(actual_list[i,iter_num,0]),:] = [255,255,255]
        #img[i,int(actual_list[i,iter_num,1]),:] = [255,255,255]

        img[i,int(prev_actual_list[i,0]),:] = [255,255,255]
        img[i,int(prev_actual_list[i,1]),:] = [255,255,255]

        if pred_ers_lft[i] == 1 :
            img[i,int(pred_left[i]),:] = [0,0,255]
        if pred_ers_rht[i] == 1 :
            img[i,int(pred_right[i]),:] = [0,0,255]
    
    cv2.imwrite(os.path.join('./data/test/'+str(int(input_str)-1)+'_pred.png'), img)