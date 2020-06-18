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

os.environ["WANDB_API_KEY"] = 'local-1479f9a9f8553920b500edc5cba063a6efb261f0'
os.environ["WANDB_BASE_URL"] = "http://localhost:8080"

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



load_mod = False
save_mod = False
total_window = 30
num_lstm_layers = 1
num_channels = 7
batch_size = 100
EPOCHS = 2
lr_rate = .0001
in_seq_num = 30
output_at = 10
model_type = 'CNN_Model_fix'
drop_rate = 0.3
time_step = 5
val_batch_size = batch_size
total_time_step = 33
data_div_step = 31
num_val_img = 2
val_img_ids = [201901, 202001]

hyperparameter_defaults = dict(
    dropout = drop_rate,
    num_channels = num_channels,
    batch_size = batch_size,
    learning_rate = lr_rate,
    epochs = EPOCHS,
    time_step = time_step,
    num_lstm_layers = num_lstm_layers,
    total_window = total_window,
    dataset='7_chann',
    model_type=model_type
    )

# WandB – Initialize a new run
#wandb.init(entity="antor", project="bank_line", config=hyperparameter_defaults)
wandb.init(entity="antor_1",project="bank_line", config=hyperparameter_defaults)


# WandB – Config is a variable that holds and saves hyperparameters and inputs
#config = wandb.config          # Initialize config

#config.update({'dataset':'7_chann','model_type':model_type})

dataset_f = tf.data.TFRecordDataset('./data/tfrecord/comp_tf.tfrecords')
dataset_f = dataset_f.window(size=data_div_step, shift=total_time_step, stride=1, drop_remainder=False)
dataset_f = dataset_f.map(lambda x: x.window(size=time_step, shift=1, stride=1,drop_remainder=True))
dataset_f = dataset_f.flat_map(lambda x: x.flat_map(lambda x: x))
dataset_f = dataset_f.map(_parse_function_).batch(time_step)
dataset_f = dataset_f.shuffle(10000)
dataset_f = dataset_f.batch(batch_size, drop_remainder=True)

dataseti1 = tf.data.TFRecordDataset('./data/tfrecord/comp_tf.tfrecords')
dataseti1 = dataseti1.window(size=total_time_step, shift=total_time_step, stride=1, drop_remainder=False)
dataseti1 = dataseti1.map(lambda x: x.skip(total_time_step-(time_step+1)).window(size=time_step, shift=1, stride=1,drop_remainder=True))
dataseti1 = dataseti1.flat_map(lambda x: x.flat_map(lambda x: x))
dataseti1 = dataseti1.map(_parse_function_).batch(time_step)
dataseti1 = dataseti1.batch(val_batch_size, drop_remainder=True)

model = CNN_Model(num_channels, batch_size, val_batch_size,time_step, num_lstm_layers, drop_rate)

#use_cuda = not False and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

if load_mod == True:
    checkpoint = torch.load('./data/model/f_temp.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

wandb.watch(model, log="all")

msk_mean = np.load('./data/mean_img/line_mean.npy')
msk_std = np.load('./data/mean_img/line_std.npy')

def plt_conf_mat(conf_mat, title):
    normalize = True
    plt.figure(clear=True)
    cmap = plt.get_cmap('Blues')
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    target_names = ['non-erosion', 'erosion']
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(conf_mat[i, j]),
                     horizontalalignment="center",
                     color="orange")
        else:
            plt.text(j, i, "{:,}".format(conf_mat[i, j]),
                     horizontalalignment="center",
                     color="orange")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    wandb.log({title: plt})

def regress_erro(act_err_bin, act_reg, pred_reg, iter_num, side):
    temp_arr = pred_reg - act_reg
    if side == 'left' :
        temp_arr = temp_arr
    elif side == 'right' :
        temp_arr = -temp_arr
    counter_pos = 0
    counter_neg = 0 
    pos_deviation = 0
    neg_deviation = 0
    for i in range(act_err_bin.shape[0]):
        if act_err_bin[i] == 1 and temp_arr[i]>=0 :
            pos_deviation = pos_deviation + temp_arr[i]
            counter_pos += 1
        elif act_err_bin[i] == 1 and temp_arr[i]<0 :
            neg_deviation = neg_deviation + (-temp_arr[i])
            counter_neg += 1

    mean_pos_dev = pos_deviation/counter_pos
    mean_neg_dev = neg_deviation/counter_neg

    wandb.log({'mean_abs_pos_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]):mean_pos_dev, 
                'mean_abs_neg_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]):mean_neg_dev})

    return mean_pos_dev

def calc_fscore(iter_num, actual_list, prev_actual_list, pred_list):
    act_left = actual_list[:,iter_num,0]
    act_right = actual_list[:,iter_num,1]

    prev_left = prev_actual_list[:, iter_num, 0]
    prev_right = prev_actual_list[:, iter_num, 1]

    pred_left = pred_list[:, iter_num, 0]
    pred_right = pred_list[:, iter_num, 1]

    actual_ers_lft = np.reshape(np.where(act_left<prev_left, 1, 0),(act_left.shape[0],1))
    actual_ers_rht = np.reshape(np.where(act_right>prev_right, 1, 0),(act_right.shape[0],1))
    
    left_mae = regress_erro(actual_ers_lft, act_left, pred_left, iter_num, 'left')
    right_mae = regress_erro(actual_ers_rht, act_right, pred_right, iter_num, 'right')

    avg_mae = (left_mae + right_mae)/2
    wandb.log({'pos_mae_for_actual_erosion'+str(val_img_ids[iter_num]) : avg_mae})

    pred_ers_lft = np.reshape(np.where(pred_left<prev_left, 1, 0),(pred_left.shape[0],1))
    pred_ers_rht = np.reshape(np.where(pred_right>prev_right, 1, 0),(pred_right.shape[0],1))

    conf_mat_lft = confusion_matrix(actual_ers_lft, pred_ers_lft)
    conf_mat_rht = confusion_matrix(actual_ers_rht, pred_ers_rht)
    combined_conf = conf_mat_lft + conf_mat_rht

    plt_conf_mat(conf_mat_lft, str(val_img_ids[iter_num])+'_conf_mat_left')
    plt_conf_mat(conf_mat_rht, str(val_img_ids[iter_num])+'_conf_mat_right')
    plt_conf_mat(combined_conf, str(val_img_ids[iter_num])+'_combined_conf_mat')

    y_true = np.concatenate((actual_ers_lft,actual_ers_rht), axis = 0)
    y_pred = np.concatenate((pred_ers_lft,pred_ers_rht), axis = 0)

    precision_comb = precision_score(y_true, y_pred, average='binary')
    recall_comb = recall_score(y_true, y_pred, average='binary')
    f1_comb = f1_score(y_true, y_pred, average='binary')
    
    wandb.log({'precision_'+ str(val_img_ids[iter_num]) : precision_comb})
    wandb.log({'recall_'+ str(val_img_ids[iter_num]) : recall_comb})
    wandb.log({'f1_score_'+ str(val_img_ids[iter_num]) : f1_comb})

    return combined_conf, precision_comb, recall_comb, f1_comb



def wrt_img(iter_num, actual_list, prev_actual_list, pred_list):
        num_rows = int(pred_list.shape[0])

        img = cv2.imread('./data/img/up_rgb/'+str(val_img_ids[iter_num])+'.png', 1)
        for i in range(num_rows):
            img[i,int(actual_list[i,iter_num,0]),:] = [255,255,255]
            img[i,int(actual_list[i,iter_num,1]),:] = [255,255,255]

            img[i,int(prev_actual_list[i,iter_num,0]),:] = [255,0,0]
            img[i,int(prev_actual_list[i,iter_num,1]),:] = [255,0,0]

            img[i,int(pred_list[i,iter_num,0]),:] = [0,0,255]
            img[i,int(pred_list[i,iter_num,1]),:] = [0,0,255]
        
        cv2.imwrite('./data/output/'+str(val_img_ids[iter_num])+'_ot.png', img)

def process_val(arr_list):
    arr_list = np.asarray(arr_list)
    total_smpls = int(arr_list.shape[0])*int(arr_list.shape[1])
    val_num_rows = int(total_smpls/num_val_img)
    arr_list = np.resize(arr_list, (val_num_rows,num_val_img,2))
    arr_list = (msk_std * arr_list) + msk_mean

    return arr_list

for epoch in range(EPOCHS):

    model.train()
    counter = 0
    epoch_loss = 0

    for input_tensor, reg_coor, _ , year_id in dataset_f:

        input_tensor = np.reshape(input_tensor, (batch_size,time_step,745,num_channels))
        reg_coor = np.reshape(reg_coor, (batch_size,time_step,2))

        input_tensor = input_tensor[:,0:time_step-1,:,:]
        reg_coor = reg_coor[:,time_step-1:time_step,:]
        
        input_tensor = torch.Tensor(input_tensor).cuda()
        reg_coor = torch.Tensor(reg_coor).cuda()
        reg_coor = torch.reshape(reg_coor, (batch_size,-1))
        
        optimizer.zero_grad()
        pred = model(input_tensor)
        loss = F.mse_loss(pred, reg_coor,reduction='mean')
        loss.backward()
        optimizer.step()

        epoch_loss = epoch_loss+loss
        counter += 1
            
    avg_epoch_loss = epoch_loss / counter
    template = 'Epoch {}, Train Loss: {}'
    print(template.format(epoch+1,avg_epoch_loss))

    wandb.log({"Train Loss": avg_epoch_loss})
    #writer.add_scalar('Loss/train', avg_epoch_loss, epoch+1)

    if save_mod == True :
        if epoch % 100 == 0:
            print('saving model....')
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, './data/model/f_temp.pt')
    model.eval()

    val_epoch_loss = 0
    counter_val = 0
    
    with torch.no_grad():

        pred_list = []
        actual_list = []
        prev_actual_list = []

        for input_tensor, reg_coor, _ , year_id in dataseti1:
            #print(year_id)
            #print(asd)
            input_tensor = np.reshape(input_tensor, (batch_size,time_step,745,num_channels))
            reg_coor = np.reshape(reg_coor, (batch_size,time_step,2))

            input_tensor = input_tensor[:,0:time_step-1,:,:]
            prev_time_step = reg_coor[:,time_step-2:time_step-1,:]
            reg_coor = reg_coor[:,time_step-1:time_step,:]
            
            input_tensor = torch.Tensor(input_tensor).cuda()
            reg_coor = torch.Tensor(reg_coor).cuda()
            reg_coor = torch.reshape(reg_coor, (batch_size,-1))

            pred = model(input_tensor)
            
            loss = F.mse_loss(pred, reg_coor,reduction='mean')

            val_epoch_loss = val_epoch_loss+loss
            counter_val += 1
            
            if epoch % 20 == 19:
                print('logging performance metrics......')
                prev_actual_list.append(prev_time_step)
                reg_coor = reg_coor.cpu()
                actual_list.append(reg_coor.numpy())
                pred_np = pred.cpu()
                pred_list.append(pred_np.numpy())

        avg_val_epoch_loss = val_epoch_loss / counter_val
        template = 'Epoch {}, Val Loss: {}'
        print(template.format(epoch+1,avg_val_epoch_loss))

        wandb.log({"Val Loss": avg_val_epoch_loss})

        ###logging performance metrics
        if epoch % 20 == 19 :
            pred_list = process_val(pred_list)
            actual_list = process_val(actual_list)
            prev_actual_list = process_val(prev_actual_list)

            for i in range(num_val_img):
                wrt_img(i, actual_list, prev_actual_list, pred_list)
                temp_conf,precision_comb,recall_comb,f1_comb = calc_fscore(i, actual_list, prev_actual_list, pred_list)
                if i == 0 :
                    final_conf = temp_conf
                    final_prec = precision_comb
                    final_recall = recall_comb
                    final_f1 = f1_comb
                else :
                    final_conf = final_conf + temp_conf
                    final_prec = final_prec + precision_comb
                    final_recall = final_recall + recall_comb
                    final_f1 = final_f1 + f1_comb

            plt_conf_mat(final_conf, 'total_test_confusion_matrix')
            wandb.log({"test_set_precision":final_prec/num_val_img, "test_set_recall":final_recall/num_val_img,
                        "test_set_f1_score":final_f1/num_val_img})