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
import wandb
from models import CNN_Model
from sklearn.metrics import mean_absolute_error

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
val_batch_size = 1
total_time_step = 33
data_div_step = 31

hyperparameter_defaults = dict(
    dropout = drop_rate,
    num_channels = num_channels,
    batch_size = batch_size,
    learning_rate = lr_rate,
    epochs = EPOCHS,
    time_step = time_step,
    num_lstm_layers = num_lstm_layers,
    total_window = total_window
    )

# WandB – Initialize a new run
wandb.init(entity="antor", project="bank_line", config=hyperparameter_defaults)


# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config

config.update({'dataset':'7_chann','model_type':model_type})

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

def val_list_update(msk, pred, val_image, gt_list, pred_list, val_image_counter):
    
    #msk = msk.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    msk = np.asarray(msk)
    msk = np.reshape(msk, (val_batch_size, -1))
    pred = (msk_std * pred) + msk_mean
    msk = (msk_std * msk) + msk_mean
    #print(msk.shape)
    #print(msk)
    
    #print(int(msk[0,0]),int(msk[4,0]))
    #line = np.zeros((batch_size,745,3))
    for k in range(val_batch_size):

        val_image[val_image_counter,int(msk[k,0]),:] = [255,255,255]
        val_image[val_image_counter,int(msk[k,1]),:] = [255,255,255]
        gt_list.append([int(msk[k,0]),int(msk[k,1])])

        val_image[val_image_counter,int(pred[k,0]),:] = [0,0,255]
        val_image[val_image_counter,int(pred[k,1]),:] = [0,0,255]
        pred_list.append([int(pred[k,0]),int(pred[k,1])])

        val_image_counter += 1

    return gt_list, pred_list, val_image, val_image_counter

def val_log(gt_list, pred_list):
    
    gt_list = np.asarray(gt_list)
    pred_list = np.asarray(pred_list)
    gt_list_left = gt_list[:,0]
    gt_list_right = gt_list[:,1]
    pred_list_left = pred_list[:,0]
    pred_list_right = pred_list[:,1]

    mae_left = mean_absolute_error(gt_list_left,pred_list_left)
    mae_right = mean_absolute_error(gt_list_right,pred_list_right)

    error_left = gt_list_left - pred_list_left
    error_right = gt_list_right - pred_list_right

    std_left = np.std(error_left)
    std_right = np.std(error_right)


    return mae_left, mae_right, std_left, std_right

def val_pass(img,msk):
    #msk = (msk - msk_mean) / msk_std
    img = torch.Tensor(img).cuda()
    msk = torch.Tensor(msk).cuda()
    msk = torch.reshape(msk, (-1,2))
    pred = model(img)
    loss = F.mse_loss(pred, msk)

    return pred, loss

for epoch in range(EPOCHS):

    model.train()
    counter = 0
    epoch_loss = 0

    for input_tensor, reg_coor, _ , year_id in dataset_f:
        #break
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
        print(loss)
        print(asd)
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
    count = 0
    
    with torch.no_grad():
        if (epoch+2) % output_at == 0:
            gt_list1 = []
            pred_list1 = []
            val_image1 = cv2.imread('./data/img/up_rgb/201901.png', 1)
            val_image_counter1 = 0

            gt_list2 = []
            pred_list2 = []
            val_image2 = cv2.imread('./data/img/up_rgb/202001.png', 1)
            val_image_counter2 = 0

            
            for img, msk, year_id in dataset_val:
                img = np.reshape(img, (-1,33,745,7))
                msk = np.reshape(msk, (val_batch_size,33,2))
                
                #print(img.shape)
                #print(msk.shape)

                img1 = img[:,in_seq_num-(time_step-2):in_seq_num+1,:,:]
                msk1 = msk[:,in_seq_num+1:in_seq_num+2,:]
                year_id1 = year_id[:,in_seq_num-(time_step-2):in_seq_num+1,:]

                img2 = img[:,in_seq_num-(time_step-2)+1:in_seq_num+2,:,:]
                msk2 = msk[:,in_seq_num+2:in_seq_num+3,:]
                year_id2 = year_id[:,in_seq_num-(time_step-2)+1:in_seq_num+2,:]

                #print(year_id1)
                #print(year_id2)
                #print(asd)
                """ print(msk1.shape)
                print(msk1,msk2)
                print(msk2.shape) """
                pred1, loss1 = val_pass(img1, msk1)
                pred2, loss2 = val_pass(img2, msk2)
                
                val_epoch_loss = val_epoch_loss + ((loss1+loss2)/2)
                #print( epoch+1 % 3)

                if (epoch+2) % output_at == 0:
                    gt_list1, pred_list1, val_image1, val_image_counter1 = val_list_update(msk1, pred1, val_image1, gt_list1, pred_list1, val_image_counter1)
                    gt_list2, pred_list2, val_image2, val_image_counter2 = val_list_update(msk2, pred2, val_image2, gt_list2, pred_list2, val_image_counter2)
                
                count += 1
    if (epoch+1) % output_at == 0:
        mae_left1, mae_right1, std_left1, std_right1 = val_log(gt_list1, pred_list1)
        mae_left2, mae_right2, std_left2, std_right2 = val_log(gt_list2, pred_list2)

        cv2.imwrite('./data/output/' + str(epoch+1) +'_18.png',val_image1)
        wandb.log({"mae_left_error_18": mae_left1,"mae_right_error_18": mae_right1,"standard_deviation_left_18": std_left1,"standard_deviation_right_18": std_right1})
        cv2.imwrite('./data/output/' + str(epoch+1) +'_19.png',val_image2)
        wandb.log({"mae_left_error_19": mae_left2,"mae_right_error_19": mae_right2,"standard_deviation_left_19": std_left2,"standard_deviation_right_19": std_right2})
    
    if (epoch+1) % 1000 == 0:
        wandb.log({"examples18" : wandb.Image(val_image1)})
        wandb.log({"examples19" : wandb.Image(val_image2)})

    if (epoch+2) % output_at == 0:
        avg_val_epoch_loss = val_epoch_loss / count
        template = 'Epoch {}, Val Loss: {}'
        print(template.format(epoch+1,avg_val_epoch_loss))
        wandb.log({"val_loss": avg_val_epoch_loss})
