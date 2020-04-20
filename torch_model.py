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

msk_mean = np.load('./data/np_arr/mean.npy')
msk_std = np.load('./data/np_arr/std.npy')

def _parse_function(example_proto):

    features = {
            "image": tf.io.FixedLenFeature((), tf.string),
            "msk": tf.io.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    image = tf.io.decode_raw(parsed_features["image"],  tf.float32)
    msk = tf.io.decode_raw(parsed_features["msk"],  tf.float32)
    
    return image, msk

def _parse_function_i(example_proto):

    features = {
            "image": tf.io.FixedLenFeature((), tf.string),
            "msk": tf.io.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    image = tf.io.decode_raw(parsed_features["image"],  tf.float32)
    #msk = tf.io.decode_raw(parsed_features["msk"],  tf.float32)
    
    return image

def _parse_function_m(example_proto):

    features = {
            "image": tf.io.FixedLenFeature((), tf.string),
            "msk": tf.io.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    #image = tf.io.decode_raw(parsed_features["image"],  tf.float32)
    msk = tf.io.decode_raw(parsed_features["msk"],  tf.float32)
    
    return msk

load_mod = False
save_mod = True
#window_shft = 1
num_lstm_layers = 2
num_channels = 6
batch_size = 400
EPOCHS = 10020
lr_rate = .0001
in_seq_num = 29
#val_batch_size = 2
output_at = 10
model_type = 'CNN_Model_fix'
drop_rate = 0.3
time_step = 5

hyperparameter_defaults = dict(
    dropout = drop_rate,
    num_channels = 6,
    batch_size = batch_size,
    learning_rate = lr_rate,
    epochs = EPOCHS,
    time_step = time_step,
    num_lstm_layers = num_lstm_layers
    )

# WandB – Initialize a new run
wandb.init(entity="antor", project="bank_line", config=hyperparameter_defaults)


# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config

config.update({'dataset':'rgb+infra','model_type':model_type}) 

""" config.batch_size = batch_size          # input batch size for training (default: 64)
config.test_batch_size = batch_size    # input batch size for testing (default: 1000)
config.epochs = EPOCHS             # number of epochs to train (default: 10)
config.lr = .01          # learning rate (default: 0.01)
#config.momentum = 0.1          # SGD momentum (default: 0.5) 
#config.no_cuda = False         # disables CUDA training
#config.seed = 42               # random seed (default: 42)
config.update({'dataset':'rgb+infra','model_type':model_type}) 
config.log_interval = output_at     # how many batches to wait before logging training status
#config.update(allow_val_change=True)
 """

model = CNN_Model(num_channels, batch_size, time_step, num_lstm_layers, drop_rate)

dataseti = tf.data.TFRecordDataset('./data/tfrecord/pix_img_var.tfrecords')
dataseti = dataseti.map(_parse_function_i)
dataseti = dataseti.window(size=30, shift=32, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(30, drop_remainder=True))
dataseti = dataseti.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
dataseti = dataseti.flat_map(lambda x: x.window(size=time_step, shift=1, stride=1,drop_remainder=True))
dataseti = dataseti.flat_map(lambda x: x.batch(time_step, drop_remainder=True))
#dataseti = dataseti.batch(batch_size)
#dataseti = dataseti.map(lambda x: tf.data.Dataset.from_tensor_slices(x)) #.batch(batch_size)
#dataset = dataset.map(_parse_function)

#dataset = dataset.shuffle(2046)
#dataset = dataset.batch(batch_size, drop_remainder=True)

datasetm = tf.data.TFRecordDataset('./data/tfrecord/pix_img_var.tfrecords')
datasetm = datasetm.map(_parse_function_m)
datasetm = datasetm.window(size=30, shift=32, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(30, drop_remainder=True))
datasetm = datasetm.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
datasetm = datasetm.flat_map(lambda x: x.window(size=time_step, shift=1, stride=1,drop_remainder=True))
datasetm = datasetm.flat_map(lambda x: x.batch(time_step, drop_remainder=True))
#datasetm = datasetm.batch(batch_size)
#datasetm = datasetm.map(lambda x: tf.data.Dataset.from_tensor_slices(x)) #.batch(batch_size)


dataset = tf.data.Dataset.zip((dataseti, datasetm))
dataset = dataset.shuffle(10000)
dataset_train = dataset.batch(batch_size, drop_remainder=True)


dataset_val = tf.data.TFRecordDataset('./data/tfrecord/pix_img_all.tfrecords')
dataset_val = dataset_val.map(_parse_function)
#dataset_val = dataset.batch(batch_size, drop_remainder=True)
dataset_val = dataset_val.batch(batch_size, drop_remainder=True)


use_cuda = not False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

#writer = SummaryWriter(log_dir='./data/runs/')

#test_list = []

if load_mod == True:
    checkpoint = torch.load('./data/model/ts7_f.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

wandb.watch(model, log="all")
offset = 385

for epoch in range(EPOCHS):
    model.train()
    counter = 0
    epoch_loss = 0
    for img, msk in dataset_train:
        #print(img.shape)
        #print(msk.size())
        #print(asd)
        img = np.reshape(img, (batch_size,time_step,745,num_channels))
        msk = np.reshape(msk, (batch_size,time_step,2))
        #print(img.shape)
        #print(msk.shape)
        #print(asd)

        img = img[:,0:time_step-1,:,:]
        msk = msk[:,time_step-1:time_step,:]
        
        msk = (msk - msk_mean) / msk_std

        img = torch.Tensor(img).cuda()
        msk = torch.Tensor(msk).cuda()
        msk = torch.reshape(msk, (batch_size,-1))
        #print(img.shape)
        #print(msk.shape)

        optimizer.zero_grad()
        pred = model(img)
        loss = F.mse_loss(pred, msk)
        loss.backward()
        optimizer.step()

        epoch_loss = epoch_loss+loss
        counter += 1
        
    avg_epoch_loss = epoch_loss / counter
    template = 'Epoch {}, Train Loss: {}'
    print(template.format(epoch+1,avg_epoch_loss))
    #print('asd')
    wandb.log({"Train Loss": avg_epoch_loss})
    #writer.add_scalar('Loss/train', avg_epoch_loss, epoch+1)

    if save_mod == True :
        if epoch % 400 == 0:
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, './data/model/ts7_f.pt')
        

    model.eval()
    val_epoch_loss = 0
    count = 0
    
    with torch.no_grad():
        if (epoch+2) % output_at == 0:
            gt_list = []
            pred_list = []
            val_image = cv2.imread('./data/img/finaljan/201801.png', 1)
            val_image_counter = 1
        for img, msk in dataset_val:
            img = np.reshape(img, (batch_size,32,745,num_channels))
            msk = np.reshape(msk, (batch_size,32,2))
            img = img[:,in_seq_num-(time_step-2):in_seq_num+1,:,:]
            msk = msk[:,in_seq_num+1:in_seq_num+2,:]
            #print(img.shape)
            #print(msk.shape)
            #print(asd)
            msk = (msk - msk_mean) / msk_std

            img = torch.Tensor(img).cuda()
            msk = torch.Tensor(msk).cuda()
            msk = torch.reshape(msk, (-1,2))

            pred = model(img)
            loss = F.mse_loss(pred, msk)
            val_epoch_loss = val_epoch_loss + loss
            #print( epoch+1 % 3)
            
            if (epoch+2) % output_at == 0:
                msk = msk.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()

                pred = (msk_std * pred) + msk_mean
                msk = (msk_std * msk) + msk_mean

                #line = np.zeros((batch_size,745,3))
                for k in range(batch_size):
                    #line[k,int(msk[k,0]),:] = [255,255,255]
                    #line[k,int(msk[k,1]),:] = [255,255,255]

                    #line[k,int(pred[k,0]),:] = [0,0,255]
                    #line[k,int(pred[k,1]),:] = [0,0,255]

                    val_image[val_image_counter,int(msk[k,0])+offset,:] = [255,255,255]
                    val_image[val_image_counter,int(msk[k,1])+offset,:] = [255,255,255]
                    gt_list.append([int(msk[k,0])+offset,int(msk[k,1])+offset])

                    val_image[val_image_counter,int(pred[k,0])+offset,:] = [0,0,255]
                    val_image[val_image_counter,int(pred[k,1])+offset,:] = [0,0,255]
                    pred_list.append([int(pred[k,0])+offset,int(pred[k,1])+offset])

                    val_image_counter += 1
                #test_list.append(line)
            
            count += 1

    if (epoch+1) % output_at == 0:
        #print(len(test_list))
        #output = np.asarray(test_list, dtype=np.uint8)   to produce binary output images
        #output = np.reshape(output, (-1,745,3))
        #output = np.reshape(output, (len(test_list)*batch_size,745,3))
        
        #cv2.imwrite('./data/output/' + str(epoch+1) +'.png',output)
        gt_list = np.asarray(gt_list)
        pred_list = np.asarray(pred_list)
        gt_list_left = gt_list[:,0]
        gt_list_right = gt_list[:,1]
        pred_list_left = pred_list[:,0]
        pred_list_right = pred_list[:,1]

        mae_left = mean_absolute_error(gt_list_left,pred_list_left)
        mae_right = mean_absolute_error(gt_list_right,pred_list_right)
        std = np.std(pred_list, axis=0)


        cv2.imwrite('./data/output/' + str(epoch+1) +'_tst.png',val_image)
        #wandb.log({"examples" : wandb.Image(val_image)})
        wandb.log({"mae_left_error": mae_left,"mae_right_error": mae_right,"standard_deviation_left": std[0],"standard_deviation_right": std[1]})
        
        #test_list = []
    if (epoch+1) % 1000 == 0:
        wandb.log({"examples" : wandb.Image(val_image)})


    avg_val_epoch_loss = val_epoch_loss / count
    template = 'Epoch {}, Val Loss: {}'
    print(template.format(epoch+1,avg_val_epoch_loss))
    wandb.log({"val_loss": avg_val_epoch_loss})
    #writer.add_scalar('Loss/Val', avg_val_epoch_loss, epoch+1)

#wandb.log({"examples" : wandb.Image(val_image)})
#wandb.save('wan_model.h5')









