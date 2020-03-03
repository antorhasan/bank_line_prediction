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


num_channels = 6
batch_size = 1
EPOCHS = 61
lr_rate = .00001
in_seq_num = 29
val_batch_size = 1
output_at = 10
model_type = 'CNN_Model'


hyperparameter_defaults = dict(
    dropout = 0.2,
    num_channels = 6,
    batch_size = 1,
    learning_rate = lr_rate,
    epochs = EPOCHS,
    )

# WandB – Initialize a new run
wandb.init(entity="antor", project="bank_line", config=hyperparameter_defaults)


# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config

config.update({'dataset':'rgb+infra','model_type':model_type}) 

""" config.batch_size = batch_size          # input batch size for training (default: 64)
config.test_batch_size = val_batch_size    # input batch size for testing (default: 1000)
config.epochs = EPOCHS             # number of epochs to train (default: 10)
config.lr = .01          # learning rate (default: 0.01)
#config.momentum = 0.1          # SGD momentum (default: 0.5) 
#config.no_cuda = False         # disables CUDA training
#config.seed = 42               # random seed (default: 42)
config.update({'dataset':'rgb+infra','model_type':model_type}) 
config.log_interval = output_at     # how many batches to wait before logging training status
#config.update(allow_val_change=True)
 """
model = CNN_Model()

dataset = tf.data.TFRecordDataset('./data/tfrecord/pix_img_all.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(200)
dataset = dataset.batch(batch_size, drop_remainder=True)

dataset_val = tf.data.TFRecordDataset('./data/tfrecord/pix_img_all.tfrecords')
dataset_val = dataset_val.map(_parse_function)
dataset_val = dataset_val.batch(val_batch_size, drop_remainder=True)


use_cuda = not False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

#writer = SummaryWriter(log_dir='./data/runs/')

#test_list = []

#checkpoint = torch.load('./data/model/cnn.pt')
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

wandb.watch(model, log="all")
offset = 385

for epoch in range(EPOCHS):
    model.train()
    counter = 0
    epoch_loss = 0
    for img, msk in dataset:
        #print(img.shape)
        #print(msk.size())
        #print(asd)
        img = np.reshape(img, (batch_size,32,745,num_channels))
        msk = np.reshape(msk, (batch_size,32,2))
        #print(img.shape)
        #print(msk.shape)
        #print(asd)

        img = img[:,0:in_seq_num,:,:]
        msk = msk[:,in_seq_num:in_seq_num+1,:]
        
        msk = (msk - msk_mean) / msk_std

        img = torch.Tensor(img).cuda()
        msk = torch.Tensor(msk).cuda()
        msk = torch.reshape(msk, (batch_size,-1))

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
    wandb.log({"Train Loss": avg_epoch_loss})
    #writer.add_scalar('Loss/train', avg_epoch_loss, epoch+1)

    """ if epoch % 20 == 0:
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, './data/model/wandb.pt')  """
        


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
            img = np.reshape(img, (val_batch_size,32,745,num_channels))
            msk = np.reshape(msk, (val_batch_size,32,2))
            img = img[:,1:in_seq_num+1,:,:]
            msk = msk[:,in_seq_num+1:in_seq_num+2,:]
            
            msk = (msk - msk_mean) / msk_std

            img = torch.Tensor(img).cuda()
            msk = torch.Tensor(msk).cuda()
            msk = torch.reshape(msk, (val_batch_size,-1))

            pred = model(img)
            loss = F.mse_loss(pred, msk)
            val_epoch_loss = val_epoch_loss + loss
            #print( epoch+1 % 3)
            
            if (epoch+2) % output_at == 0:
                msk = msk.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()

                pred = (msk_std * pred) + msk_mean
                msk = (msk_std * msk) + msk_mean

                #line = np.zeros((val_batch_size,745,3))
                for k in range(val_batch_size):
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
        #output = np.reshape(output, (len(test_list)*val_batch_size,745,3))
        
        #cv2.imwrite('./data/output/' + str(epoch+1) +'.png',output)
        gt_list = np.asarray(gt_list)
        pred_list = np.asarray(pred_list)

        mae = mean_absolute_error(gt_list,pred_list)
        std = np.std(pred_list)


        cv2.imwrite('./data/output/' + str(epoch+1) +'_tst.png',val_image)
        wandb.log({"examples" : wandb.Image(val_image)})
        wandb.log({"mean_abs_error": mae,"standard_deviation": std})
        
        #test_list = []

    avg_val_epoch_loss = val_epoch_loss / count
    template = 'Epoch {}, Val Loss: {}'
    print(template.format(epoch+1,avg_val_epoch_loss))
    wandb.log({"val_loss": avg_val_epoch_loss})
    #writer.add_scalar('Loss/Val', avg_val_epoch_loss, epoch+1)

wandb.save('./data/model/wan_model.h5')









