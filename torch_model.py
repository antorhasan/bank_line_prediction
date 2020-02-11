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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(745*3,20,num_layers=1,batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(20, 2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, inputs):
        x = torch.reshape(inputs, (batch_size,in_seq_num,-1))
        h0 = torch.zeros(1, batch_size, 20).cuda()
        c0 = torch.zeros(1, batch_size, 20).cuda()
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = x[:,-1,:]
        #print(x.size())
        #print(asd)
        x = self.dropout1(x)
        x = self.fc1(x)
        #x = self.dropout2(x)
        #print(x.size())
        #print(asd)
        return x

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3,8,(1,3), padding=0)
        self.conv2 = nn.Conv2d(8,16,(1,3), padding=0)
    
        self.conv3 = nn.Conv2d(16,16,(1,3), padding=0)
        self.conv4 = nn.Conv2d(16,32,(1,3), padding=0)

        self.conv5 = nn.Conv2d(32,32,(1,3), padding=0)
        self.conv6 = nn.Conv2d(32,64,(1,3), padding=0)

        self.conv7 = nn.Conv2d(64,64,(1,3), padding=0)
        self.conv8 = nn.Conv2d(64,128,(1,3), padding=0)

        self.conv9 = nn.Conv2d(128,128,(1,3), padding=0)
        self.conv10 = nn.Conv2d(128,256,(1,3), padding=0)

        self.lstm = nn.LSTM(256,20,num_layers=1,batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(20, 2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, inputs):
        inputs = torch.reshape(inputs, (29, 3, 1, 745))
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, (1, 3))
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.max_pool2d(x, (1, 3))
        
        #print(x.size())
        #print(asd)

        x = torch.reshape(x, (batch_size,in_seq_num,-1))
        h0 = torch.zeros(1, batch_size, 20).cuda()
        c0 = torch.zeros(1, batch_size, 20).cuda()
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = x[:,-1,:]
        #print(x.size())
        #print(asd)
        #x = self.dropout1(x)
        x = self.fc1(x)
        #x = self.dropout2(x)
        #print(x.size())
        #print(asd)
        return x

# WandB – Initialize a new run
wandb.init(entity="antor", project="bank_line")
#wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 1          # input batch size for training (default: 64)
config.test_batch_size = 1    # input batch size for testing (default: 1000)
config.epochs = 60             # number of epochs to train (default: 10)
config.lr = 0.00001               # learning rate (default: 0.01)
#config.momentum = 0.1          # SGD momentum (default: 0.5) 
#config.no_cuda = False         # disables CUDA training
#config.seed = 42               # random seed (default: 42)
config.log_interval = 10     # how many batches to wait before logging training status

batch_size = 1
EPOCHS = 60
lr_rate = .00001
in_seq_num = 29
val_batch_size = 1
output_at = 10
model = CNN_Model()

dataset = tf.data.TFRecordDataset('./data/tfrecord/pix_img.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(200)
dataset = dataset.batch(batch_size, drop_remainder=True)

dataset_val = tf.data.TFRecordDataset('./data/tfrecord/pix_img.tfrecords')
dataset_val = dataset_val.map(_parse_function)
dataset_val = dataset_val.batch(val_batch_size, drop_remainder=True)


use_cuda = not False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

writer = SummaryWriter(log_dir='./data/runs/')

test_list = []

checkpoint = torch.load('./data/model/cnn.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

wandb.watch(model, log="all")

for epoch in range(EPOCHS):
    model.train()
    counter = 0
    epoch_loss = 0
    for img, msk in dataset:
        #print(img.shape)
        #print(msk.size())
        #print(asd)
        img = np.reshape(img, (batch_size,32,745,3))
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
    writer.add_scalar('Loss/train', avg_epoch_loss, epoch+1)

    if epoch % 20 == 0:
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, './data/model/wandb.pt') 
        


    model.eval()
    val_epoch_loss = 0
    count = 0
    
    with torch.no_grad():
        for img, msk in dataset_val:
            img = np.reshape(img, (val_batch_size,32,745,3))
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

                line = np.zeros((val_batch_size,745,3))
                for k in range(val_batch_size):
                    line[k,int(msk[k,0]),:] = [255,255,255]
                    line[k,int(msk[k,1]),:] = [255,255,255]

                    line[k,int(pred[k,0]),:] = [0,0,255]
                    line[k,int(pred[k,1]),:] = [0,0,255]
            
                test_list.append(line)
            
            count += 1

    if (epoch+1) % output_at == 0:
        #print(len(test_list))
        output = np.asarray(test_list, dtype=np.uint8)
        output = np.reshape(output, (-1,745,3))
        #output = np.reshape(output, (len(test_list)*val_batch_size,745,3))
        cv2.imwrite('./data/output/' + str(epoch+1) +'.png',output)
        #print(output.shape)
        #print(asd)
        test_list = []

    avg_val_epoch_loss = val_epoch_loss / count
    template = 'Epoch {}, Val Loss: {}'
    print(template.format(epoch+1,avg_val_epoch_loss))
    wandb.log({"Test Loss": avg_val_epoch_loss})
    writer.add_scalar('Loss/Val', avg_val_epoch_loss, epoch+1)

wandb.save('./data/model/wan_model.h5')









