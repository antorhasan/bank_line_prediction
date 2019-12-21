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
        """ self.conv1 = nn.Conv2d(3,8,(1,3), padding=0)
        self.conv2 = nn.Conv2d(8,16,(1,3), padding=0)

        self.conv3 = nn.Conv2d(16,16,(1,3), padding=0)
        self.conv4 = nn.Conv2d(16,32,(1,3), padding=0)

        self.conv5 = nn.Conv2d(32,32,(1,3), padding=0)
        self.conv6 = nn.Conv2d(32,64,(1,3), padding=0) """

        self.lstm = nn.LSTM(745*3,20,num_layers=1,batch_first=True)

        self.fc1 = nn.Linear(20, 2)

    def forward(self, inputs):
        """ x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(a, 2, 2)
        
        x = F.relu(self.conv3(x))
        s = F.relu(self.conv4(x))
        x = F.max_pool2d(s, 2, 2)
        
        x = F.relu(self.conv5(x))
        d = F.relu(self.conv6(x))
        x = F.max_pool2d(d, 2, 2) """
        x = torch.reshape(inputs, (batch_size,in_seq_num,-1))
        h0 = torch.zeros(1, batch_size, 20).cuda()
        c0 = torch.zeros(1, batch_size, 20).cuda()
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.fc1(x)
        x = x[:,-1,:]
        #print(x.size())
        #print(asd)
        return x



batch_size = 1
EPOCHS = 2
lr_rate = .0001
in_seq_num = 29


dataset = tf.data.TFRecordDataset('./data/tfrecord/pix_img.tfrecords')
dataset = dataset.map(_parse_function)
#dataset = dataset.shuffle(100)
dataset = dataset.batch(batch_size)

use_cuda = not False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

for epoch in range(EPOCHS):
    model.train()
    for img, msk in dataset:
        img = np.reshape(img, (batch_size,32,745,3))
        msk = np.reshape(msk, (batch_size,32,2))
        
        img = img[:,0:in_seq_num,:,:]
        msk = img[:,0:30,:]
        img = torch.Tensor(img).cuda()
        msk = torch.Tensor(msk).cuda()

        optimizer.zero_grad()
        msk = model(img)
        #loss = dice_loss(msk, mask)
        #loss.backward()
        #optimizer.step()

