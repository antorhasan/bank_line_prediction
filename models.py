import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN_Model(nn.Module):
    def __init__(self, num_channels, batch_size, val_batch_size, time_step, num_lstm_layers, drop_out):
        super(CNN_Model, self).__init__()

        self.device = 'cuda'
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.time_step = time_step
        self.num_lstm_layers = num_lstm_layers
        self.drop_out = drop_out
        self.lstm_output_size = 256

        self.conv1 = nn.Conv2d(num_channels,8,(1,3), padding=0)
        self.batch_norm10 = nn.BatchNorm2d(8)
        #self.drop1 = nn.Dropout2d(p=drop_out)
        self.conv2 = nn.Conv2d(8,16,(1,3), padding=0)

        self.batch_norm1 = nn.BatchNorm2d(16)
        #self.drop2 = nn.Dropout2d(p=drop_out)
    
        self.conv3 = nn.Conv2d(16,16,(1,3), padding=0)
        self.batch_norm20 = nn.BatchNorm2d(16)
        #self.drop3 = nn.Dropout2d(p=drop_out)
        self.conv4 = nn.Conv2d(16,32,(1,3), padding=0)

        self.batch_norm2 = nn.BatchNorm2d(32)
        #self.drop4 = nn.Dropout2d(p=drop_out)

        self.conv5 = nn.Conv2d(32,32,(1,3), padding=0)
        self.batch_norm30 = nn.BatchNorm2d(32)
        #self.drop5 = nn.Dropout2d(p=drop_out)
        self.conv6 = nn.Conv2d(32,64,(1,3), padding=0)

        self.batch_norm3 = nn.BatchNorm2d(64)
        #self.drop6 = nn.Dropout2d(p=drop_out)

        self.conv7 = nn.Conv2d(64,64,(1,3), padding=0)
        self.batch_norm40 = nn.BatchNorm2d(64)
        #self.drop7 = nn.Dropout2d(p=drop_out)
        self.conv8 = nn.Conv2d(64,128,(1,3), padding=0)

        self.batch_norm4 = nn.BatchNorm2d(128)
        #self.drop8 = nn.Dropout2d(p=drop_out)

        self.conv9 = nn.Conv2d(128,128,(1,3), padding=0)
        self.batch_norm50 = nn.BatchNorm2d(128)
        #self.drop9 = nn.Dropout2d(p=drop_out)
        self.conv10 = nn.Conv2d(128,256,(1,3), padding=0)

        self.batch_norm5 = nn.BatchNorm2d(256)
        #self.drop10 = nn.Dropout2d(p=drop_out)

        self.lstm = nn.LSTM(self.lstm_output_size,self.lstm_output_size,num_layers=num_lstm_layers,batch_first=True)
        #self.dropout1 = nn.Dropout(drop_out)
        self.fc1 = nn.Linear(self.lstm_output_size, int(self.lstm_output_size/2))
        #self.dropout2 = nn.Dropout(drop_out)
        self.fc2 = nn.Linear(int(self.lstm_output_size/2), 2)
        

    def forward(self, inputs):
        #inputs = torch.reshape(inputs, ((self.time_step-1)*self.batch_size, self.num_channels, 1, 745))
        inputs = torch.reshape(inputs, (-1, self.num_channels, 1, 745))
        x = F.relu(self.conv1(inputs))
        x = self.batch_norm10(x)
        #x = self.drop1(x)
        x = F.relu(self.conv2(x))

        x = self.batch_norm1(x)
        #x = self.drop2(x)
        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv3(x))
        x = self.batch_norm20(x)
        #x = self.drop3(x)
        x = F.relu(self.conv4(x))

        x = self.batch_norm2(x)
        #x = self.drop4(x)

        x = F.max_pool2d(x, (1, 3))
        
        x = F.relu(self.conv5(x))
        x = self.batch_norm30(x)
        #x = self.drop5(x)
        x = F.relu(self.conv6(x))

        x = self.batch_norm3(x)
        #x = self.drop6(x)

        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv7(x))
        x = self.batch_norm40(x)
        #x = self.drop7(x)
        x = F.relu(self.conv8(x))

        x = self.batch_norm4(x)
        #x = self.drop8(x)

        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv9(x))
        x = self.batch_norm50(x)
        #x = self.drop9(x)
        x = F.relu(self.conv10(x))

        x = self.batch_norm5(x)
        #x = self.drop10(x)
        x = F.max_pool2d(x, (1, 3))

        x = torch.reshape(x, (-1,(self.time_step-1),256))
            
        if self.training:
            
            h0 = torch.zeros((self.num_lstm_layers, self.batch_size, self.lstm_output_size),device=self.device)
            c0 = torch.zeros((self.num_lstm_layers, self.batch_size, self.lstm_output_size),device=self.device)
        else:
            h0 = torch.zeros((self.num_lstm_layers, self.val_batch_size, self.lstm_output_size),device=self.device)
            c0 = torch.zeros((self.num_lstm_layers, self.val_batch_size, self.lstm_output_size),device=self.device)

        
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        if self.training:
            hn = torch.reshape(hn, (self.batch_size,256))
        else:
            hn = torch.reshape(hn, (self.val_batch_size,256))
        
        #x = self.dropout1(hn)
        x = self.fc1(hn)

        #x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


