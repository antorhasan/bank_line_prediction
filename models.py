import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN_Model(nn.Module):
    def __init__(self, num_channels, batch_size, in_seq_num, num_lstm_layers, drop_out):
        super(CNN_Model, self).__init__()

        self.num_channels = num_channels
        self.batch_size = batch_size
        self.in_seq = in_seq_num
        self.num_lstm_layers = num_lstm_layers
        self.drop_out = drop_out

        self.conv1 = nn.Conv2d(num_channels,8,(1,3), padding=0)
        self.batch_norm10 = nn.BatchNorm2d(8)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(8,16,(1,3), padding=0)

        self.batch_norm1 = nn.BatchNorm2d(16)
        self.drop2 = nn.Dropout2d(p=0.2)
    
        self.conv3 = nn.Conv2d(16,16,(1,3), padding=0)
        self.batch_norm20 = nn.BatchNorm2d(16)
        self.drop3 = nn.Dropout2d(p=0.2)
        self.conv4 = nn.Conv2d(16,32,(1,3), padding=0)

        self.batch_norm2 = nn.BatchNorm2d(32)
        self.drop4 = nn.Dropout2d(p=0.2)

        self.conv5 = nn.Conv2d(32,32,(1,3), padding=0)
        self.batch_norm30 = nn.BatchNorm2d(32)
        self.drop5 = nn.Dropout2d(p=0.2)
        self.conv6 = nn.Conv2d(32,64,(1,3), padding=0)

        self.batch_norm3 = nn.BatchNorm2d(64)
        self.drop6 = nn.Dropout2d(p=0.2)

        self.conv7 = nn.Conv2d(64,64,(1,3), padding=0)
        self.batch_norm40 = nn.BatchNorm2d(64)
        self.drop7 = nn.Dropout2d(p=0.2)
        self.conv8 = nn.Conv2d(64,128,(1,3), padding=0)

        self.batch_norm4 = nn.BatchNorm2d(128)
        self.drop8 = nn.Dropout2d(p=0.2)

        self.conv9 = nn.Conv2d(128,128,(1,3), padding=0)
        self.batch_norm50 = nn.BatchNorm2d(128)
        self.drop9 = nn.Dropout2d(p=0.2)
        self.conv10 = nn.Conv2d(128,256,(1,3), padding=0)

        self.batch_norm5 = nn.BatchNorm2d(256)
        self.drop10 = nn.Dropout2d(p=0.2)

        self.lstm = nn.LSTM(256,20,num_layers=num_lstm_layers,batch_first=True)
        self.dropout1 = nn.Dropout(drop_out)
        self.fc1 = nn.Linear(20, 2)
        self.dropout2 = nn.Dropout(drop_out)

    def forward(self, inputs):
        inputs = torch.reshape(inputs, (self.in_seq*self.batch_size, self.num_channels, 1, 745))
        x = F.relu(self.conv1(inputs))
        x = self.batch_norm10(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))

        x = self.batch_norm1(x)
        x = self.drop2(x)
        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv3(x))
        x = self.batch_norm20(x)
        x = self.drop3(x)
        x = F.relu(self.conv4(x))

        x = self.batch_norm2(x)
        x = self.drop4(x)

        x = F.max_pool2d(x, (1, 3))
        
        x = F.relu(self.conv5(x))
        x = self.batch_norm30(x)
        x = self.drop5(x)
        x = F.relu(self.conv6(x))

        x = self.batch_norm3(x)
        x = self.drop6(x)

        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv7(x))
        x = self.batch_norm40(x)
        x = self.drop7(x)
        x = F.relu(self.conv8(x))

        x = self.batch_norm4(x)
        x = self.drop8(x)

        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv9(x))
        x = self.batch_norm50(x)
        x = self.drop9(x)
        x = F.relu(self.conv10(x))

        x = self.batch_norm5(x)
        x = self.drop10(x)
        x = F.max_pool2d(x, (1, 3))
        
        #print(x.size())
        #print(asd)

        x = torch.reshape(x, (self.batch_size,self.in_seq,-1))
        h0 = torch.zeros(self.num_lstm_layers, self.batch_size, 20).cuda()
        c0 = torch.zeros(self.num_lstm_layers, self.batch_size, 20).cuda()
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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(745*3,20,num_layers=1,batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(20, 2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, inputs):
        x = torch.reshape(inputs, (1,29,-1))
        h0 = torch.zeros(1, 1, 20).cuda()
        c0 = torch.zeros(1, 1, 20).cuda()
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

if __name__ == "__main__" :
    pass