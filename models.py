import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(6,8,(1,3), padding=0)
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
        inputs = torch.reshape(inputs, (29, 6, 1, 745))
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

        x = torch.reshape(x, (1,29,-1))
        h0 = torch.zeros(1, 1, 20).cuda()
        c0 = torch.zeros(1, 1, 20).cuda()
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