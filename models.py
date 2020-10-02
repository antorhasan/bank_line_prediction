import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN_Model(nn.Module):
    def __init__(self, num_channels, batch_size, val_batch_size, time_step, num_lstm_layers, drop_out,vert_img_hgt,lf_rt_tag):
        super(CNN_Model, self).__init__()
        self.vert_img_hgt = vert_img_hgt

        if vert_img_hgt >= 3 :
            kernel_hgt = 3
        elif vert_img_hgt == 1 :
            kernel_hgt = 1

        self.device = 'cuda'
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.time_step = time_step
        self.num_lstm_layers = num_lstm_layers
        self.drop_out = drop_out
        self.lstm_output_size = 256

        self.conv1 = nn.Conv2d(num_channels,8,(kernel_hgt,3), padding=0)
        self.batch_norm10 = nn.BatchNorm2d(8)
        self.drop1 = nn.Dropout2d(p=self.drop_out[0])
        if vert_img_hgt < 5 :
            kernel_hgt = 1
        self.conv2 = nn.Conv2d(8,16,(kernel_hgt,3), padding=0)

        self.batch_norm1 = nn.BatchNorm2d(16)
        self.drop2 = nn.Dropout2d(p=self.drop_out[1])

        if vert_img_hgt < 7 :
            kernel_hgt = 1
        self.conv3 = nn.Conv2d(16,16,(kernel_hgt,3), padding=0)
        self.batch_norm20 = nn.BatchNorm2d(16)
        self.drop3 = nn.Dropout2d(p=self.drop_out[2])

        if vert_img_hgt < 9 :
            kernel_hgt = 1

        self.conv4 = nn.Conv2d(16,32,(kernel_hgt,3), padding=0)

        self.batch_norm2 = nn.BatchNorm2d(32)
        self.drop4 = nn.Dropout2d(p=self.drop_out[3])

        if vert_img_hgt < 11 :
            kernel_hgt = 1
        self.conv5 = nn.Conv2d(32,32,(kernel_hgt,3), padding=0)
        self.batch_norm30 = nn.BatchNorm2d(32)
        self.drop5 = nn.Dropout2d(p=self.drop_out[4])
        if vert_img_hgt < 13 :
            kernel_hgt = 1
        self.conv6 = nn.Conv2d(32,64,(kernel_hgt,3), padding=0)

        self.batch_norm3 = nn.BatchNorm2d(64)
        self.drop6 = nn.Dropout2d(p=self.drop_out[5])
        if vert_img_hgt < 15 :
            kernel_hgt = 1
        self.conv7 = nn.Conv2d(64,64,(kernel_hgt,3), padding=0)
        self.batch_norm40 = nn.BatchNorm2d(64)
        self.drop7 = nn.Dropout2d(p=self.drop_out[6])
        if vert_img_hgt < 17 :
            kernel_hgt = 1
        self.conv8 = nn.Conv2d(64,128,(kernel_hgt,3), padding=0)

        self.batch_norm4 = nn.BatchNorm2d(128)
        self.drop8 = nn.Dropout2d(p=self.drop_out[7])
        if vert_img_hgt < 19 :
            kernel_hgt = 1
        self.conv9 = nn.Conv2d(128,128,(kernel_hgt,3), padding=0)
        self.batch_norm50 = nn.BatchNorm2d(128)
        self.drop9 = nn.Dropout2d(p=self.drop_out[8])
        if vert_img_hgt < 21 :
            kernel_hgt = 1
        self.conv10 = nn.Conv2d(128,256,(kernel_hgt,3), padding=0)

        self.batch_norm5 = nn.BatchNorm2d(256)
        self.drop10 = nn.Dropout2d(p=self.drop_out[9])

        self.lstm = nn.LSTM(self.lstm_output_size,self.lstm_output_size,num_layers=num_lstm_layers,batch_first=True,dropout=0.2)
        self.dropout1 = nn.Dropout(self.drop_out[10])
        self.fc1 = nn.Linear(self.lstm_output_size, int(self.lstm_output_size/2))
        self.dropout2 = nn.Dropout(self.drop_out[11])

        if lf_rt_tag == 'left' or lf_rt_tag == 'right' :
            output_num = 1
        elif lf_rt_tag == 'both' :
            output_num = 2

        self.fc2 = nn.Linear(int(self.lstm_output_size/2), output_num)
        

    def forward(self, inputs):
        max_vert_kr = 2
        #inputs = torch.reshape(inputs, ((self.time_step-1)*self.batch_size, self.num_channels, 1, 745))
        inputs = torch.reshape(inputs, (-1, self.num_channels, self.vert_img_hgt, 745))
        x = F.relu(self.conv1(inputs))
        #print(x.size())
        x = self.batch_norm10(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = self.batch_norm1(x)
        x = self.drop2(x)

        if self.vert_img_hgt < 15 :
            max_vert_kr = 1
        
        x = F.max_pool2d(x, (1, 3)) 
        
        x = F.relu(self.conv3(x))
        #print(x.size())
        x = self.batch_norm20(x)
        x = self.drop3(x)
        x = F.relu(self.conv4(x))
        #print(x.size())
        x = self.batch_norm2(x)
        x = self.drop4(x)
        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv5(x))
        #print(x.size())
        x = self.batch_norm30(x)
        x = self.drop5(x)
        x = F.relu(self.conv6(x))
        #print(x.size())
        x = self.batch_norm3(x)
        x = self.drop6(x)

        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv7(x))
        #print(x.size())
        x = self.batch_norm40(x)
        x = self.drop7(x)
        x = F.relu(self.conv8(x))
        #print(x.size())
        x = self.batch_norm4(x)
        x = self.drop8(x)

        x = F.max_pool2d(x, (1, 3))

        x = F.relu(self.conv9(x))
        #print(x.size())
        x = self.batch_norm50(x)
        x = self.drop9(x)
        x = F.relu(self.conv10(x))

        #print(x.size())
        #print(asd)


        x = self.batch_norm5(x)
        x = self.drop10(x)
        x = F.max_pool2d(x, (1, 3))
        
        x = torch.reshape(x, (-1,(self.time_step-1),256))
            
        if self.training:
            
            h0 = torch.zeros((self.num_lstm_layers, self.batch_size, self.lstm_output_size),device=self.device)
            c0 = torch.zeros((self.num_lstm_layers, self.batch_size, self.lstm_output_size),device=self.device)
        else:
            h0 = torch.zeros((self.num_lstm_layers, self.val_batch_size, self.lstm_output_size),device=self.device)
            c0 = torch.zeros((self.num_lstm_layers, self.val_batch_size, self.lstm_output_size),device=self.device)

        #print(x.size())
        #print(asd)
        _, (hn, _) = self.lstm(x, (h0, c0))
        hn = hn[-1,:,:]
        #print(hn.size())
        #print(asd)
        if self.training:
            hn = torch.reshape(hn, (self.batch_size,256))
        else:
            hn = torch.reshape(hn, (self.val_batch_size,256))
        
        x = self.dropout1(hn)
        x = self.fc1(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x



class Baseline_LSTM_Model(nn.Module):
    def __init__(self, num_channels, batch_size, val_batch_size, time_step, num_lstm_layers, drop_out,vert_img_hgt,
                    inp_lr_flag, lf_rt_tag, lstm_hidden_units):
        super(Baseline_LSTM_Model, self).__init__()
        self.vert_img_hgt = vert_img_hgt

        self.lstm_dropout = 0
        self.device = 'cuda'
        self.batch_size = batch_size
        self.val_batch_size = 1
        self.time_step = time_step
        self.num_lstm_layers = num_lstm_layers
        self.drop_out = drop_out
        self.lstm_hidden_units = lstm_hidden_units
        self.flag_batch_norm = True
        
        
        #self.conv1 = nn.Conv2d(1,1,(3,1), padding=0)
        #self.conv2 = nn.Conv2d(2,4,(3,1), padding=0)
        #self.conv3 = nn.Conv2d(4,6,(3,1), padding=0)
        #self.batch_norm1 = nn.BatchNorm2d(1)
        #self.fc0 = nn.Linear(self.lstm_hidden_units, self.fc1_units)

        if inp_lr_flag == 'left' or inp_lr_flag == 'right' :
            self.inp_num = 1
        elif inp_lr_flag == 'both' :
            self.inp_num = 2

        self.lstm = nn.LSTM( ((vert_img_hgt *self.inp_num)+1), self.lstm_hidden_units, num_layers=num_lstm_layers,dropout=self.lstm_dropout,batch_first=True)
        
        if lf_rt_tag == 'left' or lf_rt_tag == 'right' :
            output_num = 1
        elif lf_rt_tag == 'both' :
            output_num = 2
        
        
        self.fc2 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm2 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc3 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm3 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc4 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm4 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc5 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm5 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc6 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm6 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc7 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm7 = nn.BatchNorm1d(self.lstm_hidden_units)

        #self.dropout1 = nn.Dropout(self.drop_out[10])
        self.fc1 = nn.Linear(self.lstm_hidden_units,(vert_img_hgt *output_num))

        #self.dropout2 = nn.Dropout(self.drop_out[11])
        #self.fc2 = nn.Linear(int(self.lstm_hidden_units/2), (vert_img_hgt *output_num))
        #self.fc3 = nn.Linear(100,100)
        #self.fc4 = nn.Linear(100,output_num)

    def forward(self, inputs):
        #print(inputs)
        reach_info = inputs[:,-1:]
        #print(reach_info)
        inputs = inputs[:,:-1]

        #x = torch.reshape(x, (-1,(self.time_step-1),int(lstm_num_features) ))
        x = torch.reshape(inputs, (-1,(self.time_step-1),int(self.vert_img_hgt * self.inp_num)))
        reach_info = torch.reshape(reach_info, (-1,1,1))
        reach_info = reach_info.expand(-1,(self.time_step-1),1)
        x = torch.cat((x,reach_info), 2)
        """ print(x)
        print(reach_info)
        print(x.size())
        print(asd) """
        """ if self.training:
            
            h0 = torch.zeros((self.num_lstm_layers, self.batch_size, self.lstm_hidden_units),device=self.device)
            c0 = torch.zeros((self.num_lstm_layers, self.batch_size, self.lstm_hidden_units),device=self.device)
        else:
            h0 = torch.zeros((self.num_lstm_layers, self.val_batch_size, self.lstm_hidden_units),device=self.device)
            c0 = torch.zeros((self.num_lstm_layers, self.val_batch_size, self.lstm_hidden_units),device=self.device) """

        
        _, (x, _) = self.lstm(x)
        x = x[-1,:,:]
        #print(hn.size())
        #output = output[:,-1,:]
        #print(output)
        #print(asd)
        #print(x.size())
        #print(asd)
        if self.training:
            x = torch.reshape(x, (-1, self.lstm_hidden_units))
        else:
            x = torch.reshape(x, (-1, self.lstm_hidden_units))
        #print(x.size())
        #print(asd)
        x = self.fc2(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm2(x)
        x = F.relu(x)

        x = self.fc3(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm3(x)
        x = F.relu(x)

        x = self.fc4(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm4(x)
        x = F.relu(x)

        x = self.fc5(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm5(x)
        x = F.relu(x)

        x = self.fc6(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm6(x)
        x = F.relu(x)

        x = self.fc7(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm7(x)
        x = F.relu(x)
        #x = self.dropout1(hn)
        x = self.fc1(x)
        #print(x.size())
        #x = self.dropout2(x)
        #x = self.fc2(x)
        #x = F.relu(self.fc3(x))
        #x = self.fc4(x)
        #print(asd)
        return x


class Baseline_ANN_Model(nn.Module):
    def __init__(self, num_channels, batch_size, val_batch_size, time_step, num_lstm_layers, drop_out,vert_img_hgt,
                    inp_lr_flag, lf_rt_tag, lstm_hidden_units, flag_reach_use):
        super(Baseline_ANN_Model, self).__init__()
        self.vert_img_hgt = vert_img_hgt

        self.lstm_dropout = 0
        self.device = 'cuda'
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.time_step = time_step
        self.num_lstm_layers = num_lstm_layers
        self.drop_out = drop_out
        self.lstm_hidden_units = lstm_hidden_units
        self.flag_batch_norm = True
        
        #self.fc0 = nn.Linear(self.lstm_hidden_units, self.fc1_units)

        if inp_lr_flag == 'left' or inp_lr_flag == 'right' :
            self.inp_num = 1
            #if flag_reach_use == True :
            #    self.inp_num = 2
        elif inp_lr_flag == 'both' :
            self.inp_num = 2
            #if flag_reach_use == True :
            #    self.inp_num = 3

        #self.lstm = nn.LSTM( (vert_img_hgt * self.inp_num), self.lstm_hidden_units, num_layers=num_lstm_layers,dropout=self.lstm_dropout,batch_first=True)
        
        if lf_rt_tag == 'left' or lf_rt_tag == 'right' :
            output_num = 1
        elif lf_rt_tag == 'both' :
            output_num = 2
        
        if flag_reach_use == True :
            if vert_img_hgt > 1 :
                input_layer_size = int((self.vert_img_hgt * ((self.time_step - 1) * self.inp_num))+1)
            elif vert_img_hgt == 1 :
                input_layer_size = int(self.vert_img_hgt * (((self.time_step - 1) * self.inp_num) + 1))
        elif flag_reach_use == False :
            input_layer_size = int(self.vert_img_hgt * ((self.time_step - 1) * self.inp_num))
        #self.dropout1 = nn.Dropout(self.drop_out[10])
        #print(int((self.vert_img_hgt * self.inp_num) * (self.time_step-1)))
        #print(asd)

        self.fc1 = nn.Linear(input_layer_size, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm1 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc2 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm2 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc3 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm3 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc4 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm4 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc5 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm5 = nn.BatchNorm1d(self.lstm_hidden_units)

        """ self.fc6 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm6 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc7 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm7 = nn.BatchNorm1d(self.lstm_hidden_units) """

        """ self.fc8 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm8 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc9 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm9 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc10 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm10 = nn.BatchNorm1d(self.lstm_hidden_units) """
        
        """ self.fc11 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm11 = nn.BatchNorm1d(self.lstm_hidden_units) 

        self.fc12 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm12 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc13 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm13 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc14 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm14 = nn.BatchNorm1d(self.lstm_hidden_units) """
        
        """ self.fc15 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm15 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc16 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm16 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc17 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm17 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc18 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm18 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc19 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm19 = nn.BatchNorm1d(self.lstm_hidden_units) """

        """ self.fc20 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm20 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc21 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm21 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc22 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm22 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc23 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm23 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc24 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm24 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc25 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm25 = nn.BatchNorm1d(self.lstm_hidden_units) """
        
        """ self.fc26 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm26 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc27 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm27 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc28 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm28 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc29 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm29 = nn.BatchNorm1d(self.lstm_hidden_units) """

        """ self.fc30 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm30 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc31 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm31 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc32 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm32 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc33 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm33 = nn.BatchNorm1d(self.lstm_hidden_units)

        self.fc34 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm34 = nn.BatchNorm1d(self.lstm_hidden_units) """

        """ self.fc35 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm35 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc36 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm36 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc37 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm37 = nn.BatchNorm1d(self.lstm_hidden_units)
        
        self.fc38 = nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm38 = nn.BatchNorm1d(self.lstm_hidden_units) """


        self.fc39 = nn.Linear(self.lstm_hidden_units,(self.vert_img_hgt *output_num))
        #self.fc4 = nn.Linear(100,output_num)

    def forward(self, x):
        #x = torch.reshape(inputs, (-1,int(self.vert_img_hgt * self.inp_num * (self.time_step-1))))
        #print(x.size())
        #print(asd)
        """ if self.training:
            h0 = torch.zeros((self.num_lstm_layers, self.batch_size, self.lstm_hidden_units),device=self.device)
            c0 = torch.zeros((self.num_lstm_layers, self.batch_size, self.lstm_hidden_units),device=self.device)
        else:
            h0 = torch.zeros((self.num_lstm_layers, self.val_batch_size, self.lstm_hidden_units),device=self.device)
            c0 = torch.zeros((self.num_lstm_layers, self.val_batch_size, self.lstm_hidden_units),device=self.device) """

        
        #_, (hn, _) = self.lstm(x, (h0, c0))
        #hn = hn[-1,:,:]
        #print(hn.size())
        #output = output[:,-1,:]
        #print(output)
        #print(asd)
        #if self.training:
        #    x = torch.reshape(hn, (self.batch_size, self.lstm_hidden_units))
        #else:
        #    x = torch.reshape(hn, (self.val_batch_size, self.lstm_hidden_units))
        #print(x.size())
        #print(asd)
        #x = self.dropout1(hn)
        x = self.fc1(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm1(x)
        x = F.relu(x)
        #print(x.size())
        #x = self.dropout2(x)
        x = self.fc2(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm2(x)
        x = F.relu(x)

        x = self.fc3(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm3(x)
        x = F.relu(x)

        x = self.fc4(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm4(x)
        x = F.relu(x)

        x = self.fc5(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm5(x)
        x = F.relu(x)

        """ x = self.fc6(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm6(x)
        x = F.relu(x)

        x = self.fc7(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm7(x)
        x = F.relu(x) """

        """ x = self.fc8(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm8(x)
        x = F.relu(x)

        x = self.fc9(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm9(x)
        x = F.relu(x)

        x = self.fc10(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm10(x)
        x = F.relu(x) """

        """ x = self.fc11(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm11(x)
        x = F.relu(x) 

        x = self.fc12(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm12(x)
        x = F.relu(x)

        x = self.fc13(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm13(x)
        x = F.relu(x)

        x = self.fc14(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm14(x)
        x = F.relu(x)

        x = self.fc15(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm15(x)
        x = F.relu(x) """

        """ x = self.fc16(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm16(x)
        x = F.relu(x)

        x = self.fc17(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm17(x)
        x = F.relu(x)

        x = self.fc18(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm18(x)
        x = F.relu(x)

        x = self.fc19(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm19(x)
        x = F.relu(x)

        x = self.fc20(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm20(x)
        x = F.relu(x) """

        """ x = self.fc21(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm21(x)
        x = F.relu(x)

        x = self.fc22(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm22(x)
        x = F.relu(x)

        x = self.fc23(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm23(x)
        x = F.relu(x)

        x = self.fc24(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm24(x)
        x = F.relu(x)

        x = self.fc25(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm25(x)
        x = F.relu(x) """

        """ x = self.fc26(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm26(x)
        x = F.relu(x)

        x = self.fc27(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm27(x)
        x = F.relu(x)

        x = self.fc28(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm28(x)
        x = F.relu(x)

        x = self.fc29(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm29(x)
        x = F.relu(x) """

        """ x = self.fc30(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm30(x)
        x = F.relu(x)

        x = self.fc31(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm31(x)
        x = F.relu(x)

        x = self.fc32(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm32(x)
        x = F.relu(x)

        x = self.fc33(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm33(x)
        x = F.relu(x)

        x = self.fc34(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm34(x)
        x = F.relu(x) """

        """ x = self.fc35(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm35(x)
        x = F.relu(x)

        x = self.fc36(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm36(x)
        x = F.relu(x)

        x = self.fc37(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm37(x)
        x = F.relu(x)

        x = self.fc38(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm38(x)
        x = F.relu(x) """


        x = self.fc39(x)
        #x = self.fc4(x)
        #print(asd)
        return x





