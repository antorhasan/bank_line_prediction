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

class Baseline_ANN_Dynamic_Model(nn.Module):
    def __init__(self, num_channels, batch_size, val_batch_size, time_step, num_lstm_layers, drop_out,vert_img_hgt,
                    inp_lr_flag, lf_rt_tag, lstm_hidden_units, flag_reach_use, num_layers,out_use_mid):

        super(Baseline_ANN_Dynamic_Model, self).__init__()
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
        self.num_layers = num_layers

        if inp_lr_flag == 'left' or inp_lr_flag == 'right' :
            self.inp_num = 1
        elif inp_lr_flag == 'both' :
            self.inp_num = 2


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


        self.fc_in = nn.Linear(input_layer_size, self.lstm_hidden_units)
        if self.flag_batch_norm == True :
            self.batch_norm1 = nn.BatchNorm1d(self.lstm_hidden_units)

        if num_layers > 0 :

            self.linear_batchnm = nn.ModuleDict({})

            for i in range(num_layers) :
                if self.flag_batch_norm == True :
                    self.linear_batchnm.update([
                        ['fc_'+str(i+2), nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)],
                        ['batch_norm_'+str(i+2), nn.BatchNorm1d(self.lstm_hidden_units)]
                    ])
                
                elif self.flag_batch_norm == False :
                    self.linear_batchnm.update([
                        ['fc_'+str(i+2), nn.Linear(self.lstm_hidden_units, self.lstm_hidden_units)],
                    ])
        
        if out_use_mid == False :
            self.fc_out = nn.Linear(self.lstm_hidden_units,(self.vert_img_hgt *output_num))
        elif out_use_mid == True :
            self.fc_out = nn.Linear(self.lstm_hidden_units,(1 *output_num))
    
    
    def forward(self, x) :
        x = self.fc_in(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm1(x)
        x = F.relu(x)

        if self.num_layers > 0 :
            for i in range(self.num_layers):
                x = self.linear_batchnm['fc_'+str(i+2)](x)
                if self.flag_batch_norm == True :
                    x = self.linear_batchnm['batch_norm_'+str(i+2)](x)
                x = F.relu(x)

        x = self.fc_out(x)

        return x









