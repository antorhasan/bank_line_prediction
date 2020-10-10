import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN_LSTM_Dynamic_Model(nn.Module):
    def __init__(self, num_channels, batch_size, val_batch_size, time_step, num_lstm_layers, drop_out,vert_img_hgt,
                    inp_lr_flag, lf_rt_tag, lstm_hidden_units, flag_reach_use, num_layers,out_use_mid,flag_batch_norm,num_cnn_layers):
        super(CNN_LSTM_Dynamic_Model, self).__init__()
        self.vert_img_hgt = vert_img_hgt

        if vert_img_hgt >= 3 :
            kernel_hgt = 3
        elif vert_img_hgt == 1 :
            kernel_hgt = 1

        if lf_rt_tag == 'left' or lf_rt_tag == 'right' :
            output_num = 1
        elif lf_rt_tag == 'both' :
            output_num = 2

        self.lstm_dropout = 0
        self.device = 'cuda'
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.time_step = time_step
        self.num_lstm_layers = num_lstm_layers
        #self.drop_out = True
        self.lstm_hidden_units = lstm_hidden_units
        self.flag_batch_norm = flag_batch_norm
        self.num_layers = num_layers
        num_filters = 16
        num_filters_out = 16
        self.num_cnn_layers = num_cnn_layers

        self.conv_inp = nn.Conv2d(num_channels,num_filters_out,(kernel_hgt,3), padding=0)
        if self.flag_batch_norm == True :
            self.batch_norm_1 = nn.BatchNorm2d(num_filters_out)
        vert_tracker = vert_img_hgt - 2
        if vert_tracker < 3 :
            kernel_hgt = 1


        self.cnn_layers_bn = nn.ModuleDict({})

        if num_cnn_layers > 0 :
            for i in range(num_cnn_layers) :
                #print(num_filters)
                if self.flag_batch_norm == True :
                    self.cnn_layers_bn.update([
                        ['cnn_'+str(i+2), nn.Conv2d(num_filters,num_filters_out,(kernel_hgt,3), padding=0) ],
                        ['batch_norm2d_'+str(i+2), nn.BatchNorm2d(num_filters_out)]
                    ])
                
                elif self.flag_batch_norm == False :
                    self.cnn_layers_bn.update([
                        ['cnn_'+str(i+2), nn.Conv2d(num_filters,num_filters_out,(kernel_hgt,3), padding=0) ],
                    ])

                if (i+2) % 2 == 0:
                    num_filters_out = num_filters * 2
                if (i+2) % 2 != 0:
                    num_filters = num_filters_out

        self.conv_out = nn.Conv2d(num_filters,num_filters*2,(kernel_hgt,3), padding=0)
        if self.flag_batch_norm == True :
            self.batch_norm_1 = nn.BatchNorm2d(num_filters_out)
        
        lstm_inp_feat = 256

        self.lstm = nn.LSTM( lstm_inp_feat, self.lstm_hidden_units, num_layers=num_lstm_layers, batch_first=True)

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
        

    def forward(self, x):

        x = torch.reshape(x, (-1, self.vert_img_hgt, 745, 7))
        x = torch.transpose(x, 1,3)
        x = torch.transpose(x, 2,3)
        #print(x.size())

        x = self.conv_inp(x)
        #print(x.size())
        if self.flag_batch_norm == True :
            x = self.batch_norm10(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, (1,2))

        #print(x.size())


        if self.num_cnn_layers > 0 :
            for i in range(self.num_cnn_layers):

                x = self.cnn_layers_bn['cnn_'+str(i+2)](x)
                #print(x.size())
                if self.flag_batch_norm == True :
                    x = self.cnn_layers_bn['batch_norm2d_'+str(i+2)](x)
                x = F.relu(x)
                x = F.avg_pool2d(x, (1,2))
                #print(x.size())

        x = self.conv_out(x)
        #print(x.size())
        if self.flag_batch_norm == True :
            x = self.batch_norm10(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim=1)

        #print(x.size())

        x = torch.reshape(x, (self.batch_size, (self.time_step-1), -1))
        
        #print(x.size())

        _, (x, _) = self.lstm(x)
        x = x[-1,:,:]

        #print(x.size())
        

        if self.num_layers > 0 :
            for i in range(self.num_layers):
                x = self.linear_batchnm['fc_'+str(i+2)](x)
                if self.flag_batch_norm == True :
                    x = self.linear_batchnm['batch_norm_'+str(i+2)](x)
                x = F.relu(x)

        #print(x.size())

        x = self.fc_out(x)

        #print(x.size())
        #print(asd)
        
        return x



class Baseline_LSTM_Dynamic_Model(nn.Module):
    def __init__(self, num_channels, batch_size, val_batch_size, time_step, num_lstm_layers, drop_out,vert_img_hgt,
                    inp_lr_flag, lf_rt_tag, lstm_hidden_units, flag_reach_use, num_layers,out_use_mid,flag_batch_norm):
        super(Baseline_LSTM_Dynamic_Model, self).__init__()
        self.vert_img_hgt = vert_img_hgt

        self.lstm_dropout = 0
        self.device = 'cuda'
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.time_step = time_step
        self.num_lstm_layers = num_lstm_layers
        self.drop_out = drop_out
        self.lstm_hidden_units = lstm_hidden_units
        self.flag_batch_norm = flag_batch_norm
        self.num_layers = num_layers
        

        if inp_lr_flag == 'left' or inp_lr_flag == 'right' :
            self.inp_num = 1
        elif inp_lr_flag == 'both' :
            self.inp_num = 2

        
        if lf_rt_tag == 'left' or lf_rt_tag == 'right' :
            output_num = 1
        elif lf_rt_tag == 'both' :
            output_num = 2

        self.lstm = nn.LSTM( ((vert_img_hgt *self.inp_num)+1), self.lstm_hidden_units, num_layers=num_lstm_layers,dropout=self.lstm_dropout,batch_first=True)
        
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


    def forward(self, inputs):
        reach_info = inputs[:,-1:]
        inputs = inputs[:,:-1]

        x = torch.reshape(inputs, (-1,(self.time_step-1),int(self.vert_img_hgt * self.inp_num)))
        reach_info = torch.reshape(reach_info, (-1,1,1))
        reach_info = reach_info.expand(-1,(self.time_step-1),1)
        x = torch.cat((x,reach_info), 2)

        _, (x, _) = self.lstm(x)
        x = x[-1,:,:]

        if self.training:
            x = torch.reshape(x, (-1, self.lstm_hidden_units))
        else:
            x = torch.reshape(x, (-1, self.lstm_hidden_units))

        if self.num_layers > 0 :
            for i in range(self.num_layers):
                x = self.linear_batchnm['fc_'+str(i+2)](x)
                if self.flag_batch_norm == True :
                    x = self.linear_batchnm['batch_norm_'+str(i+2)](x)
                x = F.relu(x)

        x = self.fc_out(x)

        return x

class Baseline_ANN_Dynamic_Model(nn.Module):
    def __init__(self, num_channels, batch_size, val_batch_size, time_step, num_lstm_layers, drop_out,vert_img_hgt,
                    inp_lr_flag, lf_rt_tag, lstm_hidden_units, flag_reach_use, num_layers,out_use_mid,flag_batch_norm):

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
        self.flag_batch_norm = flag_batch_norm
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









