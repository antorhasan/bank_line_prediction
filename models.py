import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN_LSTM_Dynamic_Model(nn.Module):
    def __init__(self, num_channels, batch_size, val_batch_size, time_step, num_lstm_layers, drop_out,vert_img_hgt,
                    inp_lr_flag, lf_rt_tag, lstm_hidden_units, flag_reach_use, num_layers,out_use_mid,flag_batch_norm,
                    num_cnn_layers,device,flag_use_lines,flag_bin_out,only_lstm_units,pooling_layer,num_branch_layers,
                    branch_layer_neurons,num_filter_choice):
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

        self.flag_reach_use = flag_reach_use

        self.lstm_dropout = 0
        self.device = device
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.time_step = time_step
        self.num_lstm_layers = num_lstm_layers
        #self.drop_out = True
        self.lstm_hidden_units = lstm_hidden_units
        self.flag_batch_norm = flag_batch_norm
        self.num_layers = num_layers
        num_filter_list = [16, 32]
        num_filters = num_filter_list[num_filter_choice]
        num_filters_out = num_filter_list[num_filter_choice]

        if num_filter_list[num_filter_choice] == 16 :
            self.before_lstm_neurons = 256
        elif num_filter_list[num_filter_choice] == 32 :
            self.before_lstm_neurons = 512
        self.num_cnn_layers = num_cnn_layers
        self.ind_lf_rg = True    ######very important modification
        self.flag_bin_out = flag_bin_out
        self.flag_use_lines = flag_use_lines
        self.only_lstm_units = only_lstm_units
        self.num_branch_layers = num_branch_layers
        self.branch_layer_neurons = branch_layer_neurons
        self.pooling_layer = pooling_layer

        self.conv_inp = nn.Conv2d(num_channels,num_filters_out,(kernel_hgt,3), padding=0)
        if self.flag_batch_norm == True :
            self.batch_norm_1 = nn.BatchNorm2d(num_filters_out)
        vert_tracker = vert_img_hgt - 2
        if vert_tracker == 1 :
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

                vert_tracker = vert_tracker - 2
                if vert_tracker == 1 :
                    kernel_hgt = 1

        self.conv_out = nn.Conv2d(num_filters,num_filters*2,(kernel_hgt,3), padding=0)
        if self.flag_batch_norm == True :
            self.batch_norm_out = nn.BatchNorm2d(num_filters*2)
        
        lstm_inp_feat = self.before_lstm_neurons
        if self.flag_use_lines :
            lstm_inp_feat = self.before_lstm_neurons + ((self.vert_img_hgt*2)+1)

        self.lstm = nn.LSTM( lstm_inp_feat, self.only_lstm_units, num_layers=num_lstm_layers, batch_first=True)
        
        #self.post_lstm_fc = nn.Linear(self.only_lstm_units, self.lstm_hidden_units)

        if num_layers > 0 :

            self.linear_batchnm = nn.ModuleDict({})
            
            for i in range(num_layers) :
                linear_layer_input_features = self.lstm_hidden_units
                if i == 0 :
                    linear_layer_input_features = self.only_lstm_units

                if self.flag_batch_norm == True :
                    self.linear_batchnm.update([
                        ['fc_'+str(i), nn.Linear(linear_layer_input_features, self.lstm_hidden_units)],
                        ['bn_'+str(i), nn.BatchNorm1d(self.lstm_hidden_units)]
                    ])
                
                elif self.flag_batch_norm == False :
                    self.linear_batchnm.update([
                        ['fc_'+str(i), nn.Linear(linear_layer_input_features, self.lstm_hidden_units)],
                    ])

        
        if self.ind_lf_rg == True :
            
            if self.num_branch_layers > 0 :

                self.linear_left_reg_branch = nn.ModuleDict({})
                self.linear_right_reg_branch = nn.ModuleDict({})

                if self.flag_bin_out :
                    self.linear_left_bin_branch = nn.ModuleDict({})
                    self.linear_right_bin_branch = nn.ModuleDict({})
                
                if self.flag_bin_out :
                    branch_layers_dic = {'left_reg':self.linear_left_reg_branch, 
                                        'right_reg':self.linear_right_reg_branch,
                                        'left_bin':self.linear_left_bin_branch, 
                                        'right_bin':self.linear_right_bin_branch}
                else :
                    branch_layers_dic = {'left_reg':self.linear_left_reg_branch, 
                                        'right_reg':self.linear_right_reg_branch}
                
                for j in branch_layers_dic :
                    
                    for i in range(self.num_branch_layers) :
                        linear_layer_input_features = self.branch_layer_neurons
                        linear_layer_output_features = self.branch_layer_neurons

                        if (i == 0) and (self.num_layers == 0) :
                            linear_layer_input_features = self.only_lstm_units
                        elif (i == 0) and (self.num_layers != 0) :
                            linear_layer_input_features = self.lstm_hidden_units

                        if self.flag_batch_norm == True :
                            branch_layers_dic[j].update([
                                ['fc_'+j+'_branch_'+str(i), nn.Linear(linear_layer_input_features, linear_layer_output_features)],
                                ['bn_'+j+'_branch_'+str(i), nn.BatchNorm1d(linear_layer_output_features)]
                            ])
                        
                        elif self.flag_batch_norm == False :
                            branch_layers_dic[j].update([
                                ['fc_'+j+'_branch_'+str(i), nn.Linear(linear_layer_input_features, linear_layer_output_features)],
                            ])


            if (self.num_branch_layers == 0) and (self.num_layers == 0) :
                last_branch_layer_input_features = self.only_lstm_units
            elif self.num_branch_layers == 0 :
                last_branch_layer_input_features = self.lstm_hidden_units
            elif self.num_branch_layers > 0 :
                last_branch_layer_input_features = self.branch_layer_neurons

            self.fc_left_out = nn.Linear(last_branch_layer_input_features, 1)
            self.fc_right_out = nn.Linear(last_branch_layer_input_features, 1)

            if self.flag_bin_out == True :
                self.fc_binl_out = nn.Linear(last_branch_layer_input_features, 2)
                self.fc_binr_out = nn.Linear(last_branch_layer_input_features, 2)


        else :
            if out_use_mid == False :
                self.fc_out = nn.Linear(self.lstm_hidden_units,(self.vert_img_hgt *output_num))
            elif out_use_mid == True :
                self.fc_out = nn.Linear(self.lstm_hidden_units,(1 *output_num))
        

    def forward(self,x, lines, reach):
        #print(x.size())
        # = inp_tuple
        last_batch_size = x.size(0)

        x = torch.reshape(x, (-1, self.vert_img_hgt, 745, 7))
        x = torch.transpose(x, 1,3)
        x = torch.transpose(x, 2,3)
        #print(x.size())

        x = self.conv_inp(x)
        #print(x.size())
        if self.flag_batch_norm == True :
            x = self.batch_norm_1(x)
        x = F.relu(x)

        if self.pooling_layer == 'AvgPool' :
            x = F.avg_pool2d(x, (1,2))
        elif self.pooling_layer == 'MaxPool' :
            x = F.max_pool2d(x, (1,2))

        #print(x.size())


        if self.num_cnn_layers > 0 :
            for i in range(self.num_cnn_layers):

                x = self.cnn_layers_bn['cnn_'+str(i+2)](x)
                #print(x.size())
                if self.flag_batch_norm == True :
                    x = self.cnn_layers_bn['batch_norm2d_'+str(i+2)](x)
                x = F.relu(x)
                #if int(x.size()[3]) > 1 :
                if self.pooling_layer == 'AvgPool' :
                    x = F.avg_pool2d(x, (1,2))
                elif self.pooling_layer == 'MaxPool' :
                    x = F.max_pool2d(x, (1,2))
                #print(x.size())

        #print(asd)

        x = self.conv_out(x)
        if self.flag_batch_norm == True :
            x = self.batch_norm_out(x)
        x = F.relu(x)
        #print(x.size())

        #print(asd)

        x = torch.flatten(x, start_dim=1)

        #print(x.size())
        """ if self.training :
            x = torch.reshape(x, (self.batch_size, (self.time_step-1), -1))
        else : """
        x = torch.reshape(x, (last_batch_size, (self.time_step-1), -1))
        #print(x.size())
        if self.flag_use_lines :
            lines = torch.reshape(lines, (last_batch_size, (self.time_step-1), -1))
            reach = torch.reshape(reach, (last_batch_size, 1, -1))
            reach = reach.expand(-1,(self.time_step-1),-1)
            #print(reach[0:5,:,:])
            x = torch.cat((x,lines,reach), 2)

        h_n = torch.zeros((self.num_lstm_layers,x.size(0),self.only_lstm_units), device=self.device)
        c_n = torch.zeros((self.num_lstm_layers,x.size(0),self.only_lstm_units), device=self.device)
        
        #print(h_n.size())
        #print(asd)

        _, (x, _) = self.lstm(x, (h_n, c_n))
        x = x[-1,:,:]
        #print(x.size())
        #print(x.size())
        x = torch.reshape(x, (-1, self.only_lstm_units))

        if self.num_layers > 0 :
            for i in range(self.num_layers):
                x = self.linear_batchnm['fc_'+str(i)](x)
                if self.flag_batch_norm == True :
                    x = self.linear_batchnm['bn_'+str(i)](x)
                x = F.relu(x)

        #print(x.size())

        if self.ind_lf_rg :
            
            if self.num_branch_layers > 0 :
                for i in range(self.num_branch_layers):
                    if i == 0 :
                        x_left = self.linear_left_reg_branch['fc_left_reg_branch_'+str(i)](x)
                    else :
                        x_left = self.linear_left_reg_branch['fc_left_reg_branch_'+str(i)](x_left)
                    #print(x_left.size())
                    
                    if self.flag_batch_norm == True :
                        x_left = self.linear_left_reg_branch['bn_left_reg_branch_'+str(i)](x_left)
                    x_left = F.relu(x_left)
                    #print(x_left.size())
                    
                #print(asd)

                for i in range(self.num_branch_layers):
                    if i == 0 :
                        x_right = self.linear_right_reg_branch['fc_right_reg_branch_'+str(i)](x)
                    else :
                        x_right = self.linear_right_reg_branch['fc_right_reg_branch_'+str(i)](x_right)

                    if self.flag_batch_norm == True :
                        x_right = self.linear_right_reg_branch['bn_right_reg_branch_'+str(i)](x_right)
                    x_right = F.relu(x_right)
                    #print(x_right.size())

                if self.flag_bin_out :
                    for i in range(self.num_branch_layers):
                        if i == 0 :
                            x_binl = self.linear_left_bin_branch['fc_left_bin_branch_'+str(i)](x)
                        else :
                            x_binl = self.linear_left_bin_branch['fc_left_bin_branch_'+str(i)](x_binl)
                        
                        if self.flag_batch_norm == True :
                            x_binl = self.linear_left_bin_branch['bn_left_bin_branch_'+str(i)](x_binl)
                        x_binl = F.relu(x_binl)

                    for i in range(self.num_branch_layers):
                        if i == 0 :
                            x_binr = self.linear_right_bin_branch['fc_right_bin_branch_'+str(i)](x)
                        else :
                            x_binr = self.linear_right_bin_branch['fc_right_bin_branch_'+str(i)](x_binr)
                        if self.flag_batch_norm == True :
                            x_binr = self.linear_left_reg_branch['bn_right_bin_branch_'+str(i)](x_binr)
                        x_binr = F.relu(x_binr)

            #print(x_left.size())
            

            if self.num_branch_layers > 0 :
                x_left = self.fc_left_out(x_left)
                x_right = self.fc_right_out(x_right)

                if self.flag_bin_out :
                    x_binl = F.softmax(self.fc_binl_out(x_binl), dim=0)
                    x_binr = F.softmax(self.fc_binr_out(x_binr), dim=0)
                else :
                    x_binl = None
                    x_binr = None
                #print(x_left.size())
            else :
                x_left = self.fc_left_out(x)
                x_right = self.fc_right_out(x)

                if self.flag_bin_out :
                    x_binl = F.softmax(self.fc_binl_out(x), dim=0)
                    x_binr = F.softmax(self.fc_binr_out(x), dim=0)
                else :
                    x_binl = None
                    x_binr = None


        else:
            x = self.fc_out(x)
            x_left = None
            x_right = None
            x_binl = None
            x_binr = None

        #print(x.size())
        #print(x_left.size())
        #print(x_right.size())
        #print(x_bin.size())

        #print(asd)
        #print(x_left.shape)
        #print(x_right.shape)
        #print(x_binl.shape)
        #print(x_binr.shape)
        #print(asd)
        return x, x_left, x_right, x_binl, x_binr



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









