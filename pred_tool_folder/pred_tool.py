#import tensorflow as tf
import torch.nn as nn
import torch
import numpy as np
#np.seterr(all='raise')
import cv2
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from models_pred import Baseline_LSTM_Dynamic_Model,Baseline_ANN_Dynamic_Model,CNN_LSTM_Dynamic_Model
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import itertools
from scipy.signal import savgol_filter
import optuna
from optuna.samplers import TPESampler,RandomSampler
from optuna.pruners import HyperbandPruner,NopPruner
from operator import add
from random import randrange
from torch.utils.data import Dataset, DataLoader


def wrt_test_img(iter_num, prev_actual_list, pred_list, val_img_ids,writer,smooth_flag,
            reach_start_indx,out_use_mid,vert_img_hgt,year_to_pred):
    
    if out_use_mid == True :
        reach_start_indx = reach_start_indx + int((vert_img_hgt-1)/2)

    num_rows = int(pred_list.shape[0])

    denoising = smooth_flag
    window = 99
    poly = 2

    img = cv2.imread(os.path.join('./data/img/up_rgb/'+str(year_to_pred-1)+'01.png'), 1)
    coun = 0
    for i,j in zip(range(reach_start_indx,reach_start_indx+num_rows,1),range(num_rows)):

        if 0<=int(round(pred_list[j,iter_num,0]))<=744 :
            pass
        else :
            pred_list[j,iter_num,0] = 0

        if 0<=int(round(pred_list[j,iter_num,1]))<=744 :
            pass
        else :
            pred_list[j,iter_num,1] = 744

        img[i,int(round(prev_actual_list[j,iter_num,0])),:] = [255,0,0]
        img[i,int(round(prev_actual_list[j,iter_num,1])),:] = [255,0,0]
        img[i,int(round(pred_list[j,iter_num,0])),:] = [0,255,0]
        img[i,int(round(pred_list[j,iter_num,1])),:] = [0,255,0]
        coun+=1
            
    writer.add_image(str(year_to_pred-1)+'01', img, dataformats='HWC')
    cv2.imwrite(os.path.join('./data/predictions/'+str(year_to_pred)+'01.png'), img)
    #return combined_conf


def process_prev(arr_list, num_val_img,prev_year_ids,prev_reach_ids,vert_img_hgt,vert_step,inp_mode,
                flag_standardize_actual,transform_constants,extra_samples):

    arr_list = np.asarray(arr_list, dtype=np.int32)
    total_rows = int(arr_list.shape[0] * arr_list.shape[1])
    arr_list = np.reshape(arr_list, (1,total_rows,2))

    arr_list = np.transpose(arr_list,[1,0,2])

    
    if extra_samples != 0 :
        arr_list = arr_list[:-extra_samples,:,:]

    return arr_list

def process_diffs(arr_list, num_val_img, prev_actual_list,act_year_ids,act_reach_ids,
                    vert_img_hgt,out_mode,flag_standardize_actual,transform_constants,output_subtracted,extra_samples):

    arr_list = np.asarray(arr_list, dtype=np.int32)

    total_rows = int(arr_list.shape[0] * arr_list.shape[1])
    arr_list = np.reshape(arr_list, (1,total_rows,2))

    arr_list = np.transpose(arr_list,[1,0,2])

    if extra_samples != 0 :
        arr_list = arr_list[:-extra_samples,:,:]

    return arr_list

def process_diffs_pred(arr_list, num_val_img, prev_actual_list,act_year_ids,act_reach_ids,
                    vert_img_hgt,out_mode,flag_standardize_actual,transform_constants,output_subtracted,extra_samples):

    arr_list = np.asarray(arr_list, dtype=np.float32)
    total_rows = int(arr_list.shape[0] * arr_list.shape[1])
    arr_list = np.reshape(arr_list, (1,total_rows,2))

    arr_list = np.transpose(arr_list,[1,0,2])

    if extra_samples != 0 :
        arr_list = arr_list[:-extra_samples,:,:]

    return arr_list

    

def model_save(model, optimizer, model_name):
    print('saving model....')
    model_path = os.path.join('./data/model/'+model_name +'.pt')
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, model_path)


def custom_mean_sdd(dataset_f,inp_mode,inp_lr_flag,out_mode,out_lr_tag,flag_standardize_actual,
                    flag_reach_use,batch_size,time_step,vert_img_hgt,flag_sdd_act_data,output_subtracted,
                    out_use_mid,dataset_type,dataset_dic,flag_use_lines):
    print('calculating standardization constants .........')
    inp_list = []
    out_list = []

    prox_counter = 0

    #dataset_type = 'pydic'
    if dataset_type == 'pydic' :
        #flag_use_reachid = True
        num_batches = len(dataset_f)
        for i_batch, sample_batched in enumerate(dataset_f) :
            if i_batch > 130 :
                break

            lines = sample_batched['lines']

            if (i_batch+1) == num_batches :
                last_batch_size = lines.shape[0]
            else :
                last_batch_size = 0
            
            lines_last = lines[:,-1:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]
            lines_prev = lines[:,-2:-1,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]

            lines_last = np.asarray(np.reshape(lines_last, (lines_last.shape[0], 2)))
            lines_prev = np.asarray(np.reshape(lines_prev, (lines_prev.shape[0], 2)))

            if output_subtracted == True :
                lines_sub = lines_last - lines_prev
            elif output_subtracted == False :
                lines_sub = lines_last
            
            lines_sub = np.sum(lines_sub, axis=0)

            if i_batch == 0 :
                np_aggr = np.zeros(lines_sub.shape)

            np_aggr = np_aggr + lines_sub

            if flag_use_lines :

                
                if flag_reach_use :
                    reach_id = sample_batched['reaches']
                    reach_id = reach_id[:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1]
                    reach_id = np.asarray(reach_id)
                    reach_id = np.sum(reach_id, axis=0)
                    if i_batch == 0 :
                        np_aggr_reach = np.zeros(reach_id.shape)
                    np_aggr_reach = np_aggr_reach + reach_id

                lines_prev = lines[:,:-1,:,:]
                lines_prev = np.reshape(lines_prev,(lines_prev.shape[0],-1))
                lines_prev = np.asarray(lines_prev)
                lines_prev = np.sum(lines_prev, axis=0)
                if i_batch == 0 :
                    np_aggr_linprev = np.zeros(lines_prev.shape)
                np_aggr_linprev = np_aggr_linprev + lines_prev

    
        np_aggr_mean = np_aggr / ((i_batch * batch_size) + last_batch_size)



        if flag_use_lines:
            if flag_reach_use :
                np_aggr_reach_mean = np_aggr_reach / ((i_batch * batch_size) + last_batch_size)
            np_aggr_linprev_mean = np_aggr_linprev / ((i_batch * batch_size) + last_batch_size)


        for i_batch, sample_batched in enumerate(dataset_f) :
            if i_batch > 130 :
                break

            lines = sample_batched['lines']

            lines_last = lines[:,-1:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]
            lines_prev = lines[:,-2:-1,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]

            lines_last = np.asarray(np.reshape(lines_last, (lines_last.shape[0], 2)))
            lines_prev = np.asarray(np.reshape(lines_prev, (lines_prev.shape[0], 2)))

            if output_subtracted == True :
                lines_sub = lines_last - lines_prev
            elif output_subtracted == False :
                lines_sub = lines_last

            lines_sub = (lines_sub - np_aggr_mean)**2

            lines_sub = np.sum(lines_sub, axis=0)

            if i_batch == 0 :
                np_aggr = np.zeros(lines_sub.shape)

            np_aggr = np_aggr + lines_sub

            if flag_use_lines :
                if flag_reach_use :
                    reach_id = sample_batched['reaches']
                    reach_id = reach_id[:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1]
                    reach_id = (reach_id - np_aggr_reach_mean)**2
                    reach_id = np.asarray(reach_id)
                    reach_id = np.sum(reach_id, axis=0)
                    if i_batch == 0 :
                        np_aggr_reach = np.zeros(reach_id.shape)
                    np_aggr_reach = np_aggr_reach + reach_id

                lines_prev = lines[:,:-1,:,:]
                lines_prev = np.reshape(lines_prev,(lines_prev.shape[0],-1))
                lines_prev = (lines_prev - np_aggr_linprev_mean)**2
                lines_prev = np.asarray(lines_prev)
                lines_prev = np.sum(lines_prev, axis=0)
                if i_batch == 0 :
                    np_aggr_linprev = np.zeros(lines_prev.shape)
                np_aggr_linprev = np_aggr_linprev + lines_prev
                
        np_aggr_std = np.sqrt(np_aggr / ((i_batch * batch_size) + last_batch_size) )
        if flag_use_lines:
            if flag_reach_use :
                np_aggr_reach_std = np.sqrt(np_aggr_reach / ((i_batch * batch_size) + last_batch_size) )
            np_aggr_linprev_std = np.sqrt(np_aggr_linprev / ((i_batch * batch_size) + last_batch_size) )

        if flag_use_lines :
            #if flag_reach_use :
            transform_constants = {'inp_lines_mean':np_aggr_linprev_mean,'inp_lines_std':np_aggr_linprev_std,
                        'inp_reach_mean':np_aggr_reach_mean,'inp_reach_std':np_aggr_reach_std,
                        'out_mean':np_aggr_mean,'out_std':np_aggr_std}
        else :
            transform_constants = {'inp_lines_mean':None,'inp_lines_std':None,
                        'inp_reach_mean':None,'inp_reach_std':None,
                        'out_mean':np_aggr_mean,'out_std':np_aggr_std}

    #print(transform_constants)
    #print(np_aggr_std)
    #print(asd)
    return transform_constants
    

def pt_train_per(dataset_tr_pr,inp_mode,inp_lr_flag,out_mode,out_lr_tag,flag_standardize_actual,flag_reach_use,
        batch_size,time_step,vert_img_hgt,flag_sdd_act_data,inp_mean,inp_std,out_mean,out_std,model,
        loss_func,transform_constants,num_val_img,output_subtracted,out_use_mid,flag_use_lines,flag_use_imgs,device):
    
    out_mean = transform_constants['out_mean']
    out_std = transform_constants['out_std']
    inp_lines_mean = transform_constants['inp_lines_mean']
    inp_lines_std = transform_constants['inp_lines_std']
    inp_reach_mean = transform_constants['inp_reach_mean']
    inp_reach_std = transform_constants['inp_reach_std']

    model.eval()
    with torch.no_grad():

        num_batches = len(dataset_tr_pr)
        for i_batch, sample_batched in enumerate(dataset_tr_pr) :
            if i_batch > 130 :
                break
            if flag_use_imgs :
                inp_flatten = sample_batched['img']
                inp_flatten = inp_flatten[:,:-1,:,:,:]
                inp_flatten = inp_flatten / 255.0
                inp_flatten = np.asarray(inp_flatten, dtype=np.float32)
                inp_flatten = torch.Tensor(inp_flatten).to(device)
                #inp_flatten = torch.Tensor(inp_flatten).cuda()   
            else :
                inp_flatten = None             

            out_flatten_org = sample_batched['lines']

            if (i_batch+1) == num_batches :
                last_batch_size = out_flatten_org.shape[0]
            else :
                last_batch_size = 0

            
            lines_last = out_flatten_org[:,-1:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]
            lines_prev = out_flatten_org[:,-2:-1,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]

            lines_last = torch.reshape(lines_last, (lines_last.shape[0], 2))
            lines_prev = torch.reshape(lines_prev, (lines_prev.shape[0], 2))

            if flag_use_lines :
                lines_prev_inp = out_flatten_org[:,:-1,:,:]
                lines_prev_inp = torch.reshape(lines_prev_inp, (lines_prev_inp.size()[0],-1))
                lines_prev_inp = (lines_prev_inp - inp_lines_mean) / inp_lines_std
                lines_prev_inp = lines_prev_inp.float().to(device).requires_grad_(False)
                #lines_prev_inp = lines_prev_inp.float().cuda().requires_grad_(False)
                if flag_reach_use :
                    reach_id = sample_batched['reaches']
                    reach_id = reach_id[:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1]                        
                    reach_id = torch.reshape(reach_id, (reach_id.shape[0], 1))
                    reach_id = (reach_id - inp_reach_mean) / inp_reach_std
                    reach_id = reach_id.float().to(device).requires_grad_(False)
                    #reach_id = reach_id.float().cuda().requires_grad_(False)

            else :
                lines_prev_inp = None
                reach_id = None
            
            _, pred_left, pred_right, _, _ = model(inp_flatten, lines_prev_inp, reach_id)
            pred_left = pred_left.cpu()
            pred_left = pred_left.numpy()
            pred_right = pred_right.cpu()
            pred_right = pred_right.numpy()

            pred = np.concatenate((pred_left,pred_right), axis=1)
            pred = np.add(np.multiply(pred,out_std),out_mean)
            
            if output_subtracted == True :
                pred = np.add(pred,lines_prev)

            lines_last = np.asarray(lines_last)
            pred = np.asarray(pred)

            abs_batch_mae = np.absolute(lines_last-pred)
            abs_batch_mae = np.sum(abs_batch_mae, axis=0)

            if i_batch == 0 :
                np_aggr = np.zeros(abs_batch_mae.shape)

            np_aggr = np_aggr + abs_batch_mae

    train_mae = np_aggr / ((i_batch * batch_size) + last_batch_size)

    return train_mae



class Pytorch_Dataset(Dataset):

    def __init__(self, data_ids, data_dic, time_step, vert_img_hgt):

        self.data_ids = data_ids
        self.data_dic = data_dic
        self.time_step = time_step
        self.vert_img_hgt = vert_img_hgt

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_data = self.data_dic['imgs']
        year_data = self.data_dic['years']
        reach_data = self.data_dic['reaches']
        lines_data = self.data_dic['lines']

        data_id = self.data_ids[idx]
        year_idx = data_id[0]
        reach_idx = data_id[1]

        year_id = year_data[year_idx : year_idx + self.time_step]
        reach_id = reach_data[year_idx, reach_idx : reach_idx + self.vert_img_hgt]

        image = img_data[year_idx:year_idx+self.time_step, reach_idx:reach_idx+self.vert_img_hgt, :,:]
        lines = lines_data[year_idx:year_idx+self.time_step, reach_idx:reach_idx+self.vert_img_hgt,:]

        sample = {'img':image,'lines':lines,'years':year_id,'reaches':reach_id}

        return sample



def create_dic_train_dataset(data_div_step,total_time_step,start_indx,reach_win_size,reach_shift_cons,reach_start_indx,
            vert_img_hgt,vert_step,end_indx,time_step,time_win_shift,train_shuffle,batch_size,dataset_dic):

    
    time_step_start = start_indx
    time_step_end = data_div_step
    time_step_values = np.arange(time_step_start, time_step_end, 1)
    time_step_values = time_step_values[:-(time_step-1)]

    reach_start = reach_start_indx
    reach_end = reach_win_size
    reach_values = np.arange(reach_start, reach_end, 1)
    reach_values = reach_values[:-(vert_img_hgt-1)]

    data_ids = []

    for i in time_step_values:
        for j in reach_values:
            data_ids.append([i,j])

    pytorch_dataset = Pytorch_Dataset(data_ids,dataset_dic,time_step,vert_img_hgt)

    if train_shuffle == True :   
        dataloader = DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=6,drop_last=True,pin_memory=True)
    elif train_shuffle == False :
        dataloader = DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=6,drop_last=False,pin_memory=True)
    #print(time_step_values)
    #print(asd)


    return dataloader

def create_dic_val_dataset(total_time_step,val_split,val_skip,reach_win_size,reach_shift_cons,reach_start_indx,
        vert_img_hgt,vert_step,val_img_range,time_step,val_batch_size,dataset_dic):

    time_step_start = total_time_step - (val_split+1)
    time_step_end = total_time_step - val_skip
    time_step_values = np.arange(time_step_start, time_step_end, 1)
    time_step_values = time_step_values[:-(time_step-1)]

    reach_start = reach_start_indx
    reach_end = reach_win_size
    reach_values = np.arange(reach_start, reach_end, 1)
    reach_values = reach_values[:-(vert_img_hgt-1)]

    data_ids = []

    for i in time_step_values:
        for j in reach_values:
            data_ids.append([i,j])

    pytorch_dataset = Pytorch_Dataset(data_ids,dataset_dic,time_step,vert_img_hgt)

    dataloader = DataLoader(pytorch_dataset, batch_size=val_batch_size, shuffle=False, num_workers=6,drop_last=False,pin_memory=True)
    
    return dataloader

def create_dic_test_dataset(total_time_step,val_split,val_skip,reach_win_size,reach_shift_cons,reach_start_indx,
        vert_img_hgt,vert_step,val_img_range,time_step,val_batch_size,dataset_dic):

    #time_step_start = total_time_step - (val_split+1)
    #time_step_end = total_time_step - val_skip
    #print(val_split)
    time_step_start = val_split - (time_step-1)
    time_step_end = val_split + 1

    time_step_values = np.arange(time_step_start, time_step_end, 1)
    
    #print(time_step_start)
    #if time_step_start + (time_step-1) >= total_time_step :
    #    time_step_values = time_step_values[:-(time_step-1)]
    #elif time_step_start + (time_step-1) < total_time_step :
        #time_step_values = time_step_values[:-(time_step)]
    #    pass
    time_step_values = time_step_values[:-(time_step-1)]
    #print(time_step_values)
    #print(asd)

    reach_start = reach_start_indx
    reach_end = reach_win_size
    reach_values = np.arange(reach_start, reach_end, 1)
    reach_values = reach_values[:-(vert_img_hgt-1)]

    data_ids = []

    for i in time_step_values:
        for j in reach_values:
            data_ids.append([i,j])

    #print(time_step_start)
    #print(time_step_end)
    #print(time_step_values)
    #print(data_ids)
    #print(asd)

    pytorch_dataset = Pytorch_Dataset(data_ids,dataset_dic,time_step,vert_img_hgt)

    dataloader = DataLoader(pytorch_dataset, batch_size=val_batch_size, shuffle=False, num_workers=6,drop_last=False,pin_memory=True)
    
    return dataloader

def calculate_loss(pred_left, pred_right, pred_binl, pred_binr, out_flatten, bin_out_flatten_left,
                        bin_out_flatten_right,flag_use_lines,flag_bin_out,loss_func,right_loss_weight):
    
    #print(out_flatten[:,0:1].shape)
    

    if loss_func == 'l1_loss' :
        if flag_bin_out :
            loss = F.l1_loss(pred_left, out_flatten[:,0:1],reduction='mean') + \
                    F.l1_loss(pred_right, out_flatten[:,1:2],reduction='mean') + \
                    F.binary_cross_entropy(pred_binl, bin_out_flatten_left,reduction='mean') + \
                    F.binary_cross_entropy(pred_binr, bin_out_flatten_right,reduction='mean')
        else :
            loss = ((1-right_loss_weight)*F.l1_loss(pred_left, out_flatten[:,0:1],reduction='mean')) + \
                    (right_loss_weight*F.l1_loss(pred_right, out_flatten[:,1:2],reduction='mean'))
    elif loss_func == 'mse_loss' :
        if flag_bin_out :
            loss = F.mse_loss(pred_left, out_flatten[:,0:1],reduction='mean') + \
                    F.mse_loss(pred_right, out_flatten[:,1:2],reduction='mean') + \
                    F.binary_cross_entropy(pred_binl, bin_out_flatten_left,reduction='mean') + \
                    F.binary_cross_entropy(pred_binr, bin_out_flatten_right,reduction='mean')
        else :
            loss = ((1-right_loss_weight)*F.mse_loss(pred_left, out_flatten[:,0:1],reduction='mean')) + \
                    (right_loss_weight*F.mse_loss(pred_right, out_flatten[:,1:2],reduction='mean'))
    elif loss_func == 'huber_loss' :
        if flag_bin_out :
            loss = F.smooth_l1_loss(pred_left, out_flatten[:,0:1],reduction='mean') + \
                    F.smooth_l1_loss(pred_right, out_flatten[:,1:2],reduction='mean') + \
                    F.binary_cross_entropy(pred_binl, bin_out_flatten_left,reduction='mean') + \
                    F.binary_cross_entropy(pred_binr, bin_out_flatten_right,reduction='mean')
        else :
            loss = ((1-right_loss_weight)*F.smooth_l1_loss(pred_left, out_flatten[:,0:1],reduction='mean')) + \
                    (right_loss_weight*F.smooth_l1_loss(pred_right, out_flatten[:,1:2],reduction='mean'))
    elif loss_func == 'log_cosh' :
        def log_cosh(pred, ground_t):
            return torch.mean(torch.log(torch.cosh((pred - ground_t) + 1e-12)))

        if flag_bin_out :
            loss = log_cosh(pred_left, out_flatten[:,0:1]) + \
                    log_cosh(pred_right, out_flatten[:,1:2]) + \
                    F.binary_cross_entropy(pred_binl, bin_out_flatten_left,reduction='mean') + \
                    F.binary_cross_entropy(pred_binr, bin_out_flatten_right,reduction='mean')
        else :
            loss = ((1-right_loss_weight)*log_cosh(pred_left, out_flatten[:,0:1])) + \
                    (right_loss_weight*log_cosh(pred_right, out_flatten[:,1:2]))

    return loss

def pytorch_process_inp(sample_batched,vert_img_hgt,output_subtracted,flag_bin_out,out_mean,out_std,
    flag_use_lines,inp_lines_mean,inp_lines_std,flag_reach_use,inp_reach_mean,inp_reach_std,flag_use_imgs,device):

    if flag_use_imgs :
        inp_flatten = sample_batched['img']
        inp_flatten = inp_flatten[:,:-1,:,:,:]
        inp_flatten = inp_flatten / 255.0
        inp_flatten = inp_flatten.to(device).requires_grad_(False)
        #inp_flatten = inp_flatten.cuda().requires_grad_(False)
    else :
        inp_flatten = None

    out_flatten_org = sample_batched['lines']


    lines_last = out_flatten_org[:,-1:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]
    lines_prev = out_flatten_org[:,-2:-1,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]

    lines_last = torch.reshape(lines_last, (lines_last.shape[0], 2))
    lines_prev = torch.reshape(lines_prev, (lines_prev.shape[0], 2))

    if output_subtracted == True :
        out_flatten = lines_last - lines_prev
    elif output_subtracted == False :
        out_flatten = lines_last

    if flag_bin_out :
        bin_out_flatten_left = lines_prev[:,0] - lines_last[:,0]
        bin_out_flatten_left = torch.reshape(torch.where(bin_out_flatten_left > 2, torch.ones(bin_out_flatten_left.size()),
                                        torch.zeros(bin_out_flatten_left.size())), (-1, 1))
        bin_out_flatten_left_c = torch.reshape(torch.where(bin_out_flatten_left == 1, torch.zeros(bin_out_flatten_left.size()),
                                        torch.ones(bin_out_flatten_left.size())), (-1,1))
        bin_out_flatten_left = torch.cat((bin_out_flatten_left_c,bin_out_flatten_left), 1)
        bin_out_flatten_left = bin_out_flatten_left.to(device).requires_grad_(False)
        #bin_out_flatten_left = bin_out_flatten_left.cuda().requires_grad_(False)
        
        bin_out_flatten_right = lines_last[:,1] - lines_prev[:,1]
        bin_out_flatten_right = torch.reshape(torch.where(bin_out_flatten_right > 2, torch.ones(bin_out_flatten_right.size()),
                                        torch.zeros(bin_out_flatten_right.size())), (-1, 1))
        bin_out_flatten_right_c = torch.reshape(torch.where(bin_out_flatten_right == 1, torch.zeros(bin_out_flatten_right.size()),
                                        torch.ones(bin_out_flatten_right.size())), (-1,1))
        bin_out_flatten_right = torch.cat((bin_out_flatten_right_c,bin_out_flatten_right), 1)
        bin_out_flatten_right = bin_out_flatten_right.to(device).requires_grad_(False)
        #bin_out_flatten_right = bin_out_flatten_right.cuda().requires_grad_(False)
    else :
        bin_out_flatten_left = None
        bin_out_flatten_right = None


    out_flatten = (out_flatten - out_mean) / out_std 

    out_flatten = out_flatten.float().to(device).requires_grad_(False)
    #out_flatten = out_flatten.float().cuda().requires_grad_(False)

    if flag_use_lines :
        lines_prev_inp = out_flatten_org[:,:-1,:,:]
        lines_prev_inp = torch.reshape(lines_prev_inp, (lines_prev_inp.size()[0],-1))
        lines_prev_inp = (lines_prev_inp - inp_lines_mean) / inp_lines_std
        lines_prev_inp = lines_prev_inp.float().to(device).requires_grad_(False)
        #lines_prev_inp = lines_prev_inp.float().cuda().requires_grad_(False)


        if flag_reach_use :
            reach_id = sample_batched['reaches']
            reach_id = reach_id[:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1]                        
            reach_id = torch.reshape(reach_id, (reach_id.shape[0], 1))
            reach_id = (reach_id - inp_reach_mean) / inp_reach_std
            reach_id = reach_id.float().to(device).requires_grad_(False)
            #reach_id = reach_id.float().cuda().requires_grad_(False)
    else :
        lines_prev_inp = None
        reach_id = None

    return inp_flatten,lines_prev_inp,reach_id,out_flatten,bin_out_flatten_left,bin_out_flatten_right,lines_prev

def pytorch_process_test_inp(sample_batched,vert_img_hgt,output_subtracted,flag_bin_out,out_mean,out_std,
        flag_use_lines,inp_lines_mean,inp_lines_std,flag_reach_use,inp_reach_mean,inp_reach_std,flag_use_imgs,device,flag_test_inp):

    if flag_use_imgs :
        inp_flatten = sample_batched['img']
        if flag_test_inp == False :
            inp_flatten = inp_flatten[:,:,:,:,:]
            #out_flatten_org = sample_batched['lines']
        elif flag_test_inp == True :
            inp_flatten = inp_flatten[:,:-1,:,:,:]
            #out_flatten_org = sample_batched['lines']
        inp_flatten = inp_flatten / 255.0
        inp_flatten = inp_flatten.to(device).requires_grad_(False)
        #inp_flatten = inp_flatten.cuda().requires_grad_(False)
    else :
        inp_flatten = None

    out_flatten_org = sample_batched['lines']

    #print(inp_flatten.shape)
    #print(out_flatten_org.shape)
    #print(inp_lines_mean.shape)
    #print(inp_lines_std)
    #print(asd)

    #lines_last = out_flatten_org[:,-1:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]
    if flag_test_inp == False :
        lines_prev = out_flatten_org[:,-1:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]
    elif flag_test_inp == True :
        lines_prev = out_flatten_org[:,-2:-1,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]
    #lines_last = torch.reshape(lines_last, (lines_last.shape[0], 2))
    lines_prev = torch.reshape(lines_prev, (lines_prev.shape[0], 2))


    if flag_use_lines :
        if flag_test_inp == False :
            lines_prev_inp = out_flatten_org[:,:,:,:]
        elif flag_test_inp == True :
            lines_prev_inp = out_flatten_org[:,:-1,:,:]
        lines_prev_inp = torch.reshape(lines_prev_inp, (lines_prev_inp.size()[0],-1))
        lines_prev_inp = (lines_prev_inp - inp_lines_mean) / inp_lines_std
        lines_prev_inp = lines_prev_inp.float().to(device).requires_grad_(False)
        #lines_prev_inp = lines_prev_inp.float().cuda().requires_grad_(False)


        if flag_reach_use :
            reach_id = sample_batched['reaches']
            reach_id = reach_id[:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1]                        
            reach_id = torch.reshape(reach_id, (reach_id.shape[0], 1))
            reach_id = (reach_id - inp_reach_mean) / inp_reach_std
            reach_id = reach_id.float().to(device).requires_grad_(False)
            #reach_id = reach_id.float().cuda().requires_grad_(False)
    else :
        lines_prev_inp = None
        reach_id = None

    return inp_flatten,lines_prev_inp,reach_id,lines_prev


def objective(tm_stp, strt, lr_pow, ad_pow, vert_hgt, vert_step_num, num_epochs,train_shuffle,get_train_mae,transform_constants,
                lstm_layers,lstm_hidden_units,batch_size,inp_bnk,out_bnk,val_split,val_skip,model_type,num_layers,
                model_optim,loss_func,save_mod,load_mod,load_file,skip_training,output_subtracted,train_val_gap,
                out_use_mid,trail_id,flag_batch_norm,dataset_dic,num_cnn_layers,flag_use_lines,pooling_layer,flag_bin_out,
                only_lstm_units,num_branch_layers,branch_layer_neurons,right_loss_weight,strtn_num_chanls,flag_use_imgs,
                erosion_thresh,num_lft_brn_lyrs,num_rgt_brn_lyrs,lstm_dropout,flag_dilated_cov,year_to_pred):
    
    load_mod = load_mod
    load_file = load_file
    save_mod = save_mod
    model_save_at = 20
    get_train_mae = get_train_mae
    num_lstm_layers = lstm_layers
    num_channels = 7
    inp_lr_flag = inp_bnk
    out_lr_tag = out_bnk
    EPOCHS = num_epochs
    data_mode = 'imgs' 
    lr_rate = 1*(10**lr_pow)
    vert_img_hgt = vert_hgt
    vert_step = vert_step_num            #vert skip step
    wgt_seed_flag = True
    if wgt_seed_flag :
        torch.manual_seed(0)
    val_skip = val_skip
    out_use_mid = out_use_mid

    model_optim = model_optim
    
    drop_rate = []
    smooth_flag = False
    time_step = tm_stp
    
    log_performance = 1 ###number of epochs after which performance metrics are calculated
    log_val_loss_at = get_train_mae
    #log_val_loss_at = 1
    early_stop_flag = False
    early_stop_thresh = 30
    path_to_val_img = os.path.join('./data/img/up_rgb/')
    val_img_ids = [int(f.split('.')[0]) for f in listdir(path_to_val_img) if isfile(join(path_to_val_img, f))]
    val_img_ids.sort()
    org_val_img = val_img_ids
    start_indx = strt
    #val_split = val_split
    
    val_numbers_id = (val_split+1) - (time_step-1)
    if (skip_training == False) and (val_skip == 0) :
        val_img_ids = val_img_ids[-(val_numbers_id):]
    elif skip_training == False :
        val_img_ids = val_img_ids[-(val_numbers_id):-(val_skip)]
    elif (skip_training == True) and (val_skip == 0):
        val_img_ids = val_img_ids[-(val_numbers_id):]
    elif (skip_training == True) and (val_skip > 0):
        val_img_ids = val_img_ids[-(val_numbers_id):-(val_skip)]


    total_time_step = len(org_val_img)    ###number of total year images
    #print(total_time_step)
    #print(asd)
    num_val_img = len(val_img_ids)

    tr_pr_batch_size = 2222 - (2*(int((vert_img_hgt-1)/2)))
    val_batch_size = tr_pr_batch_size
    
    """ if train_val_gap == True :
        data_div_step = total_time_step - (val_split)
    elif train_val_gap == False :
        data_div_step = total_time_step - (val_split - time_step + 2) """

    data_div_step = val_split
    #print(data_div_step)
    #print(asd)
    if data_div_step == total_time_step :
        flag_test_inp = False
    elif data_div_step < total_time_step :
        flag_test_inp = True

    end_indx = data_div_step-1
    log_hist = 6
    writer = SummaryWriter()
    model_name = writer.get_logdir().split("\\")[1]
    adm_wd = ad_pow
    val_img_range = time_step+num_val_img-1
    time_win_shift = 1

    reach_start_indx = 0 
    reach_end_num = 0

    reach_shift_cons = 2222
    reach_win_size = reach_shift_cons - reach_end_num 
    reach_end_indx = reach_win_size - 1

    flag_reach_use = True
    flag_sdd_act_data = True
    flag_standardize_actual = True
    loss_func = loss_func

    

    out_mode = 'act'
    inp_mode = 'act'

    loaded_from = 'None'
    if load_mod == True :
        loaded_from = load_file

    
    hyperparameter_defaults = dict(
        adam_wdecay = adm_wd,
        num_channels = num_channels,
        batch_size = batch_size,
        learning_rate = lr_rate,
        time_step = time_step,
        num_lstm_layers = num_lstm_layers,
        dataset='7_chann',
        model_type=model_type,
        vertical_image_window = vert_img_hgt,
        start_indx = org_val_img[start_indx],
        end_indx = org_val_img[end_indx],
        weight_seed = wgt_seed_flag,
        vertical_pix_step = vert_step,
        input_data = data_mode,
        model_optimizer = model_optim,
        reach_start_index = reach_start_indx,
        reach_end_index = reach_end_indx,
        input_lft_rgt_tag = inp_lr_flag,
        output_lft_rgt_tag = out_lr_tag,
        output_mode = out_mode,
        flag_reach_use = flag_reach_use,
        loss_function = loss_func,
        num_layers = num_layers,
        num_neurons_per_layer = lstm_hidden_units,
        out_use_mid = out_use_mid,
        model_loaded_from = loaded_from,
        trail_id = trail_id,
        output_subtracted= output_subtracted,
        flag_batch_norm = flag_batch_norm,
        flag_bin_out = flag_bin_out,
        flag_use_lines = flag_use_lines,
        only_lstm_units = only_lstm_units,
        num_branch_layers = num_branch_layers,
        num_lft_brn_lyrs = num_lft_brn_lyrs,
        num_rgt_brn_lyrs = num_rgt_brn_lyrs,
        branch_layer_neurons = branch_layer_neurons,
        pooling_layer = pooling_layer,
        flag_use_imgs = flag_use_imgs,
        erosion_thresh = erosion_thresh,
        right_loss_weight = right_loss_weight,
        lstm_dropout = lstm_dropout,
        strtn_num_chanls = strtn_num_chanls,
        flag_dilated_cov = flag_dilated_cov
        )

    if skip_training == False :
        if data_mode == 'imgs' :
            dataset_f = create_dic_train_dataset(data_div_step,total_time_step,start_indx,reach_win_size,reach_shift_cons,reach_start_indx,
                vert_img_hgt,vert_step,end_indx,time_step,time_win_shift,train_shuffle,batch_size,dataset_dic)

    if skip_training == False :
        if data_mode == 'imgs' :
            time_step_start = start_indx
            time_step_end = data_div_step
            time_step_values = np.arange(time_step_start, time_step_end, 1)
            time_step_values = time_step_values[:-(time_step-1)]

            dataset_tr_pr = create_dic_train_dataset(data_div_step,total_time_step,start_indx,reach_win_size,reach_shift_cons,reach_start_indx,
                vert_img_hgt,vert_step,end_indx,time_step,time_win_shift,train_shuffle=False,batch_size=batch_size,dataset_dic=dataset_dic)
    
    if data_mode == 'imgs' :
        dataseti1 = create_dic_test_dataset(total_time_step,val_split,val_skip,reach_win_size,reach_shift_cons,reach_start_indx,
                vert_img_hgt,vert_step,val_img_range,time_step,batch_size,dataset_dic)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)

    if model_type == 'ANN':
        model = Baseline_ANN_Dynamic_Model(num_channels, batch_size, val_batch_size,time_step, num_lstm_layers, drop_rate, 
                vert_img_hgt, inp_lr_flag, out_lr_tag, lstm_hidden_units,flag_reach_use,num_layers,out_use_mid,flag_batch_norm)
    elif model_type == 'LSTM':
        model = Baseline_LSTM_Dynamic_Model(num_channels, batch_size, val_batch_size,time_step, num_lstm_layers, drop_rate, 
                vert_img_hgt, inp_lr_flag, out_lr_tag, lstm_hidden_units,flag_reach_use,num_layers,out_use_mid,flag_batch_norm)
    elif (model_type == 'CNN_LSTM') and (inp_lr_flag == 'img'):
        model = CNN_LSTM_Dynamic_Model(num_channels, batch_size, val_batch_size,time_step, num_lstm_layers, drop_rate, 
                vert_img_hgt, inp_lr_flag, out_lr_tag, lstm_hidden_units,flag_reach_use,num_layers,out_use_mid,flag_batch_norm,
                num_cnn_layers,device,flag_use_lines,flag_bin_out,only_lstm_units,pooling_layer,num_branch_layers,
                branch_layer_neurons,strtn_num_chanls,flag_use_imgs,num_lft_brn_lyrs,num_rgt_brn_lyrs,lstm_dropout,flag_dilated_cov)


    model = model.to(device)

    print('Model layers and properties ........')
    print(model)
    #print(asd)

    if model_optim == 'SGD' :
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, weight_decay=adm_wd)
    elif model_optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=adm_wd)
    elif model_optim == 'SGD_M' :
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay=adm_wd)

    if (load_mod == True) and (load_file != None):
        print('loading model .......')
        checkpoint = torch.load(os.path.join('./data/model/'+ load_file +'.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    early_stop_counter = 0
    global_train_counter = 0

    dataset_type = 'pydic'
    if transform_constants == None :
        transform_constants = custom_mean_sdd(dataset_tr_pr,inp_mode,inp_lr_flag,out_mode,out_lr_tag,flag_standardize_actual,
            flag_reach_use,batch_size,time_step,vert_img_hgt,flag_sdd_act_data,output_subtracted,out_use_mid,dataset_type,
            dataset_dic,flag_use_lines)

    if flag_use_lines :
        inp_lines_mean = transform_constants['inp_lines_mean']
        inp_lines_std = transform_constants['inp_lines_std']
        inp_reach_mean = transform_constants['inp_reach_mean']
        inp_reach_std = transform_constants['inp_reach_std']
    else :
        inp_lines_mean = None
        inp_lines_std = None
        inp_reach_mean = None
        inp_reach_std = None

    #print(transform_constants)
    #print(asd)

    out_mean = transform_constants['out_mean']
    out_std = transform_constants['out_std']

    #print(inp_mean)
    #print(out_mean)

    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    train_batch_count_global = 0
    print('started training ..........')
    for epoch in range(EPOCHS):

        model.train()
        counter = 0
        
        epoch_loss = 0
        batch_loss_counter = 0
        batch_loss = 0
        batch_counter = 0

        if skip_training == False :
            for i_batch, sample_batched in enumerate(dataset_f) :
                
                if i_batch > 130 :
                    break

                inp_flatten,lines_prev_inp,reach_id,out_flatten,bin_out_flatten_left, \
                bin_out_flatten_right,_ = pytorch_process_inp(sample_batched,vert_img_hgt,
                                            output_subtracted,flag_bin_out,out_mean,out_std,
                                            flag_use_lines,inp_lines_mean,inp_lines_std,
                                            flag_reach_use,inp_reach_mean,inp_reach_std,flag_use_imgs,device)

                if i_batch == 0 :
                    if flag_use_imgs :
                        print(inp_flatten.shape)
                        print(inp_flatten.dtype)
                    print(out_flatten.shape)
                    print(out_flatten.dtype)

                    if lines_prev_inp != None :
                        print(lines_prev_inp.shape)
                        print(lines_prev_inp.dtype)
                        print(reach_id.shape)
                        print(reach_id.dtype)
                    

                optimizer.zero_grad()

                _, pred_left, pred_right, pred_binl, pred_binr = model(inp_flatten, lines_prev_inp, reach_id)

                
                loss = calculate_loss(pred_left, pred_right, pred_binl, pred_binr, out_flatten, bin_out_flatten_left,
                        bin_out_flatten_right,flag_use_lines,flag_bin_out,loss_func,right_loss_weight)

                loss.backward()
                optimizer.step()

                epoch_loss = epoch_loss+loss
                counter += 1

                batch_loss = batch_loss + loss
                
                batch_counter += 1

                if ((i_batch+1)*batch_size) <= ((batch_loss_counter+1)*500) < ((i_batch+2)*batch_size) :
                    
                    batch_loss_counter += 1
                    batch_loss = batch_loss / batch_counter

                    batch_template = 'Epoch {}, {} batch Loss: {}'
                    print(batch_template.format(epoch+1, i_batch+1, batch_loss))
                    writer.add_scalar('Loss/train_batch', batch_loss, train_batch_count_global)
                    batch_counter = 0
                    batch_loss = 0
                    train_batch_count_global += 1

        
            flag_sdd_act_data == False
            avg_epoch_loss = epoch_loss / counter
            template = 'Epoch {}, Train Loss: {}'
            print(template.format(epoch+1,avg_epoch_loss))

            writer.add_scalar('Loss/train', avg_epoch_loss, epoch+1)
            train_losses.append(avg_epoch_loss.item())


            if epoch % log_hist == log_hist-1:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram('parameters_'+str(name), param.data, epoch + 1)
                        writer.add_histogram('gradients'+str(name), param.grad, epoch + 1)

        counter_val = 0
        model.eval()

        if True :
            print('started testing .......')
            with torch.no_grad():
                #print('dummy')
                #print(inp_lr_flag)
            
                pred_list = []
                #actual_list = []
                prev_actual_list = []
                prev_reach_ids = []
                prev_year_ids = []
                act_reach_ids = []
                act_year_ids = []

                #counter_temp = 0

                #prev_sum_temp = []

                for i_batch, sample_batched in enumerate(dataseti1) :

                    #print('dummy')
                    #print(inp_lr_flag)
                    #print(inp_lr_flag)
                    inp_flatten,lines_prev_inp,reach_id,lines_prev = pytorch_process_test_inp(sample_batched,vert_img_hgt,
                                            output_subtracted,flag_bin_out,out_mean,out_std,
                                            flag_use_lines,inp_lines_mean,inp_lines_std,
                                            flag_reach_use,inp_reach_mean,inp_reach_std,flag_use_imgs,device,flag_test_inp)

                    if i_batch == 0 :
                        if flag_use_imgs :
                            print(inp_flatten.shape)
                            print(inp_flatten.dtype)
                        #print(out_flatten.shape)
                        #print(out_flatten.dtype)

                        if lines_prev_inp != None :
                            print(lines_prev_inp.shape)
                            print(lines_prev_inp.dtype)
                            print(reach_id.shape)
                            print(reach_id.dtype)

                    _, pred_left, pred_right, pred_binl, pred_binr = model(inp_flatten, lines_prev_inp, reach_id)                    

                    counter_val += 1

                    #if inp_lr_flag == 'img' :
                        #out_flatten = out_flatten.cpu()
                        #out_flatten = out_flatten.numpy()
                        
                    pred_left = pred_left.cpu()
                    pred_left = pred_left.numpy()
                    pred_right = pred_right.cpu()
                    pred_right = pred_right.numpy()

                    pred = np.concatenate((pred_left,pred_right), axis=1)
                    #out_flatten = np.add(np.multiply(out_flatten, out_std), out_mean)
                    
                    pred = np.add(np.multiply(pred, out_std), out_mean)
                    #print('dummy')
                    extra_samples = 0
                    #print('ran_samp')
                    if ((i_batch+1) == len(dataseti1)) and (batch_size != lines_prev.shape[0]) :
                        extra_samples = int(batch_size -lines_prev.shape[0])
                        lines_prev = np.pad(lines_prev, ((0,extra_samples),(0,0)) )
                        #out_flatten = np.pad(out_flatten, ((0,extra_samples),(0,0)) )
                        pred = np.pad(pred, ((0,extra_samples),(0,0)) )

                    lines_prev = np.asarray(lines_prev)
                    prev_actual_list.append(lines_prev)
                    #actual_list.append(out_flatten)
                    pred_list.append(pred) 

            #print('dummy')
            if epoch % log_performance == log_performance-1 :
                prev_actual_list = process_prev(prev_actual_list,num_val_img,prev_year_ids,prev_reach_ids,vert_img_hgt,vert_step,inp_mode,
                                    flag_standardize_actual,transform_constants,extra_samples)
                #actual_list = process_diffs(actual_list,num_val_img, prev_actual_list,act_year_ids,act_reach_ids,vert_img_hgt,out_mode,
                #                    flag_standardize_actual,transform_constants,output_subtracted,extra_samples)
                pred_list = process_diffs_pred(pred_list,num_val_img, prev_actual_list,act_year_ids,act_reach_ids,vert_img_hgt,out_mode,
                                    flag_standardize_actual,transform_constants,output_subtracted,extra_samples)
                
                if output_subtracted == False :
                    if inp_lr_flag == 'right' :
                        #actual_list[:,:,0] = 0
                        pred_list[:,:,0] = 0 
                    elif inp_lr_flag == 'left' :
                        #actual_list[:,:,1] = 0
                        pred_list[:,:,1] = 0 
                
                #test_logs, test_logs_scores, imp_val_logs = log_performance_metrics(pred_list,actual_list,prev_actual_list,
                #                                    num_val_img, epoch, val_img_ids,writer,erosion_thresh)
                #print('validation reach MAE ......')
                #print(test_logs['reach_mae'])
                #val_maes.append(test_logs['reach_mae'])

    for iter_num in range((1)):
        wrt_test_img(iter_num, prev_actual_list, pred_list, val_img_ids, writer,smooth_flag,
                    reach_start_indx,out_use_mid,vert_img_hgt,year_to_pred)

        """ if iter_num == 0 :
            final_conf = temp_conf
        else :
            final_conf = final_conf + temp_conf """

    hyperparameter_defaults.update(epochs=epoch+1)
    hyperparameter_defaults.update(trial_name=model_name)
    
    if skip_training == False :
        print('calculating training MAE performance ........')
        train_mae = pt_train_per(dataset_tr_pr,inp_mode,inp_lr_flag,out_mode,out_lr_tag,flag_standardize_actual,flag_reach_use,
            batch_size,time_step,vert_img_hgt,flag_sdd_act_data,inp_lines_mean,inp_lines_std,out_mean,out_std,model,loss_func,
            transform_constants,num_val_img,output_subtracted,out_use_mid,flag_use_lines,flag_use_imgs,device)
        print(train_mae)
        train_maes.append(train_mae)
    

    if skip_training == False :
        if out_lr_tag == 'both' :
            temp_hparam = {'hparam/train_loss':avg_epoch_loss,'hparam/left_train_mae':train_mae[0],'hparam/right_train_mae':train_mae[1]}
        elif out_lr_tag == 'left' or out_lr_tag == 'right' :
            temp_hparam = {'hparam/train_loss':avg_epoch_loss,'hparam/train_mae':train_mae}
    elif skip_training == True :
        train_maes = [0]
        pass
    
    """ if skip_training == False :
        hparam_logs = {**temp_hparam,**hparam_logs}
    elif skip_training == True :
        pass """

    #hparam_logs.update(imp_val_logs)
    #writer.add_hparams(hyperparameter_defaults, hparam_logs)
    writer.close()
    if save_mod == True:
        model_save(model, optimizer, model_name)


    return model_name, train_losses, val_losses, train_maes, val_maes, hyperparameter_defaults,transform_constants


def get_all_data():

    data_type = 'img'

    npy_path = os.path.join('./data/img/up_npy/')
    rgb_path = os.path.join('./data/img/up_rgb/')
    rgb_list = [f for f in listdir(rgb_path) if isfile(join(rgb_path, f))]
    rgb_list = [int(f.split('.')[0]) for f in rgb_list]
    rgb_list.sort()
    rgb_list = [str(f) for f in rgb_list]

    infra_path = os.path.join('./data/img/up_infra/')
    msk_path = os.path.join('./data/img/up_msk/')

    concat_img_list = []
    year_id_list = []
    reach_id_list = []
    lines_list = []

    for i in range(len(rgb_list)):
        print(i)
        rgb_img = cv2.imread(rgb_path+rgb_list[i]+'.png',1)
        infra_img = cv2.imread(infra_path+rgb_list[i]+'.png',1)
        msk_img = cv2.imread(msk_path+rgb_list[i]+'.png',0)
        msk_img = np.reshape(msk_img, (msk_img.shape[0],msk_img.shape[1],1))

        concat_img = np.concatenate((rgb_img,infra_img,msk_img), axis=2)
        concat_img_list.append(concat_img)

        year_id = np.asarray(int(rgb_list[i]))
        year_id_list.append(year_id)

        reach_id = np.arange(2222)
        reach_id_list.append(reach_id)

        line_npy = np.load(npy_path+rgb_list[i]+'.npy')
        lines_list.append(line_npy)


    concat_img_list = np.asarray(concat_img_list, dtype=np.uint8)
    year_id_list = np.asarray(year_id_list)
    reach_id_list = np.asarray(reach_id_list)
    lines_list = np.asarray(lines_list)

    print(concat_img_list.shape)
    print(year_id_list.shape)
    print(reach_id_list.shape)
    print(lines_list.shape)

    dataset_dict = {'imgs':concat_img_list, 'years':year_id_list,
            'reaches':reach_id_list, 'lines':lines_list}

    return dataset_dict

def main_program(year_to_pred):
    #print(year_to_pred)
    #print(asd)
    super_epochs = 1
    num_epochs = 2

    def objtv(trial):

        dataset_dic = get_all_data()
        #print(asd)

        super_epochs = 1
        num_epochs = 2

        trail_id = trial.number
        load_models_list = []
        transform_constants_list = []
        #tm_stp=trial.suggest_int('time_step', 3, 6, 1)
        tm_stp = 5
        #lr_pow = trial.suggest_discrete_uniform('learning_rate', -5.0, -3.0, 0.5)
        lr_pow = -4.5
        #lstm_hidden_units = trial.suggest_int('neurons_per_layer', 200, 500, 50 )
        lstm_hidden_units = 50
        #batch_size_pow = trial.suggest_int('batch_size_power', 2, 6 , 1)
        batch_size_pow = 2
        #num_layers = trial.suggest_int('num_of_layers', 3, 5, 1)
        num_layers = 0
        num_cnn_layers = 6
        #strt = trial.suggest_int('starting_year', 0, 20, 5)
        strt = 0
        #vert_hgt = trial.suggest_int('vertical_window_size', 128, 256, 128)
        vert_hgt = 128
        #loss_func = trial.suggest_categorical('loss_function', ['mse_loss', 'l1_loss', 'huber_loss','log_cosh])
        loss_func = 'huber_loss'
        #output_subtracted = trial.suggest_categorical('output_subtracted', [0,False])
        #lstm_layers = trial.suggest_int('lstm_depth_layers', 1, 3, 1)
        lstm_layers = 1
        #model_type = trial.suggest_categorical('model_type', ['ANN', 'LSTM'])
        model_type = 'CNN_LSTM'
        #flag_batch_norm_bin = trial.suggest_int('batch_norm', 0, 1, 1)
        #flag_batch_norm_bin = 0
        flag_dilated_cov = False
        flag_use_lines = True
        flag_use_imgs = True
        flag_bin_out = False
        output_subtracted = False
        lstm_dropout = 0.0
        #pooling_layer = trial.suggest_categorical('pooling_layer', ['MaxPool', 'AvgPool'])
        pooling_layer = 'AvgPool'
        #only_lstm_units = trial.suggest_int('only_lstm_units', 200, 500, 50 )
        only_lstm_units = 150
        #num_branch_layers = trial.suggest_int('num_branch_layers', 2, 10, 2)
        num_branch_layers = 1
        num_lft_brn_lyrs = 0
        num_rgt_brn_lyrs = 0
        #branch_layer_neurons = trial.suggest_int('branch_layer_neurons', 50, 150, 50)
        branch_layer_neurons = 100
        #right_loss_weight = trial.suggest_discrete_uniform('right_loss_weight', 0.5, 0.95, 0.05)
        right_loss_weight = 0.4
        #num_filter_choice = trial.suggest_int('num_filter_choice', 0, 1, 1)
        #num_filter_choice = 2
        #num_filter_list = [4, 8, 16, 32]
        strtn_num_chanls = 16
        model_optim = 'Adam'
        #ad_pow = 1*(10**-3.0)
        ad_pow = 0
        erosion_thresh = 1
        #temp_model_list = ['Nov27_01-50-46_DESKTOP-8SUO90F','Nov27_03-09-25_DESKTOP-8SUO90F',
        #    'Nov27_04-26-01_DESKTOP-8SUO90F','Nov27_05-34-13_DESKTOP-8SUO90F','Nov27_06-38-34_DESKTOP-8SUO90F']

        for j in range(super_epochs):

            cross_val_nums = 1
            val_split_org = int(year_to_pred - 1988)
            val_skip = 0
            out_use_mid = True
            #strt=20
            batch_size = 2**batch_size_pow
            change_start = False
            #batch_size = 2**batch_size_pow
            get_train_mae = num_epochs
            #lr_pow=-3.0
            #ad_pow=1*(10**-1.0)
            
            #vert_hgt=1
            vert_step_num=1
            #num_epochs=num_epochs
            #lstm_layers=1
            #neurons_per_layer_list = [20,50,70,]
            inp_bnk = 'img'
            out_bnk = 'both'
            
            #loss_func='mse_loss'
            
            train_shuffle = True
            train_val_gap = False
            #flg_btch_list = [False, True]
            #flag_batch_norm = flg_btch_list[int(flag_batch_norm_bin)]
            flag_batch_norm = True
            
            #model_type = 'ANN'
            #num_layers_list = [1,3,5,7,9,12,14]
    

            crs_train_ls = []
            #crs_val_ls = []
            crs_train_maes = []
            #crs_val_maes = []
            crs_test_maes = []

            for i in range(cross_val_nums) :
                #val_split = val_split_org + (tm_stp - 2)
                val_split = val_split_org

                if j == 0 :
                    model_name, train_losses, val_losses, train_maes, val_maes, hparam_def, transform_constants = objective(tm_stp=tm_stp,strt=strt,lr_pow=lr_pow,ad_pow=ad_pow,vert_hgt=vert_hgt,vert_step_num=vert_step_num,num_epochs=num_epochs,train_shuffle=train_shuffle,
                    get_train_mae=get_train_mae,transform_constants=None,lstm_layers=lstm_layers,lstm_hidden_units=lstm_hidden_units,batch_size=batch_size,inp_bnk=inp_bnk,out_bnk=out_bnk,val_split=val_split,val_skip=val_skip,model_type=model_type,num_layers=num_layers,
                    model_optim=model_optim,loss_func=loss_func,save_mod=True,load_mod=False,load_file=None,skip_training=False,output_subtracted=output_subtracted,train_val_gap=train_val_gap,out_use_mid=out_use_mid,trail_id=trail_id,flag_batch_norm=flag_batch_norm,dataset_dic=dataset_dic,
                    num_cnn_layers=num_cnn_layers,flag_use_lines=flag_use_lines,pooling_layer=pooling_layer,flag_bin_out=flag_bin_out,only_lstm_units=only_lstm_units,num_branch_layers=num_branch_layers,branch_layer_neurons=branch_layer_neurons,right_loss_weight=right_loss_weight,
                    strtn_num_chanls=strtn_num_chanls,flag_use_imgs=flag_use_imgs,erosion_thresh=erosion_thresh,num_lft_brn_lyrs=num_lft_brn_lyrs,num_rgt_brn_lyrs=num_rgt_brn_lyrs,lstm_dropout=lstm_dropout,flag_dilated_cov=flag_dilated_cov,year_to_pred=year_to_pred)
                elif j > 0 :
                    model_name, train_losses, val_losses, train_maes, val_maes, hparam_def, transform_constants = objective(tm_stp=tm_stp,strt=strt,lr_pow=lr_pow,ad_pow=ad_pow,vert_hgt=vert_hgt,vert_step_num=vert_step_num,num_epochs=num_epochs,train_shuffle=train_shuffle,
                    get_train_mae=get_train_mae,transform_constants=transform_constants_list[i],lstm_layers=lstm_layers,lstm_hidden_units=lstm_hidden_units,batch_size=batch_size,inp_bnk=inp_bnk,out_bnk=out_bnk,val_split=val_split,val_skip=val_skip,model_type=model_type,num_layers=num_layers,
                    model_optim=model_optim,loss_func=loss_func,save_mod=True,load_mod=True,load_file=load_models_list[0],skip_training=False,output_subtracted=output_subtracted,train_val_gap=train_val_gap,out_use_mid=out_use_mid,trail_id=trail_id,flag_batch_norm=flag_batch_norm,dataset_dic=dataset_dic,
                    num_cnn_layers=num_cnn_layers,flag_use_lines=flag_use_lines,pooling_layer=pooling_layer,flag_bin_out=flag_bin_out,only_lstm_units=only_lstm_units,num_branch_layers=num_branch_layers,branch_layer_neurons=branch_layer_neurons,right_loss_weight=right_loss_weight,
                    strtn_num_chanls=strtn_num_chanls,flag_use_imgs=flag_use_imgs,erosion_thresh=erosion_thresh,num_lft_brn_lyrs=num_lft_brn_lyrs,num_rgt_brn_lyrs=num_rgt_brn_lyrs,lstm_dropout=lstm_dropout,flag_dilated_cov=flag_dilated_cov,year_to_pred=year_to_pred)

                    load_models_list.pop(0)

                load_models_list.append(model_name)
                if j == 0 :
                    transform_constants_list.append(transform_constants)
                #print(val_losses)
                crs_train_ls.append(train_losses)
                #crs_val_ls.append(val_losses)
                crs_train_maes.append(train_maes)
                #crs_val_maes.append(val_maes)

                """ if val_skip > 0 :
                    _, _, _, _, test_val_maes, _, _ = objective(tm_stp=tm_stp,strt=strt,lr_pow=lr_pow,ad_pow=ad_pow,vert_hgt=vert_hgt,vert_step_num=vert_step_num,num_epochs=1,train_shuffle=train_shuffle,get_train_mae=1,transform_constants=transform_constants,
                        lstm_layers=lstm_layers,lstm_hidden_units=lstm_hidden_units,batch_size=batch_size,inp_bnk=inp_bnk,out_bnk=out_bnk,val_split=val_split-(val_split_org-val_skip),val_skip=(val_skip-1),model_type=model_type,num_layers=num_layers,
                        model_optim=model_optim,loss_func=loss_func,save_mod=False,load_mod=True,load_file=model_name,skip_training=True,output_subtracted=output_subtracted,train_val_gap=train_val_gap,out_use_mid=out_use_mid,trail_id=trail_id,flag_batch_norm=flag_batch_norm,
                        dataset_dic=dataset_dic,num_cnn_layers=num_cnn_layers,flag_use_lines=flag_use_lines,pooling_layer=pooling_layer,flag_bin_out=flag_bin_out,only_lstm_units=only_lstm_units,num_branch_layers=num_branch_layers,branch_layer_neurons=branch_layer_neurons,
                        right_loss_weight=right_loss_weight,strtn_num_chanls=strtn_num_chanls,flag_use_imgs=flag_use_imgs,erosion_thresh=erosion_thresh,num_lft_brn_lyrs=num_lft_brn_lyrs,num_rgt_brn_lyrs=num_rgt_brn_lyrs,lstm_dropout=lstm_dropout,flag_dilated_cov=flag_dilated_cov)
                
                    crs_test_maes.append(test_val_maes)

                    val_split_org = val_split_org + 1
                    val_skip = val_split_org - 1
                    if change_start == True :
                        strt = strt - 1 """

                #print(asd)
                """ print(crs_train_ls)
                print(crs_val_ls)
                print(crs_train_maes)
                print(crs_val_maes) """

                

            crs_train_ls = np.mean(np.asarray(crs_train_ls),axis=0)
            #crs_val_ls = np.mean(np.asarray(crs_val_ls),axis=0)
            crs_train_maes = np.mean(np.asarray(crs_train_maes),axis=0)
            #crs_val_maes = np.mean(np.asarray(crs_val_maes),axis=0)
            if val_skip > 0 :
                crs_test_mae = np.mean(np.asarray(crs_test_maes),axis=0)

            #print(crs_val_ls)

            writer = SummaryWriter()

            for i in range(crs_train_ls.shape[0]) :
                
                writer.add_scalar('cross_val/train',crs_train_ls[i], i+1)
            
            #for i in range(crs_val_ls.shape[0]) :   
            #    writer.add_scalar('cross_val/val', crs_val_ls[i], (i+1)*get_train_mae)

            if out_bnk == 'right' :
                #crs_val_maes = crs_val_maes[:,1]
                if val_skip > 0 :
                    crs_test_mae = crs_test_mae[:,1]
                crs_train_maes = crs_train_maes[:,1]

            elif out_bnk == 'left' :
                #crs_val_maes = crs_val_maes[:,0]
                if val_skip > 0 :
                    crs_test_mae = crs_test_mae[:,0]
                crs_train_maes = crs_train_maes[:,0]

            elif out_bnk == 'both' :
                #crs_val_maes = (crs_val_maes[:,0] + crs_val_maes[:,1]) / 2
                if val_skip > 0 :
                    crs_test_mae = (crs_test_mae[:,0] + crs_test_mae[:,1]) / 2
                crs_train_maes = (crs_train_maes[:,0] + crs_train_maes[:,1]) / 2

            """ print(crs_train_ls)
            print(crs_train_ls[-1])
            print(crs_val_ls)
            print(crs_val_ls[-1])

            print(crs_train_maes)
            print(crs_train_maes[-1])
            print(crs_val_maes)
            print(crs_val_maes[-1]) """


            #for i in range(crs_val_maes.shape[0]) :
            #    writer.add_scalar('cross_val/Val_Reach_MAEs', crs_val_maes[i], i+1)

            counter = 1
            for i in range(crs_train_maes.shape[0]) :
                writer.add_scalar('cross_val/Train_MAEs', crs_train_maes[i], counter)
                counter += get_train_mae

            if val_skip <= 0 :
                crs_test_mae = -1

            crs_hparam_logs = {'cross_val/crs_train_loss':crs_train_ls[-1],      #'cross_val/crs_val_loss':crs_val_ls[-1],
                        'cross_val/crs_train_MAE':crs_train_maes[-1], #'cross_val/crs_val_MAE':crs_val_maes[-1],
                        'cross_val/crs_test_MAE':crs_test_mae}

            writer.add_hparams(hparam_def, crs_hparam_logs)
            writer.close()

            trial.report(crs_train_maes[-1], ((j+1)*num_epochs))

            if trial.should_prune():
                raise optuna.TrialPruned()


        return crs_train_maes[-1]

    #study = optuna.create_study(study_name='batch_norm',storage='sqlite:///data\\sqdb\\lin_both_fls_man.db',load_if_exists=True,direction='minimize',sampler=RandomSampler(),
    #        pruner=HyperbandPruner(min_resource=1, max_resource=int(super_epochs*num_epochs), reduction_factor=3))

    study = optuna.create_study(study_name='batch_norm',storage='sqlite:///data\\sqdb\\lin_imgs_both_fls_man_2021.db',
            load_if_exists=True,direction='minimize',pruner=NopPruner())

    study.optimize(objtv,n_trials=1)
    #study.optimize(objtv)
    pass

if __name__ == "__main__":

    #main_program(number_of_steps=9)

    pass