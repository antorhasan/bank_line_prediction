import tensorflow as tf
import torch.nn as nn
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from Models import CNN_Model, Baseline_Model, Three_Model
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import itertools
from scipy.signal import savgol_filter
import optuna
from operator import add
from random import randrange


def _parse_function_(example_proto):

    features = {
            'input_tensor': tf.io.FixedLenFeature((), tf.string),
            'reg_coor': tf.io.FixedLenFeature((), tf.string),
            'bin_label': tf.io.FixedLenFeature((), tf.string),
            'year_id': tf.io.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    input_tensor = tf.io.decode_raw(parsed_features["input_tensor"],  tf.float32)
    reg_coor = tf.io.decode_raw(parsed_features["reg_coor"],  tf.float32)
    bin_label = tf.io.decode_raw(parsed_features["bin_label"],  tf.float32)
    year_id = tf.io.decode_raw(parsed_features['year_id'],  tf.float32)
    
    return input_tensor, reg_coor, bin_label, year_id


def plt_conf_mat(conf_mat, title, writer):
    normalize = True
    mat_fig = plt.figure(clear=True)
    cmap = plt.get_cmap('Blues')
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    target_names = ['non-erosion', 'erosion']
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(conf_mat[i, j]),
                     horizontalalignment="center",
                     color="orange")
        else:
            plt.text(j, i, "{:,}".format(conf_mat[i, j]),
                     horizontalalignment="center",
                     color="orange")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    writer.add_figure(title, mat_fig)
    plt.close()

def regress_erro(act_err_bin, act_reg, pred_reg, prev_reg, iter_num, side,writer, val_img_ids,epoch):
    temp_arr = pred_reg - act_reg
    temp_ero_diff = act_reg - prev_reg
    overfit_error = pred_reg - prev_reg
    if side == 'left' :
        temp_arr = temp_arr
    elif side == 'right' :
        temp_arr = -temp_arr



    counter_pos = 0
    counter_neg = 0 
    counter_non = 0
    #pos_deviation = 0
    #neg_deviation = 0

    pos_list = []
    neg_list = []
    non_ero_list = []
    ero_diff = []
    overfit_error_list = []

    for i in range(act_err_bin.shape[0]):
        if act_err_bin[i] == 1 and temp_arr[i]>=0 :
            pos_list.append(temp_arr[i])
            #actual_ero_abs.append(abs(temp_arr[i]))
            #pos_deviation = pos_deviation + temp_arr[i]
            counter_pos += 1
        elif act_err_bin[i] == 1 and temp_arr[i]<0 :
            neg_list.append(-temp_arr[i])
            #actual_ero_abs.append(abs(temp_arr[i]))
            #neg_deviation = neg_deviation + (-temp_arr[i])
            counter_neg += 1
        elif act_err_bin[i] != 1 :
            non_ero_list.append(abs(temp_arr[i]))
            counter_non += 1

        if act_err_bin[i] == 1 :
            ero_diff.append(abs(temp_ero_diff[i]))
            overfit_error_list.append(abs(overfit_error[i]))


    pos_np = np.asarray(pos_list)
    neg_np = np.asarray(neg_list)
    comb_np = np.asarray(pos_list+neg_list)
    non_np = np.asarray(non_ero_list)
    ero_diff_np = np.asarray(ero_diff)

    pos_sum = np.sum(pos_np)
    neg_sum = np.sum(neg_np)
    comb_sum = np.sum(comb_np)
    non_sum = np.sum(non_np)
    ero_diff_sum = np.sum(ero_diff_np)
    overfit_error_sum = np.sum(np.asarray(overfit_error_list))

    pos_std = np.std(pos_np)
    neg_std = np.std(neg_np)
    comb_std = np.std(comb_sum)
    non_std = np.std(non_np)

    if len(pos_list) != 0 :
        pos_max = np.amax(np.asarray(pos_list))
        writer.add_scalar(str(val_img_ids[iter_num])+'/max_pos_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), pos_max, epoch+1)

    if len(neg_list) != 0 :
        neg_max = np.amax(np.asarray(neg_list))
        writer.add_scalar(str(val_img_ids[iter_num])+'/max_neg_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), neg_max, epoch+1)

    try:
        mean_pos_dev = pos_sum/counter_pos
    except:
        mean_pos_dev = pos_sum/.0001
    try:
        mean_neg_dev = neg_sum/counter_neg
    except:
        mean_neg_dev = neg_sum/.0001
    try:
        ero_mae = comb_sum/(counter_pos + counter_neg)
    except:
        ero_mae = comb_sum/.0001
    try:
        non_ero_mae = non_sum/counter_non
    except:
        non_ero_mae = non_sum/.0001
    try:
        mean_ero_diff = ero_diff_sum/(counter_pos + counter_neg)
    except:
        mean_ero_diff = ero_diff_sum/.0001
    
    try:
        mean_act_overfit = overfit_error_sum/(counter_pos + counter_neg)
    except:
        mean_act_overfit = overfit_error_sum/.0001

    reach_mae = np.mean(np.absolute(temp_arr))
    reach_std = np.std(np.absolute(temp_arr))
    
    reach_diff = np.mean(np.absolute(temp_ero_diff))
    reach_overfit = np.mean(np.absolute(overfit_error))

    writer.add_scalar(str(val_img_ids[iter_num])+'/mean_abs_pos_error_for_actual_'+side+'_erosion_'+str(val_img_ids[iter_num]), mean_pos_dev, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mean_abs_neg_error_for_actual_'+side+'_erosion_'+str(val_img_ids[iter_num]), mean_neg_dev, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mae_for_actual_'+side+'_erosion_'+str(val_img_ids[iter_num]), ero_mae, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mae_for_non_error_'+side+str(val_img_ids[iter_num]), non_ero_mae, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/reach_mae'+side+str(val_img_ids[iter_num]), reach_mae, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mean_reach_diff'+side+str(val_img_ids[iter_num]), reach_diff, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mean_act_ero_diff'+side+str(val_img_ids[iter_num]), mean_ero_diff, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mean_act_overfit_'+side+str(val_img_ids[iter_num]), mean_act_overfit, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mean_reach_overfit_'+side+str(val_img_ids[iter_num]), reach_overfit, epoch+1)


    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_pos_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), pos_std, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_neg_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), neg_std, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_comb_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), comb_std, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_non_error_'+side+str(val_img_ids[iter_num]), non_std, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_reach_error'+side+str(val_img_ids[iter_num]), reach_std, epoch+1)

    return mean_pos_dev, pos_std, mean_neg_dev, neg_std, ero_mae, comb_std, non_ero_mae, non_std, reach_mae, reach_std, reach_diff, mean_ero_diff,mean_act_overfit,reach_overfit

def log_perform_lef_rght(log_item, left_comp, right_comp, writer, val_img_ids, iter_num, epoch):
    '''log the various mae and std in calc_fscore'''
    writer.add_scalar('left_'+log_item+'_for_actual_erosion'+str(val_img_ids[iter_num]), left_comp, epoch+1)
    writer.add_scalar('right_'+log_item+'_for_actual_erosion'+str(val_img_ids[iter_num]), right_comp, epoch+1)
    avg_comp = (left_comp + right_comp)/2
    writer.add_scalar('left_right_avg_'+log_item+'_for_actual_erosion'+str(val_img_ids[iter_num]), avg_comp, epoch+1)


def calc_fscore(iter_num, actual_list, prev_actual_list, pred_list, epoch,writer,val_img_ids):
    act_left = actual_list[:,iter_num,0]
    act_right = actual_list[:,iter_num,1]

    prev_left = prev_actual_list[:, iter_num, 0]
    prev_right = prev_actual_list[:, iter_num, 1]

    pred_left = pred_list[:, iter_num, 0]
    pred_right = pred_list[:, iter_num, 1]

    actual_ers_lft = np.reshape(np.where(act_left<prev_left, 1, 0),(act_left.shape[0],1))
    actual_ers_rht = np.reshape(np.where(act_right>prev_right, 1, 0),(act_right.shape[0],1))
    
    left_mae_pos,left_std_pos,left_mae_neg,left_std_neg,lft_cm_m,lft_cm_std,lft_non_m,lft_non_s,lft_r_m,lft_r_s,lft_reach_diff,lft_ero_dif,lft_ove_act,lft_ove_rea = regress_erro(actual_ers_lft, act_left, pred_left, prev_left, iter_num, 'left',writer,val_img_ids,epoch)
    right_mae_pos,right_std_pos,right_mae_neg,right_std_neg,rg_cm_m,rg_cm_std,rg_non_m,rg_non_s,rg_r_m,rg_r_s,rg_reach_diff,rg_ero_dif,rg_ove_act,rg_ove_rea = regress_erro(actual_ers_rht, act_right, pred_right, prev_right, iter_num, 'right',writer,val_img_ids,epoch)

    log_dic_lef_rght = {'pos_mae': [left_mae_pos,right_mae_pos], 'pos_std':[left_std_pos,right_std_pos],
    'neg_mae':[left_mae_neg,right_mae_neg],'neg_std':[left_std_neg,right_std_neg],
    'full_act_mae':[lft_cm_m,rg_cm_m],'full_act_std':[lft_cm_std,rg_cm_std],
    'full_non_erosion_mae':[lft_non_m,rg_non_m],'full_non_erosion_std':[lft_non_s,rg_non_s],
    'reach_mae':[lft_r_m,rg_r_m],'reach_std':[lft_r_s,rg_r_s],
    'reach_diff':[lft_reach_diff,rg_reach_diff],'mean_ero_diff':[lft_ero_dif,rg_ero_dif],
    'reach_overfit_diff':[lft_ove_rea,rg_ove_rea],'act_overfit_diff':[lft_ove_act,rg_ove_act]}

    #for i,j in zip(log_dic_lef_rght.keys(),log_dic_lef_rght.values()):
    #    log_perform_lef_rght(i, j[0], j[1], writer, val_img_ids, iter_num, epoch)

    pred_ers_lft = np.reshape(np.where(pred_left<prev_left, 1, 0),(pred_left.shape[0],1))
    pred_ers_rht = np.reshape(np.where(pred_right>prev_right, 1, 0),(pred_right.shape[0],1))

    """ conf_mat_lft = confusion_matrix(actual_ers_lft, pred_ers_lft)
    conf_mat_rht = confusion_matrix(actual_ers_rht, pred_ers_rht)
    combined_conf = conf_mat_lft + conf_mat_rht """

    act_th_ers_lft = np.reshape(np.where((prev_left-act_left)>3, 1, 0),(act_left.shape[0],1))
    act_th_ers_rht = np.reshape(np.where((act_right-prev_right)>3, 1, 0),(act_right.shape[0],1))

    pred_th_ers_lft = np.reshape(np.where((prev_left-pred_left)>3, 1, 0),(pred_left.shape[0],1))
    pred_th_ers_rht = np.reshape(np.where((pred_right-prev_right)>3, 1, 0),(pred_right.shape[0],1))

    prec_th_lft = precision_score(act_th_ers_lft, pred_th_ers_lft, average='binary')
    recall_th_lft = recall_score(act_th_ers_lft, pred_th_ers_lft, average='binary')
    f1_th_lft = f1_score(act_th_ers_lft, pred_th_ers_lft, average='binary')

    prec_th_rht = precision_score(act_th_ers_rht, pred_th_ers_rht, average='binary')
    recall_th_rht = recall_score(act_th_ers_rht, pred_th_ers_rht, average='binary')
    f1_th_rht = f1_score(act_th_ers_rht, pred_th_ers_rht, average='binary')

    writer.add_scalar(str(val_img_ids[iter_num])+'/left_precision_th_'+ str(val_img_ids[iter_num]), prec_th_lft, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/left_recall_th_'+ str(val_img_ids[iter_num]), recall_th_lft, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/left_f1_th_'+ str(val_img_ids[iter_num]), f1_th_lft, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/right_precision_th_'+ str(val_img_ids[iter_num]), prec_th_rht, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/right_recall_th'+ str(val_img_ids[iter_num]), recall_th_rht, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/right_f1_th_'+ str(val_img_ids[iter_num]), f1_th_rht, epoch+1)

    y_true = np.concatenate((actual_ers_lft,actual_ers_rht), axis = 0)
    y_pred = np.concatenate((pred_ers_lft,pred_ers_rht), axis = 0)

    precision_comb = precision_score(y_true, y_pred, average='binary')
    recall_comb = recall_score(y_true, y_pred, average='binary')
    f1_comb = f1_score(y_true, y_pred, average='binary')
    
    precision_lft = precision_score(actual_ers_lft, pred_ers_lft, average='binary')
    recall_lft = recall_score(actual_ers_lft, pred_ers_lft, average='binary')
    f1_lft = f1_score(actual_ers_lft, pred_ers_lft, average='binary')

    precision_rht = precision_score(actual_ers_rht, pred_ers_rht, average='binary')
    recall_rht = recall_score(actual_ers_rht, pred_ers_rht, average='binary')
    f1_rht = f1_score(actual_ers_rht, pred_ers_rht, average='binary')

    
    writer.add_scalar(str(val_img_ids[iter_num])+'/precision_'+ str(val_img_ids[iter_num]), precision_comb, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/recall_'+ str(val_img_ids[iter_num]), recall_comb, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/f1_score_'+ str(val_img_ids[iter_num]), f1_comb, epoch+1)

    writer.add_scalar(str(val_img_ids[iter_num])+'/left_precision_'+ str(val_img_ids[iter_num]), precision_comb, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/left_recall_'+ str(val_img_ids[iter_num]), recall_comb, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/left_f1_score_'+ str(val_img_ids[iter_num]), f1_comb, epoch+1)

    writer.add_scalar(str(val_img_ids[iter_num])+'/right_precision_'+ str(val_img_ids[iter_num]), precision_comb, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/right_recall_'+ str(val_img_ids[iter_num]), recall_comb, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/right_f1_score_'+ str(val_img_ids[iter_num]), f1_comb, epoch+1)

    avg_reach_mae = (lft_r_m + rg_r_m) / 2
    left_overfit_metric = abs(lft_reach_diff - lft_ove_rea) 
    right_overfit_metric = abs(rg_reach_diff - rg_ove_rea)
    avg_overfit_metric = (left_overfit_metric + right_overfit_metric) / 2
    augmented_metric = avg_reach_mae + avg_overfit_metric
    lr_f1_score = f1_comb

    writer.add_scalar('AM_score/'+str(val_img_ids[iter_num])+'_augmented_metric_', augmented_metric, epoch+1)
    writer.add_scalar('F1_score/'+str(val_img_ids[iter_num])+'_f1_score_', lr_f1_score, epoch+1)
    writer.add_scalar('Reach_MAE/'+str(val_img_ids[iter_num])+'_lr_reach_mae_', avg_reach_mae, epoch+1)

    log_dic_scores = {'AM_score':augmented_metric, 'lr_f1_score':lr_f1_score, 'lr_reach_mae':avg_reach_mae,
    'left_precision':precision_lft,'left_recall':recall_lft,'left_f1':f1_lft,
    'right_precision':precision_rht,'right_recall':recall_rht,'right_f1':f1_rht,
    'lft_rht_precision':precision_comb,'lft_rgt_recall':recall_comb,'lft_rht_f1':f1_comb,
    'left_prec_th':prec_th_lft,'left_recall_th':recall_th_lft,'left_f1_th':f1_th_lft,
    'right_prec_th':prec_th_rht,'right_recall_th':recall_th_rht,'right_f1_th':f1_th_rht}

    imp_val_log = {str(val_img_ids[iter_num])+'_augmented_metric':augmented_metric, str(val_img_ids[iter_num])+'_lr_f1_score':lr_f1_score,
                str(val_img_ids[iter_num])+'_lr_reach_mae':avg_reach_mae,
                str(val_img_ids[iter_num])+'_left_reach_mae':lft_r_m, str(val_img_ids[iter_num])+'_right_reach_mae':rg_r_m,
                str(val_img_ids[iter_num])+'_left_f1score':f1_lft, str(val_img_ids[iter_num])+'_right_f1score':f1_rht,
                str(val_img_ids[iter_num])+'_left_erosion_diff':lft_ero_dif, str(val_img_ids[iter_num])+'_right_erosion_diff':rg_ero_dif,
                str(val_img_ids[iter_num])+'_left_act_erosion_mae':lft_cm_m, str(val_img_ids[iter_num])+'_right_act_erosion_mae':rg_cm_m,
                str(val_img_ids[iter_num])+'_left_f1_th':f1_th_lft, str(val_img_ids[iter_num])+'_right_f1_th':f1_th_rht,
                str(val_img_ids[iter_num])+'_left_act_overfit_diff':lft_ove_act,str(val_img_ids[iter_num])+'_right_act_overfit_diff':rg_ove_act}
    #return avg_mae_pos, avg_std_pos, avg_mae_neg, avg_std_neg, precision_comb, recall_comb, f1_comb

    return log_dic_lef_rght, log_dic_scores, imp_val_log



def wrt_img(iter_num, actual_list, prev_actual_list, pred_list, val_img_ids,writer,smooth_flag):
    
    num_rows = int(pred_list.shape[0])

    actual_ers_lft = np.reshape(np.where(actual_list[:,iter_num,0]<prev_actual_list[:,iter_num,0], 1, 0),(actual_list[:,iter_num,0].shape[0],1))
    actual_ers_rht = np.reshape(np.where(actual_list[:,iter_num,1]>prev_actual_list[:,iter_num,1], 1, 0),(actual_list[:,iter_num,1].shape[0],1))

    pred_ers_lft = np.reshape(np.where(pred_list[:,iter_num,0]<prev_actual_list[:,iter_num,0], 1, 0),(pred_list[:,iter_num,0].shape[0],1))
    pred_ers_rht = np.reshape(np.where(pred_list[:,iter_num,1]>prev_actual_list[:,iter_num,1], 1, 0),(pred_list[:,iter_num,1].shape[0],1))

    conf_mat_lft = confusion_matrix(actual_ers_lft, pred_ers_lft)
    conf_mat_rht = confusion_matrix(actual_ers_rht, pred_ers_rht)
    combined_conf = conf_mat_lft + conf_mat_rht

    plt_conf_mat(conf_mat_lft, str(val_img_ids[iter_num])+'_conf_mat_left', writer)
    plt_conf_mat(conf_mat_rht, str(val_img_ids[iter_num])+'_conf_mat_right', writer)
    plt_conf_mat(combined_conf, str(val_img_ids[iter_num])+'_combined_conf_mat', writer)

    denoising = smooth_flag
    window = 99
    poly = 2

    if denoising :
        try :
            pred_left_den = savgol_filter(pred_list[:,iter_num,0], window, poly)
            pred_right_den = savgol_filter(pred_list[:,iter_num,1], window, poly)
        except :
            pred_left_den = pred_list[:,iter_num,0]
            pred_right_den = pred_list[:,iter_num,1]

    img = cv2.imread(os.path.join('./data/img/up_rgb/'+str(val_img_ids[iter_num])+'.png'), 1)
    for i in range(num_rows):

        img[i,int(actual_list[i,iter_num,0]),:] = [255,255,255]
        img[i,int(actual_list[i,iter_num,1]),:] = [255,255,255]

        if 0<=int(pred_list[i,iter_num,0])<=744 :
            pass
        else :
            pred_list[i,iter_num,0] = 0

        if 0<=int(pred_list[i,iter_num,1])<=744 :
            pass
        else :
            pred_list[i,iter_num,1] = 744

        if denoising :
            if 0<=int(pred_left_den[i])<=744 :
                pass
            else :
                pred_left_den[i] = 0 

            if 0<=int(pred_right_den[i])<=744 : 
                pass
            else :
                pred_right_den[i] = 744


        """ if actual_ers_lft[i] == 1 :
            img[i,int(prev_actual_list[i,iter_num,0]),:] = [255,0,0]
            img[i,int(pred_list[i,iter_num,0]),:] = [0,255,0]
            if denoising:
                img[i,int(pred_left_den[i]),:] = [0,0,255]
            

        if actual_ers_rht[i] == 1 :
            img[i,int(prev_actual_list[i,iter_num,1]),:] = [255,0,0]
            img[i,int(pred_list[i,iter_num,1]),:] = [0,255,0]
            if denoising:
                img[i,int(pred_right_den[i]),:] = [0,0,255] """

        img[i,int(prev_actual_list[i,iter_num,0]),:] = [255,0,0]
        img[i,int(prev_actual_list[i,iter_num,1]),:] = [255,0,0]
        img[i,int(pred_list[i,iter_num,0]),:] = [0,255,0]
        img[i,int(pred_list[i,iter_num,1]),:] = [0,255,0]
        if denoising:
            img[i,int(pred_left_den[i]),:] = [0,0,255]
            img[i,int(pred_right_den[i]),:] = [0,0,255]
            
    writer.add_image(str(val_img_ids[iter_num]), img, dataformats='HWC')
    return combined_conf

def process_val(arr_list, num_val_img, msk_mean, msk_std):
    arr_list = np.asarray(arr_list)
    total_smpls = int(arr_list.shape[0])*int(arr_list.shape[1])
    val_num_rows = int(total_smpls/num_val_img)
    arr_list = np.resize(arr_list, (val_num_rows,num_val_img,2))
    arr_list = (msk_std * arr_list) + msk_mean

    return arr_list

def model_save(model, optimizer, model_name):
    print('saving model....')
    model_path = os.path.join('./data/model/'+model_name +'.pt')
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, model_path)

def log_performance_metrics(pred_list,actual_list,prev_actual_list,num_val_img, epoch, msk_mean, msk_std, val_img_ids,writer):
    print('logging performance metrics........')
    imp_val_logs = {}
    for i in range(num_val_img):
        log_dic_lef_rght, log_dic_scores, imp_val_log_ind = calc_fscore(i, actual_list, prev_actual_list, pred_list, epoch,writer,val_img_ids)
        imp_val_logs.update(imp_val_log_ind)
        if i == 0 :
            test_logs = log_dic_lef_rght
            test_logs_scores = log_dic_scores
        else :
            for i in test_logs.keys() :
                test_logs[i] = list(map(add, test_logs[i], log_dic_lef_rght[i]))
            for i in test_logs_scores.keys() :
                test_logs_scores[i] = test_logs_scores[i] + log_dic_scores[i]

    for i in test_logs.keys() :
        test_logs[i] = [x/num_val_img for x in test_logs[i]]
    for i in test_logs_scores.keys() :
        test_logs_scores[i] = test_logs_scores[i] / num_val_img
    
    for i in test_logs.keys() :
        writer.add_scalar('test_lr/test_set_left'+ i, test_logs[i][0], epoch+1)
        writer.add_scalar('test_lr/test_set_right'+ i, test_logs[i][1], epoch+1)

    for i in test_logs_scores.keys() :
        writer.add_scalar('test_scores/test_set_'+ i, test_logs_scores[i], epoch+1)


    #return test_pos_mae, test_pos_std,test_neg_mae,test_neg_std, test_prec, test_recall, test_f1_score
    return test_logs, test_logs_scores, imp_val_logs

def objective(tm_stp, strt, lr_pow, ad_pow, vert_hgt, vert_step_num, num_epochs,
                lstm_layers):
    
    load_mod = False
    save_mod = False
    #total_window = 52
    num_lstm_layers = lstm_layers
    num_channels = 7
    lf_rt_tag = 'both'
    EPOCHS = num_epochs
    data_mode = 'imgs' 
    #EPOCHS = trial.suggest_discrete_uniform('epochs', 100, 150, 5)
    #EPOCHS = int(EPOCHS)
    #lr_pow = trial.suggest_discrete_uniform('lr_power',-5 , -3, 0.2)
    #print(1*(10**lr_pow))
    #print(asd)
    #lr_rate = trial.suggest_loguniform('lr_rate', .00005, .0005)                       #.0001
    #####lr_rate = trial.suggest_uniform('lr_rate', .0001, .0005)
    #lr_rate = 0.000932098670370034
    #lr_rate = lr_exp
    #lr_rate = 0.000158489319246111

    lr_rate = 1*(10**lr_pow)
    #vert_img_hgt = int(trial.suggest_discrete_uniform('vert_hgt', 3,5,2))
    vert_img_hgt = vert_hgt
    vert_step = vert_step_num            #vert skip step
    #print(vert_img_hgt)
    #print(asd)
    model_type = 'CNN_Model_dropout_reg'
    #input_seed = 0
    wgt_seed_flag = True

    """ dr_1 = trial.suggest_discrete_uniform('drop_out_1',0.07, 0.17, 0.02)
    dr_2 = trial.suggest_discrete_uniform('drop_out_2',0.1, 0.2, 0.02)
    dr_3 = trial.suggest_discrete_uniform('drop_out_3',0.1, 0.2, 0.02)
    dr_4 = trial.suggest_discrete_uniform('drop_out_4',0.19, 0.29, 0.02)
    dr_5 = trial.suggest_discrete_uniform('drop_out_5',0.36, 0.46, 0.02)
    dr_6 = trial.suggest_discrete_uniform('drop_out_6',0.39, 0.49, 0.02)
    dr_7 = trial.suggest_discrete_uniform('drop_out_7',0.28, 0.38, 0.02)
    dr_8 = trial.suggest_discrete_uniform('drop_out_8',0.5, 0.6, 0.02)
    dr_9 = trial.suggest_discrete_uniform('drop_out_9',0.15, 0.25, 0.02)
    dr_10 = trial.suggest_discrete_uniform('drop_out_10',0.26, 0.36, 0.02)
    dr_11 = trial.suggest_discrete_uniform('drop_out_11',0.4, 0.5, 0.02)
    dr_12 = trial.suggest_discrete_uniform('drop_out_12',0.22, 0.32, 0.02) """
    
    dr_1 = 0
    dr_2 = 0
    dr_3 = 0
    dr_4 = 0
    dr_5 = 0
    dr_6 = 0
    dr_7 = 0
    dr_8 = 0
    dr_9 = 0
    dr_10 = 0
    dr_11 = 0
    dr_12 = 0

    drop_rate = [dr_1,dr_2,dr_3,dr_4,dr_5,dr_6,dr_7,dr_8,dr_9,dr_10,dr_11,dr_12]
    smooth_flag = False
    #time_step = trial.suggest_int('time_step', 16, 20)
    time_step = tm_stp
    batch_size = int(int((500/time_step) - 2)/vert_img_hgt)
    batch_size = 16
    val_batch_size = batch_size
    total_time_step = 33    ###number of total year images
    log_performance = 1 ###number of epochs after which performance metrics are calculated
    model_save_at = 50     ###number of epochs after which to save model
    early_stop_flag = False
    early_stop_thresh = 30
    #val_img_ids = [201701, 201801, 201901, 202001]
    path_to_val_img = os.path.join('./data/img/up_rgb/')
    val_img_ids = [int(f.split('.')[0]) for f in listdir(path_to_val_img) if isfile(join(path_to_val_img, f))]
    val_img_ids.sort()
    org_val_img = val_img_ids
    #start_indx = org_val_img.index(200501)    ###year to start training set from
    #start_indx = trial.suggest_int('start_indx', 0, 27)
    #start_indx = randrange(0,25,5)
    start_indx = strt
    
    #print(start_indx) 
    #print(org_val_img[start_indx])
    #print(asd)
    #division_parameter = trial.suggest_int('great_division', 1, 10)
    val_img_ids = val_img_ids[-5:]
    #print(val_img_ids)

    #print(asd)
    num_val_img = len(val_img_ids)
    data_div_step = total_time_step - num_val_img
    end_indx = data_div_step-1
    #print(end_indx-start_indx+1)
    #print(asd)
    log_hist = 5
    writer = SummaryWriter()
    model_name = writer.get_logdir().split("\\")[1]
    adm_wd = 1*(10**ad_pow)
    #adm_wd = 1*(10**adm_wd_exp)
    val_img_range = time_step+num_val_img-1
    #print(val_img_range)
    #print(asd)
    output_vert_indx = int((vert_img_hgt-1)/2)

    hyperparameter_defaults = dict(
        drop_1 = dr_1,
        drop_2 = dr_2,
        drop_3 = dr_3,
        drop_4 = dr_4,
        drop_5 = dr_5,
        drop_6 = dr_6,
        drop_7 = dr_7,
        drop_8 = dr_8,
        drop_9 = dr_9,
        drop_10 = dr_10,
        drop_11 = dr_11,
        drop_12 = dr_12,
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
        reach_id = lf_rt_tag,
        vertical_pix_step = vert_step,
        input_data = data_mode
        )

    dataset_f = tf.data.TFRecordDataset(os.path.join('./data/tfrecord/comp_tf.tfrecords'))
    dataset_f = dataset_f.window(size=data_div_step, shift=total_time_step, stride=1, drop_remainder=False)
    dataset_f = dataset_f.map(lambda x: x.skip(start_indx))
    dataset_f = dataset_f.window(size=vert_img_hgt, shift=vert_step, stride=1,drop_remainder=True)
    dataset_f = dataset_f.map(lambda x: x.flat_map(lambda x1: x1))
    dataset_f = dataset_f.map(lambda x: x.window(size=vert_img_hgt,shift=1,stride=end_indx-start_indx+1,drop_remainder=True))
    dataset_f = dataset_f.map(lambda x: x.window(size=time_step, shift=1, stride=1,drop_remainder=True))
    dataset_f = dataset_f.flat_map(lambda x: x)
    dataset_f = dataset_f.flat_map(lambda x: x.flat_map(lambda x1: x1))
    dataset_f = dataset_f.map(_parse_function_).batch(vert_img_hgt).batch(time_step)
    dataset_f = dataset_f.shuffle(10000, reshuffle_each_iteration=True)
    dataset_f = dataset_f.batch(batch_size, drop_remainder=True)

    dataseti1 = tf.data.TFRecordDataset(os.path.join('./data/tfrecord/comp_tf.tfrecords'))
    dataseti1 = dataseti1.window(size=total_time_step, shift=total_time_step, stride=1, drop_remainder=False)
    dataseti1 = dataseti1.map(lambda x: x.skip(total_time_step-val_img_range))
    dataseti1 = dataseti1.window(size=vert_img_hgt, shift=1, stride=1,drop_remainder=True)
    dataseti1 = dataseti1.map(lambda x: x.flat_map(lambda x1: x1))
    dataseti1 = dataseti1.map(lambda x: x.window(size=vert_img_hgt,shift=1,stride=val_img_range,drop_remainder=True))
    dataseti1 = dataseti1.map(lambda x: x.window(size=time_step, shift=1, stride=1,drop_remainder=True))
    dataseti1 = dataseti1.flat_map(lambda x: x)
    dataseti1 = dataseti1.flat_map(lambda x: x.flat_map(lambda x1: x1))
    dataseti1 = dataseti1.map(_parse_function_).batch(vert_img_hgt).batch(time_step)
    dataseti1 = dataseti1.batch(val_batch_size, drop_remainder=True)

    if wgt_seed_flag :
        torch.manual_seed(0)

    if data_mode == 'imgs' :
        model = CNN_Model(num_channels, batch_size, val_batch_size,time_step, num_lstm_layers, drop_rate,vert_img_hgt,lf_rt_tag)
    if data_mode == 'lines' :
        model = Three_Model(num_channels, batch_size, val_batch_size,time_step, num_lstm_layers, drop_rate, vert_img_hgt, lf_rt_tag)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=adm_wd)

    if load_mod == True:
        checkpoint = torch.load(os.path.join('./data/model/f_temp.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    msk_mean = np.load(os.path.join('./data/mean_img/line_mean.npy'))
    msk_std = np.load(os.path.join('./data/mean_img/line_std.npy'))

    early_stop_counter = 0
    global_train_counter = 0
    for epoch in range(EPOCHS):

        model.train()
        counter = 0
        epoch_loss = 0

        for input_tensor, reg_coor, _ , year_id in dataset_f:
            #year_id = np.reshape(year_id, (batch_size,time_step,vert_img_hgt,1))
            #print(year_id)
            #print(asd)

            if data_mode == 'imgs' :
                input_tensor = np.reshape(input_tensor, (batch_size,time_step,vert_img_hgt,745,num_channels))
                input_tensor = input_tensor[:,0:time_step-1,:,:,:]
            elif data_mode == 'lines' :
                input_tensor = np.reshape(reg_coor, (batch_size,time_step,vert_img_hgt,2))
                input_tensor = input_tensor[:,0:time_step-1,:,:]

            reg_coor = np.reshape(reg_coor, (batch_size,time_step,vert_img_hgt,2))
            
            if lf_rt_tag == 'left':
                reg_coor = reg_coor[:,time_step-1:time_step,output_vert_indx:output_vert_indx+1,0:1]
            elif lf_rt_tag == 'right':
                reg_coor = reg_coor[:,time_step-1:time_step,output_vert_indx:output_vert_indx+1,1:2]
            elif lf_rt_tag == 'both':
                reg_coor = reg_coor[:,time_step-1:time_step,output_vert_indx:output_vert_indx+1,:]
            
            input_tensor = torch.Tensor(input_tensor).cuda().requires_grad_(False)
            reg_coor = torch.Tensor(reg_coor).cuda().requires_grad_(False)
            reg_coor = torch.reshape(reg_coor, (batch_size,-1))
            
            optimizer.zero_grad()
            pred = model(input_tensor)
            #print(pred.size())
            #print(reg_coor.size())
            #print(asd)
            loss = F.mse_loss(pred, reg_coor,reduction='mean')
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss+loss
            counter += 1
            global_train_counter += 1

        avg_epoch_loss = epoch_loss / counter
        template = 'Epoch {}, Train Loss: {}'
        print(template.format(epoch+1,avg_epoch_loss))

        #print(asd)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch+1)

        if epoch == 0 :
            writer.add_graph(model, input_tensor)

        if epoch % log_hist == log_hist-1:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram('parameters_'+str(name), param.data, epoch + 1)
                    writer.add_histogram('gradients'+str(name), param.grad, epoch + 1)

        #if save_mod == True :
        #    if epoch % model_save_at == 0:
        #        model_save(model, optimizer, model_name)
            
        model.eval()

        val_epoch_loss = 0
        counter_val = 0
        
        with torch.no_grad():
            
            if epoch % log_performance == log_performance-1:
                pred_list = []
                actual_list = []
                prev_actual_list = []

            for input_tensor, reg_coor, _ , year_id in dataseti1:
                #year_id = np.reshape(year_id, (batch_size,time_step,vert_img_hgt,1))
                #print(year_id)
                #rint(asd)
                if data_mode == 'imgs' :
                    input_tensor = np.reshape(input_tensor, (batch_size,time_step,vert_img_hgt,745,num_channels))
                    input_tensor = input_tensor[:,0:time_step-1,:,:,:]
                elif data_mode == 'lines' :
                    input_tensor = np.reshape(reg_coor, (batch_size,time_step,vert_img_hgt,2))
                    input_tensor = input_tensor[:,0:time_step-1,:,:]

                #reg_coor = np.reshape(reg_coor, (batch_size,time_step,vert_img_hgt,2))

                prev_time_step = reg_coor[:,time_step-2:time_step-1,output_vert_indx:output_vert_indx+1,:]

                if lf_rt_tag == 'left':
                    prev_time_step = reg_coor[:,time_step-2:time_step-1,output_vert_indx:output_vert_indx+1,0:1]
                    reg_coor = reg_coor[:,time_step-1:time_step,output_vert_indx:output_vert_indx+1,0:1]
                
                elif lf_rt_tag == 'right':
                    prev_time_step = reg_coor[:,time_step-2:time_step-1,output_vert_indx:output_vert_indx+1,1:2]
                    reg_coor = reg_coor[:,time_step-1:time_step,output_vert_indx:output_vert_indx+1,1:2]
                
                elif lf_rt_tag == 'both':
                    prev_time_step = reg_coor[:,time_step-2:time_step-1,output_vert_indx:output_vert_indx+1,:]
                    reg_coor = reg_coor[:,time_step-1:time_step,output_vert_indx:output_vert_indx+1,:]

                prev_time_step = np.reshape(prev_time_step,(batch_size,-1))

                input_tensor = torch.Tensor(input_tensor).cuda()
                reg_coor = torch.Tensor(reg_coor).cuda()
                reg_coor = torch.reshape(reg_coor, (batch_size,-1))
                #prev_time_step = torch.reshape(prev_time_step, (batch_size,-1))

                pred = model(input_tensor)
                
                loss = F.mse_loss(pred, reg_coor,reduction='mean')

                val_epoch_loss = val_epoch_loss+loss
                counter_val += 1

                if epoch % log_performance == log_performance-1:
                    reg_coor = reg_coor.cpu()
                    reg_coor = reg_coor.numpy()
                    pred_np = pred.cpu()
                    pred_np = pred_np.numpy()

                    if lf_rt_tag == 'left' :
                        np_zero = np.zeros((batch_size,1))
                        prev_time_step = np.concatenate((prev_time_step,np_zero),axis=1) 
                        reg_coor = np.concatenate((reg_coor,np_zero),axis=1)
                        pred_np = np.concatenate((pred_np,np_zero),axis=1)
                    elif lf_rt_tag == 'right' :
                        np_zero = np.zeros((batch_size,1))
                        prev_time_step = np.concatenate((np_zero,prev_time_step),axis=1) 
                        reg_coor = np.concatenate((np_zero,reg_coor),axis=1)
                        pred_np = np.concatenate((np_zero,pred_np),axis=1)
                    
                    prev_actual_list.append(prev_time_step)
                    actual_list.append(reg_coor)
                    pred_list.append(pred_np)

            avg_val_epoch_loss = val_epoch_loss / counter_val
            template = 'Epoch {}, Val_Loss: {}'
            print(template.format(epoch+1,avg_val_epoch_loss))

            writer.add_scalar('Loss/Val', avg_val_epoch_loss, epoch+1)

            #writer.add_scalar('Loss/variance', (avg_epoch_loss - avg_val_epoch_loss), epoch+1)
            writer.add_scalars('Loss/train_val', {'train':avg_epoch_loss,
                                    'val':avg_val_epoch_loss}, epoch+1)
            #writer.add_scalars('Loss/bias_variance', {'bias':avg_epoch_loss,
            #                        'variance':(avg_epoch_loss - avg_val_epoch_loss)}, epoch+1)
            if epoch == 0 :
                best_val_loss = avg_val_epoch_loss
            else :
                if avg_val_epoch_loss < best_val_loss :
                    best_val_loss = avg_val_epoch_loss
                    early_stop_counter = 0
                else :
                    early_stop_counter += 1

            ###logging performance metrics
            if epoch % log_performance == log_performance-1 :
                pred_list = process_val(pred_list,num_val_img, msk_mean, msk_std)
                actual_list = process_val(actual_list,num_val_img, msk_mean, msk_std)
                prev_actual_list = process_val(prev_actual_list,num_val_img, msk_mean, msk_std)
                test_logs, test_logs_scores, imp_val_logs = log_performance_metrics(pred_list,actual_list,prev_actual_list,
                                                    num_val_img, epoch, msk_mean, msk_std, val_img_ids,writer)
                
        if early_stop_flag:
            if early_stop_counter > early_stop_thresh :
                print('early stopping as val loss is not improving ........')
                #model_save(model, optimizer, model_name)
                break

    for iter_num in range(num_val_img):
        temp_conf = wrt_img(iter_num, actual_list, prev_actual_list, pred_list, val_img_ids, writer,smooth_flag)

        if iter_num == 0 :
            final_conf = temp_conf
        else :
            final_conf = final_conf + temp_conf
        
    plt_conf_mat(final_conf, 'total_test_confusion_matrix', writer) #'trial_name':str(model_name),
    hyperparameter_defaults.update(epochs=epoch+1)
    hyperparameter_defaults.update(trial_name=model_name)
    
    hparam_logs = {'hparam/train_loss':avg_epoch_loss,'hparam/val_loss':avg_val_epoch_loss,
        'hparam/augmented_metric':test_logs_scores['AM_score'], 'hparam/lr_f1_score':test_logs_scores['lr_f1_score'], 'hparam/lr_reach_mae':test_logs_scores['lr_reach_mae'],
        'hparam/left_reach_mae':test_logs['reach_mae'][0],'hparam/right_reach_mae':test_logs['reach_mae'][1],
        'hparam/left_f1score':test_logs_scores['left_f1'],'hparam/right_f1score':test_logs_scores['right_f1'],
        'hparam/left_pos_mae':test_logs['pos_mae'][0],'hparam/right_pos_mae':test_logs['pos_mae'][1],
        'hparam/left_non_erosion_mae':test_logs['full_non_erosion_mae'][0],'hparam/right_non_erosion_mae':test_logs['full_non_erosion_mae'][1],
        'hparam/left_neg_mae':test_logs['neg_mae'][0],'hparam/right_neg_mae':test_logs['neg_mae'][1],
        'hparam/left_pos_neg_mae':test_logs['full_act_mae'][0],'hparam/right_pos_neg_mae':test_logs['full_act_mae'][1],
        'hparam/left_act_ero_diff':test_logs['mean_ero_diff'][0],'hparam/right_act_ero_diff':test_logs['mean_ero_diff'][1],
        'hparam/left_act_overfit_dif':test_logs['act_overfit_diff'][0],'hparam/right_act_overfit_dif':test_logs['act_overfit_diff'][1]}
    hparam_logs.update(imp_val_logs)
    writer.add_hparams(hyperparameter_defaults, hparam_logs)
    writer.close()
    if save_mod == True:
        model_save(model, optimizer, model_name)

    avg_act_ero_mae = (test_logs['full_act_mae'][0] + test_logs['full_act_mae'][1] ) / 2 

    return avg_val_epoch_loss, avg_act_ero_mae, lr_pow


if __name__ == "__main__":
    #objective(2, 30, -5.0, 0, 1,1,7)
    #print(asd)
    #study = optuna.create_study(direction='minimize',sampler= optuna.samplers.RandomSampler())
    #study = optuna.create_study(direction='minimize',sampler=optuna.samplers.GridSampler(search_space))
    #study.optimize(objective, n_trials=10) 
    #objective()

    #objective(5,27,-5.0,0,15)
    #print(asd)
    
    #ad_pow = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    #lr_pow = [-3.8]
    #print(lr_pow)
    #vert_hgt = np.arange(1,23,2)
    #tm_stp_list = [5, 10, 15, 20, 25, 30]
    #tm_stp_list = [5]
    #strt_list = [27, 21, 15, 7, 0]
    #strt_list = [27]
    #vert_hgt = [5,7,9]
    #lr_rate_exp = [0.000251188643150958,0.000251188643150958,0.0000794328234724275]
    #print(int(vert_hgt[2]))
    #print(asd)
    
    #lowest_mae_list = [5.299081087, 6.96048522, 6.942534924, 6.488059044, 6.468466282]
    
    #lr_pow = np.arange(-5.0, -1.8, 0.4)
    #vert_hgt = [9]
    #strt_list = [27, 21, 15, 7, 0]
    #tm_stp_list = [5]
    #strt_list = [27]

    #objective(tm_stp=5,strt=0,lr_pow=-4.0,ad_pow=0,vert_hgt=1,vert_step_num=1,num_epochs=150,lstm_layers=2)
    #print(asd)

    #lr_rate_list = [-5.0,-4.0,-3.0,-2.0,-1.0]
    #lr_rate_list = np.arange(-5.0, -2.8, 1.0)
    #ad_pow = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    ad_pow = np.arange(-3.0, -1.8, 1.0)
    for i in range(len(ad_pow)):
        objective(tm_stp=5,strt=0,lr_pow=-4.0,ad_pow=ad_pow[i],vert_hgt=1,vert_step_num=1,num_epochs=20,lstm_layers=2)
    #print('hidden units ',ls_hid_list[i])
    """ for i in range(len(strt_list)):
        mae_count = 0
        for j in range(len(lr_pow)):
            val_loss, temp_mae = objective(5, strt_list[i], float(lr_pow[j]), 0, 9,1,30)
            print(lr_pow[j], strt_list[i])
            if j == 0 :
                lowest_mae = temp_mae
                continue
            
            if temp_mae < lowest_mae or temp_mae == lowest_mae :
                lowest_mae = temp_mae
            elif temp_mae > lowest_mae :
                mae_count += 1
            
            if mae_count == 3 :
                break """




    """ for i in range(len(vert_hgt)):
        for j in range(len(wd_pow)):
            objective(int(vert_hgt[i]), wd_pow[j], lr_rate_exp[i])
            print(wd_pow[j],vert_hgt[i]) """        

    pass