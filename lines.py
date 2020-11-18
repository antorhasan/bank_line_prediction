#import tensorflow as tf
import torch.nn as nn
import torch
import numpy as np
np.seterr(all='raise')
import cv2
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from models import Baseline_LSTM_Dynamic_Model,Baseline_ANN_Dynamic_Model,CNN_LSTM_Dynamic_Model
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



def _parse_function_(example_proto):

    features = {
            'sdd_input': tf.io.FixedLenFeature((), tf.string),
            'year_id': tf.io.FixedLenFeature((), tf.string),
            'reach_id': tf.io.FixedLenFeature((), tf.string),
            'bin_class': tf.io.FixedLenFeature((), tf.string),
            'sdd_output':tf.io.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    input_tensor = tf.io.decode_raw(parsed_features["sdd_input"],  tf.float32)
    year_id = tf.io.decode_raw(parsed_features['year_id'],  tf.float32)
    reach_id = tf.io.decode_raw(parsed_features["reach_id"],  tf.float32)
    bin_class = tf.io.decode_raw(parsed_features["bin_class"],  tf.float32)
    sdd_output = tf.io.decode_raw(parsed_features["sdd_output"],  tf.float32)
    
    return input_tensor, year_id, reach_id, bin_class, sdd_output


def _parse_function_org_(example_proto):

    features = {
            'reg_coor': tf.io.FixedLenFeature((), tf.string),
            'year_id': tf.io.FixedLenFeature((), tf.string),
            'reach_id': tf.io.FixedLenFeature((), tf.string),
            'bin_class': tf.io.FixedLenFeature((), tf.string),
            'reach_diff':tf.io.FixedLenFeature((), tf.string),
            'concat_img':tf.io.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    input_tensor = tf.io.decode_raw(parsed_features["reg_coor"],  tf.float32)
    year_id = tf.io.decode_raw(parsed_features['year_id'],  tf.float32)
    reach_id = tf.io.decode_raw(parsed_features["reach_id"],  tf.float32)
    bin_class = tf.io.decode_raw(parsed_features["bin_class"],  tf.float32)
    sdd_output = tf.io.decode_raw(parsed_features["reach_diff"],  tf.float32)
    concat_img = tf.io.decode_raw(parsed_features["concat_img"],  tf.float32)
    
    return input_tensor, year_id, reach_id, bin_class, sdd_output, concat_img


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
    #print(pred_reg)
    #print(act_reg)
    #print(asd)

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

    try :
        pos_std = np.std(pos_np)
        neg_std = np.std(neg_np)
        comb_std = np.std(comb_sum)
        non_std = np.std(non_np)
    except FloatingPointError :
        pos_std = 0
        neg_std = 0
        comb_std = 0
        non_std = 0
        #print('error raised')

    if len(pos_list) != 0 :
        pos_max = np.amax(np.asarray(pos_list))
        writer.add_scalar(str(val_img_ids[iter_num])+'/max_pos_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), pos_max, epoch+1)

    if len(neg_list) != 0 :
        neg_max = np.amax(np.asarray(neg_list))
        writer.add_scalar(str(val_img_ids[iter_num])+'/max_neg_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), neg_max, epoch+1)

    #np.seterr(all='raise')
    #print(counter_pos)
    try :
        mean_pos_dev = pos_sum/counter_pos
    except :
        mean_pos_dev = pos_sum/.0001

    #print(asd)
    try:
        mean_neg_dev = neg_sum/counter_neg
        #print(mean_neg_dev)
    except:
        mean_neg_dev = neg_sum/.0001
    try:
        ero_mae = comb_sum/(counter_pos + counter_neg)
        #print(ero_mae)
    except:
        ero_mae = comb_sum/.0001
    """ try:
        non_ero_mae = non_sum/counter_non
    except:
        non_ero_mae = non_sum/.0001 """
    """ try:
        mean_ero_diff = ero_diff_sum/(counter_pos + counter_neg)
    except:
        mean_ero_diff = ero_diff_sum/.0001
    
    try:
        mean_act_overfit = overfit_error_sum/(counter_pos + counter_neg)
    except:
        mean_act_overfit = overfit_error_sum/.0001 """

    reach_mae = np.mean(np.absolute(temp_arr))
    reach_std = np.std(np.absolute(temp_arr))
    
    reach_diff = np.mean(np.absolute(temp_ero_diff))
    reach_overfit = np.mean(np.absolute(overfit_error))
    non_ero_mae = 0.0
    writer.add_scalar(str(val_img_ids[iter_num])+'/mean_abs_pos_error_for_actual_'+side+'_erosion_'+str(val_img_ids[iter_num]), mean_pos_dev, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mean_abs_neg_error_for_actual_'+side+'_erosion_'+str(val_img_ids[iter_num]), mean_neg_dev, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mae_for_actual_'+side+'_erosion_'+str(val_img_ids[iter_num]), ero_mae, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mae_for_non_error_'+side+str(val_img_ids[iter_num]), non_ero_mae, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/reach_mae'+side+str(val_img_ids[iter_num]), reach_mae, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mean_reach_diff'+side+str(val_img_ids[iter_num]), reach_diff, epoch+1)
    #writer.add_scalar(str(val_img_ids[iter_num])+'/mean_act_ero_diff'+side+str(val_img_ids[iter_num]), mean_ero_diff, epoch+1)
    #writer.add_scalar(str(val_img_ids[iter_num])+'/mean_act_overfit_'+side+str(val_img_ids[iter_num]), mean_act_overfit, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/mean_reach_overfit_'+side+str(val_img_ids[iter_num]), reach_overfit, epoch+1)


    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_pos_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), pos_std, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_neg_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), neg_std, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_comb_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), comb_std, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_non_error_'+side+str(val_img_ids[iter_num]), non_std, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_reach_error'+side+str(val_img_ids[iter_num]), reach_std, epoch+1)

    mean_ero_diff = 0.0
    mean_act_overfit = 0.0

    return mean_pos_dev, pos_std, mean_neg_dev, neg_std, ero_mae, comb_std, non_ero_mae, non_std, reach_mae, reach_std, reach_diff, mean_ero_diff,mean_act_overfit,reach_overfit

def regress_erro_up(act_err_bin, act_reg, pred_reg, prev_reg, iter_num, side,writer, val_img_ids,epoch):
    #print(pred_reg)
    #print(act_reg)
    #print(asd)
    
    temp_arr = np.reshape(pred_reg - act_reg, (act_err_bin.shape[0],1))
    temp_ero_diff = act_reg - prev_reg
    overfit_error = pred_reg - prev_reg
    """ if side == 'left' :
        temp_arr = pred_reg - act_reg
    elif side == 'right' :
        temp_arr = -temp_arr """

    """ print(act_err_bin[0:20])
    print(act_err_bin.shape)
    print(temp_arr[0:20])
    print(pred_reg[0:20])
    print(temp_arr.dtype)
    print(temp_arr.shape) """

    erosion_arr = np.where(act_err_bin == 1, temp_arr, 0)
    erosion_arr = erosion_arr[erosion_arr != 0.0]
    erosion_arr = np.reshape(erosion_arr, (erosion_arr.shape[0],1))

    erosion_mae = np.mean(np.absolute(erosion_arr))
    erosion_std = np.std(np.absolute(erosion_arr))

    reach_mae = np.mean(np.absolute(temp_arr))
    reach_std = np.std(np.absolute(temp_arr))
    
    writer.add_scalar(str(val_img_ids[iter_num])+'/reach_mae'+side+str(val_img_ids[iter_num]), reach_mae, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_reach_error'+side+str(val_img_ids[iter_num]), reach_std, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/erosion_mae'+side+str(val_img_ids[iter_num]), erosion_mae, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/std_of_erosion_error'+side+str(val_img_ids[iter_num]), erosion_std, epoch+1)
    

    #return mean_pos_dev, pos_std, mean_neg_dev, neg_std, ero_mae, comb_std, non_ero_mae, non_std, reach_mae, reach_std, reach_diff, mean_ero_diff,mean_act_overfit,reach_overfit
    return reach_mae, reach_std, erosion_mae, erosion_std

def log_perform_lef_rght(log_item, left_comp, right_comp, writer, val_img_ids, iter_num, epoch):
    '''log the various mae and std in calc_fscore'''
    writer.add_scalar('left_'+log_item+'_for_actual_erosion'+str(val_img_ids[iter_num]), left_comp, epoch+1)
    writer.add_scalar('right_'+log_item+'_for_actual_erosion'+str(val_img_ids[iter_num]), right_comp, epoch+1)
    avg_comp = (left_comp + right_comp)/2
    writer.add_scalar('left_right_avg_'+log_item+'_for_actual_erosion'+str(val_img_ids[iter_num]), avg_comp, epoch+1)


def calc_fscore(iter_num, actual_list, prev_actual_list, pred_list, epoch,writer,val_img_ids,erosion_thresh):
    act_left = actual_list[:,iter_num,0]
    act_right = actual_list[:,iter_num,1]

    prev_left = prev_actual_list[:, iter_num, 0]
    prev_right = prev_actual_list[:, iter_num, 1]

    pred_left = pred_list[:, iter_num, 0]
    pred_right = pred_list[:, iter_num, 1]

    actual_ers_lft = np.reshape(np.where(prev_left - act_left >= erosion_thresh, 1, 0),(act_left.shape[0],1))
    actual_ers_rht = np.reshape(np.where(act_right - prev_right >= erosion_thresh, 1, 0),(act_right.shape[0],1))


    #left_mae_pos,left_std_pos,left_mae_neg,left_std_neg,lft_cm_m,lft_cm_std,lft_non_m,lft_non_s,lft_r_m,lft_r_s,lft_reach_diff,lft_ero_dif,lft_ove_act,lft_ove_rea = regress_erro(actual_ers_lft, act_left, pred_left, prev_left, iter_num, 'left',writer,val_img_ids,epoch)
    #right_mae_pos,right_std_pos,right_mae_neg,right_std_neg,rg_cm_m,rg_cm_std,rg_non_m,rg_non_s,rg_r_m,rg_r_s,rg_reach_diff,rg_ero_dif,rg_ove_act,rg_ove_rea = regress_erro(actual_ers_rht, act_right, pred_right, prev_right, iter_num, 'right',writer,val_img_ids,epoch)
    #print(asd)
    
    lft_r_m,lft_r_s,lft_er_m,lft_er_std = regress_erro_up(actual_ers_lft, act_left, pred_left, prev_left, iter_num, 'left',writer,val_img_ids,epoch)
    rg_r_m,rg_r_s,rg_er_m,rg_er_std = regress_erro_up(actual_ers_rht, act_right, pred_right, prev_right, iter_num, 'right',writer,val_img_ids,epoch)

    """ log_dic_lef_rght = {'pos_mae': [left_mae_pos,right_mae_pos], 'pos_std':[left_std_pos,right_std_pos],
    'neg_mae':[left_mae_neg,right_mae_neg],'neg_std':[left_std_neg,right_std_neg],
    'full_act_mae':[lft_cm_m,rg_cm_m],'full_act_std':[lft_cm_std,rg_cm_std],
    'full_non_erosion_mae':[lft_non_m,rg_non_m],'full_non_erosion_std':[lft_non_s,rg_non_s],
    'reach_mae':[lft_r_m,rg_r_m],'reach_std':[lft_r_s,rg_r_s],
    'reach_diff':[lft_reach_diff,rg_reach_diff],'mean_ero_diff':[lft_ero_dif,rg_ero_dif],
    'reach_overfit_diff':[lft_ove_rea,rg_ove_rea],'act_overfit_diff':[lft_ove_act,rg_ove_act]} """

    log_dic_lef_rght = {'reach_mae':[lft_r_m,rg_r_m],'reach_std':[lft_r_s,rg_r_s],
                        'erosion_mae':[lft_er_m,rg_er_m],'erosion_std':[lft_er_std,rg_er_std]}

    #for i,j in zip(log_dic_lef_rght.keys(),log_dic_lef_rght.values()):
    #    log_perform_lef_rght(i, j[0], j[1], writer, val_img_ids, iter_num, epoch)

    pred_ers_lft = np.reshape(np.where(prev_left - pred_left >= erosion_thresh, 1, 0),(pred_left.shape[0],1))
    pred_ers_rht = np.reshape(np.where(pred_right - prev_right >= erosion_thresh, 1, 0),(pred_right.shape[0],1))

    """ conf_mat_lft = confusion_matrix(actual_ers_lft, pred_ers_lft)
    conf_mat_rht = confusion_matrix(actual_ers_rht, pred_ers_rht)
    combined_conf = conf_mat_lft + conf_mat_rht """

    #act_th_ers_lft = np.reshape(np.where((prev_left-act_left)>3, 1, 0),(act_left.shape[0],1))
    #act_th_ers_rht = np.reshape(np.where((act_right-prev_right)>3, 1, 0),(act_right.shape[0],1))

    #pred_th_ers_lft = np.reshape(np.where((prev_left-pred_left)>3, 1, 0),(pred_left.shape[0],1))
    #red_th_ers_rht = np.reshape(np.where((pred_right-prev_right)>3, 1, 0),(pred_right.shape[0],1))

    """ prec_th_lft = precision_score(act_th_ers_lft, pred_th_ers_lft, average='binary')
    recall_th_lft = recall_score(act_th_ers_lft, pred_th_ers_lft, average='binary')
    #f1_th_lft = f1_score(act_th_ers_lft, pred_th_ers_lft, average='binary')

    prec_th_rht = precision_score(act_th_ers_rht, pred_th_ers_rht, average='binary')
    recall_th_rht = recall_score(act_th_ers_rht, pred_th_ers_rht, average='binary') """
    #f1_th_rht = f1_score(act_th_ers_rht, pred_th_ers_rht, average='binary')

    #writer.add_scalar(str(val_img_ids[iter_num])+'/left_precision_th_'+ str(val_img_ids[iter_num]), prec_th_lft, epoch+1)
    #writer.add_scalar(str(val_img_ids[iter_num])+'/left_recall_th_'+ str(val_img_ids[iter_num]), recall_th_lft, epoch+1)
    #writer.add_scalar(str(val_img_ids[iter_num])+'/left_f1_th_'+ str(val_img_ids[iter_num]), f1_th_lft, epoch+1)
    #writer.add_scalar(str(val_img_ids[iter_num])+'/right_precision_th_'+ str(val_img_ids[iter_num]), prec_th_rht, epoch+1)
    #writer.add_scalar(str(val_img_ids[iter_num])+'/right_recall_th'+ str(val_img_ids[iter_num]), recall_th_rht, epoch+1)
    #writer.add_scalar(str(val_img_ids[iter_num])+'/right_f1_th_'+ str(val_img_ids[iter_num]), f1_th_rht, epoch+1)

    y_true = np.concatenate((actual_ers_lft,actual_ers_rht), axis = 0)
    y_pred = np.concatenate((pred_ers_lft,pred_ers_rht), axis = 0)

    precision_comb = precision_score(y_true, y_pred, average='binary')
    recall_comb = recall_score(y_true, y_pred, average='binary')
    #f1_comb = f1_score(y_true, y_pred, average='binary')
    
    precision_lft = precision_score(actual_ers_lft, pred_ers_lft, average='binary')
    recall_lft = recall_score(actual_ers_lft, pred_ers_lft, average='binary')
    #f1_lft = f1_score(actual_ers_lft, pred_ers_lft, average='binary')

    precision_rht = precision_score(actual_ers_rht, pred_ers_rht, average='binary')
    recall_rht = recall_score(actual_ers_rht, pred_ers_rht, average='binary')
    #f1_rht = f1_score(actual_ers_rht, pred_ers_rht, average='binary')

    
    #writer.add_scalar(str(val_img_ids[iter_num])+'/precision_'+ str(val_img_ids[iter_num]), precision_comb, epoch+1)
    #writer.add_scalar(str(val_img_ids[iter_num])+'/recall_'+ str(val_img_ids[iter_num]), recall_comb, epoch+1)
    #writer.add_scalar(str(val_img_ids[iter_num])+'/f1_score_'+ str(val_img_ids[iter_num]), f1_comb, epoch+1)

    writer.add_scalar(str(val_img_ids[iter_num])+'/left_precision_'+ str(val_img_ids[iter_num]), precision_comb, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/left_recall_'+ str(val_img_ids[iter_num]), recall_comb, epoch+1)
    #writer.add_scalar(str(val_img_ids[iter_num])+'/left_f1_score_'+ str(val_img_ids[iter_num]), f1_comb, epoch+1)

    writer.add_scalar(str(val_img_ids[iter_num])+'/right_precision_'+ str(val_img_ids[iter_num]), precision_comb, epoch+1)
    writer.add_scalar(str(val_img_ids[iter_num])+'/right_recall_'+ str(val_img_ids[iter_num]), recall_comb, epoch+1)
    #writer.add_scalar(str(val_img_ids[iter_num])+'/right_f1_score_'+ str(val_img_ids[iter_num]), f1_comb, epoch+1)

    avg_reach_mae = (lft_r_m + rg_r_m) / 2
    avg_erosion_mae = (lft_er_m + rg_er_m) / 2 
    #left_overfit_metric = abs(lft_reach_diff - lft_ove_rea) 
    #right_overfit_metric = abs(rg_reach_diff - rg_ove_rea)
    #avg_overfit_metric = (left_overfit_metric + right_overfit_metric) / 2
    #augmented_metric = avg_reach_mae + avg_overfit_metric
    #lr_f1_score = f1_comb
    #lft_aug_metric = left_overfit_metric + lft_r_m
    #rgt_aug_metric = right_overfit_metric + rg_r_m
    

    #writer.add_scalar('AM_score/'+str(val_img_ids[iter_num])+'_augmented_metric_', augmented_metric, epoch+1)
    #writer.add_scalar('AM_score/'+str(val_img_ids[iter_num])+'_left_augmented_metric_', lft_aug_metric, epoch+1)
    #writer.add_scalar('AM_score/'+str(val_img_ids[iter_num])+'_right_augmented_metric_', rgt_aug_metric, epoch+1)
    #writer.add_scalar('F1_score/'+str(val_img_ids[iter_num])+'_f1_score_', lr_f1_score, epoch+1)
    writer.add_scalar('Reach_MAE/'+str(val_img_ids[iter_num])+'_lr_reach_mae_', avg_reach_mae, epoch+1)
    writer.add_scalar('Erosion_MAE/'+str(val_img_ids[iter_num])+'_lr_erosion_mae_', avg_erosion_mae, epoch+1)

    #log_dic_lef_rght.update()
    #lr_f1_score = 0.0
    """ log_dic_scores = {'AM_score':augmented_metric, 'lr_f1_score':lr_f1_score, 'lr_reach_mae':avg_reach_mae,
    'left_precision':precision_lft,'left_recall':recall_lft,'left_f1':f1_lft,
    'right_precision':precision_rht,'right_recall':recall_rht,'right_f1':f1_rht,
    'lft_rht_precision':precision_comb,'lft_rgt_recall':recall_comb,'lft_rht_f1':f1_comb,
    'left_prec_th':prec_th_lft,'left_recall_th':recall_th_lft,'left_f1_th':f1_th_lft,
    'right_prec_th':prec_th_rht,'right_recall_th':recall_th_rht,'right_f1_th':f1_th_rht} """

    """ log_dic_scores = {'AM_score':augmented_metric, 'Left_AM_Score': lft_aug_metric,'Right_AM_Score':rgt_aug_metric,'lr_reach_mae':avg_reach_mae,
    'left_precision':precision_lft,'left_recall':recall_lft,
    'right_precision':precision_rht,'right_recall':recall_rht,
    'lft_rgt_precision':precision_comb,'lft_rgt_recall':recall_comb,
    'left_prec_th':prec_th_lft,'left_recall_th':recall_th_lft,
    'right_prec_th':prec_th_rht,'right_recall_th':recall_th_rht} """

    log_dic_scores = {'lr_reach_mae':avg_reach_mae, 'lr_erosion_mae':avg_erosion_mae,
    'left_precision':precision_lft,'left_recall':recall_lft,
    'right_precision':precision_rht,'right_recall':recall_rht
    }

    """ imp_val_log = {str(val_img_ids[iter_num])+'_augmented_metric':augmented_metric, str(val_img_ids[iter_num])+'_lr_f1_score':lr_f1_score,
                str(val_img_ids[iter_num])+'_lr_reach_mae':avg_reach_mae,
                str(val_img_ids[iter_num])+'_left_overfit_metric':left_overfit_metric,str(val_img_ids[iter_num])+'_right_overfit_metric':right_overfit_metric,
                str(val_img_ids[iter_num])+'_left_AM_metric':left_overfit_metric+lft_r_m,str(val_img_ids[iter_num])+'_right_AM_metric':right_overfit_metric+rg_r_m,
                str(val_img_ids[iter_num])+'_left_reach_mae':lft_r_m, str(val_img_ids[iter_num])+'_right_reach_mae':rg_r_m,
                str(val_img_ids[iter_num])+'_left_f1score':f1_lft, str(val_img_ids[iter_num])+'_right_f1score':f1_rht,
                str(val_img_ids[iter_num])+'_left_erosion_diff':lft_ero_dif, str(val_img_ids[iter_num])+'_right_erosion_diff':rg_ero_dif,
                str(val_img_ids[iter_num])+'_left_act_erosion_mae':lft_cm_m, str(val_img_ids[iter_num])+'_right_act_erosion_mae':rg_cm_m,
                str(val_img_ids[iter_num])+'_left_f1_th':f1_th_lft, str(val_img_ids[iter_num])+'_right_f1_th':f1_th_rht,
                str(val_img_ids[iter_num])+'_left_act_overfit_diff':lft_ove_act,str(val_img_ids[iter_num])+'_right_act_overfit_diff':rg_ove_act} """

    """ imp_val_log = {str(val_img_ids[iter_num])+'_augmented_metric':augmented_metric,
    str(val_img_ids[iter_num])+'_lr_reach_mae':avg_reach_mae,
    str(val_img_ids[iter_num])+'_left_overfit_metric':left_overfit_metric,str(val_img_ids[iter_num])+'_right_overfit_metric':right_overfit_metric,
    str(val_img_ids[iter_num])+'_left_AM_metric':left_overfit_metric+lft_r_m,str(val_img_ids[iter_num])+'_right_AM_metric':right_overfit_metric+rg_r_m,
    str(val_img_ids[iter_num])+'_left_reach_mae':lft_r_m, str(val_img_ids[iter_num])+'_right_reach_mae':rg_r_m,
    str(val_img_ids[iter_num])+'_left_erosion_diff':lft_ero_dif, str(val_img_ids[iter_num])+'_right_erosion_diff':rg_ero_dif,
    str(val_img_ids[iter_num])+'_left_act_erosion_mae':lft_cm_m, str(val_img_ids[iter_num])+'_right_act_erosion_mae':rg_cm_m,
    str(val_img_ids[iter_num])+'_left_act_overfit_diff':lft_ove_act,str(val_img_ids[iter_num])+'_right_act_overfit_diff':rg_ove_act} """
    #return avg_mae_pos, avg_std_pos, avg_mae_neg, avg_std_neg, precision_comb, recall_comb, f1_comb

    imp_val_log = {
    str(val_img_ids[iter_num])+'_lr_reach_mae':avg_reach_mae, str(val_img_ids[iter_num])+'_lr_eroison_mae':avg_erosion_mae,
    str(val_img_ids[iter_num])+'_left_reach_mae':lft_r_m, str(val_img_ids[iter_num])+'_right_reach_mae':rg_r_m,
    str(val_img_ids[iter_num])+'_left_erosion_mae':lft_er_m, str(val_img_ids[iter_num])+'_right_erosion_mae':rg_er_m,
    }

    return log_dic_lef_rght, log_dic_scores, imp_val_log



def wrt_img(iter_num, actual_list, prev_actual_list, pred_list, val_img_ids,writer,smooth_flag,
            reach_start_indx,out_use_mid,vert_img_hgt):
    
    if out_use_mid == True :
        reach_start_indx = reach_start_indx + int((vert_img_hgt-1)/2)


    #print(actual_list[0:20,:,:])
    #print(actual_list)
    #print(asd)

    num_rows = int(pred_list.shape[0])
    #print(num_rows)

    actual_ers_lft = np.reshape(np.where(actual_list[:,iter_num,0]<prev_actual_list[:,iter_num,0], 1, 0),(actual_list[:,iter_num,0].shape[0],1))
    actual_ers_rht = np.reshape(np.where(actual_list[:,iter_num,1]>prev_actual_list[:,iter_num,1], 1, 0),(actual_list[:,iter_num,1].shape[0],1))

    pred_ers_lft = np.reshape(np.where(pred_list[:,iter_num,0]<prev_actual_list[:,iter_num,0], 1, 0),(pred_list[:,iter_num,0].shape[0],1))
    pred_ers_rht = np.reshape(np.where(pred_list[:,iter_num,1]>prev_actual_list[:,iter_num,1], 1, 0),(pred_list[:,iter_num,1].shape[0],1))

    conf_mat_lft = confusion_matrix(actual_ers_lft, pred_ers_lft)
    conf_mat_rht = confusion_matrix(actual_ers_rht, pred_ers_rht)
    combined_conf = conf_mat_lft + conf_mat_rht

    #plt_conf_mat(conf_mat_lft, str(val_img_ids[iter_num])+'_conf_mat_left', writer)
    #plt_conf_mat(conf_mat_rht, str(val_img_ids[iter_num])+'_conf_mat_right', writer)
    #plt_conf_mat(combined_conf, str(val_img_ids[iter_num])+'_combined_conf_mat', writer)

    denoising = smooth_flag
    window = 99
    poly = 2

    """ if denoising :
        try :
            pred_left_den = savgol_filter(pred_list[:,iter_num,0], window, poly)
            pred_right_den = savgol_filter(pred_list[:,iter_num,1], window, poly)
        except :
            pred_left_den = pred_list[:,iter_num,0]
            pred_right_den = pred_list[:,iter_num,1] """

    if str(val_img_ids[iter_num]) == '201901' :
        img = cv2.imread(os.path.join('./data/cegis_refined.png'), 1)
    else :
        img = cv2.imread(os.path.join('./data/img/up_rgb/'+str(val_img_ids[iter_num])+'.png'), 1)
    coun = 0
    for i,j in zip(range(reach_start_indx,reach_start_indx+num_rows,1),range(num_rows)):
        #print(coun)
        """ print(i,j)
        print(len(range(reach_start_indx,reach_start_indx+num_rows,1)))
        
        print(len((range(num_rows))))
        assert len(range(reach_start_indx,reach_start_indx+num_rows-1,1)) == len(range(num_rows)) 
        print(img.shape)"""
        #print(actual_list)
        #print(asd)
        #print(actual_list)
        #print(actual_list[j,iter_num,1])
        #print(asd)
        """ if coun == 475 :
            print(actual_list[j,iter_num,0])
            print(actual_list[j,iter_num,1]) """
        img[i,int(round(actual_list[j,iter_num,0])),:] = [255,255,255]
        img[i,int(round(actual_list[j,iter_num,1])),:] = [255,255,255]

        if 0<=int(round(pred_list[j,iter_num,0]))<=744 :
            pass
        else :
            pred_list[j,iter_num,0] = 0

        if 0<=int(round(pred_list[j,iter_num,1]))<=744 :
            pass
        else :
            pred_list[j,iter_num,1] = 744

        """ if denoising :
            if 0<=int(pred_left_den[i])<=744 :
                pass
            else :
                pred_left_den[j] = 0 

            if 0<=int(pred_right_den[i])<=744 : 
                pass
            else :
                pred_right_den[i] = 744 """


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

        img[i,int(round(prev_actual_list[j,iter_num,0])),:] = [255,0,0]
        img[i,int(round(prev_actual_list[j,iter_num,1])),:] = [255,0,0]
        img[i,int(round(pred_list[j,iter_num,0])),:] = [0,255,0]
        img[i,int(round(pred_list[j,iter_num,1])),:] = [0,255,0]
        """ if denoising:
            img[i,int(pred_left_den[i]),:] = [0,0,255]
            img[i,int(pred_right_den[i]),:] = [0,0,255] """
        coun+=1
            
    writer.add_image(str(val_img_ids[iter_num]), img, dataformats='HWC')
    return combined_conf

def process_prev(arr_list, num_val_img,prev_year_ids,prev_reach_ids,vert_img_hgt,vert_step,inp_mode,
                flag_standardize_actual,transform_constants,extra_samples):

    arr_list = np.asarray(arr_list, dtype=np.int32)
    #print(arr_list.shape)
    #print(asd)
    """ total_rows = int(arr_list.shape[1] * arr_list.shape[2])
    arr_list = np.transpose(arr_list,[1,0,2,3])
    
    arr_list = np.reshape(arr_list, (num_val_img,total_rows,2)) """
    total_rows = int(arr_list.shape[0] * arr_list.shape[1])
    arr_list = np.reshape(arr_list, (1,total_rows,2))

    arr_list = np.transpose(arr_list,[1,0,2])

    """ prev_year_ids = np.asarray(prev_year_ids)
    prev_year_ids = np.transpose(prev_year_ids,[1,0,2,3])
    prev_year_ids = np.reshape(prev_year_ids, (num_val_img,total_rows,1))
    prev_year_ids = np.transpose(prev_year_ids,[1,0,2])

    prev_reach_ids = np.asarray(prev_reach_ids)
    prev_reach_ids = np.transpose(prev_reach_ids,[1,0,2,3])
    prev_reach_ids = np.reshape(prev_reach_ids, (num_val_img,total_rows,1))
    prev_reach_ids = np.transpose(prev_reach_ids,[1,0,2]) """
    
    if extra_samples != 0 :
        arr_list = arr_list[:-extra_samples,:,:]

    return arr_list

def process_diffs(arr_list, num_val_img, prev_actual_list,act_year_ids,act_reach_ids,
                    vert_img_hgt,out_mode,flag_standardize_actual,transform_constants,output_subtracted,extra_samples):

    arr_list = np.asarray(arr_list, dtype=np.int32)
    """ total_rows = int(arr_list.shape[1] * arr_list.shape[2])
    #print(arr_list.shape)
    #print(asd)

    arr_list = np.transpose(arr_list,[1,0,2,3])
    
    arr_list = np.reshape(arr_list, (num_val_img,total_rows,2)) """

    total_rows = int(arr_list.shape[0] * arr_list.shape[1])
    arr_list = np.reshape(arr_list, (1,total_rows,2))

    arr_list = np.transpose(arr_list,[1,0,2])

    """ act_year_ids = np.asarray(act_year_ids)
    act_year_ids = np.transpose(act_year_ids,[1,0,2,3])
    act_year_ids = np.reshape(act_year_ids, (num_val_img,total_rows,1))
    act_year_ids = np.transpose(act_year_ids,[1,0,2])

    act_reach_ids = np.asarray(act_reach_ids)
    act_reach_ids = np.transpose(act_reach_ids,[1,0,2,3])
    act_reach_ids = np.reshape(act_reach_ids, (num_val_img,total_rows,1))
    act_reach_ids = np.transpose(act_reach_ids,[1,0,2]) """
    if extra_samples != 0 :
        arr_list = arr_list[:-extra_samples,:,:]


    """ if output_subtracted == True :
        arr_list = np.add(arr_list,prev_actual_list)
    elif out_mode == 'act':
        pass """

    #print(arr_list.shape)
    #print(asd)
    return arr_list

def process_diffs_pred(arr_list, num_val_img, prev_actual_list,act_year_ids,act_reach_ids,
                    vert_img_hgt,out_mode,flag_standardize_actual,transform_constants,output_subtracted,extra_samples):

    arr_list = np.asarray(arr_list, dtype=np.float32)
    """ total_rows = int(arr_list.shape[1] * arr_list.shape[2])
    #print(arr_list.shape)
    #print(asd)

    arr_list = np.transpose(arr_list,[1,0,2,3])
    
    arr_list = np.reshape(arr_list, (num_val_img,total_rows,2)) """

    total_rows = int(arr_list.shape[0] * arr_list.shape[1])
    arr_list = np.reshape(arr_list, (1,total_rows,2))

    arr_list = np.transpose(arr_list,[1,0,2])

    """ act_year_ids = np.asarray(act_year_ids)
    act_year_ids = np.transpose(act_year_ids,[1,0,2,3])
    act_year_ids = np.reshape(act_year_ids, (num_val_img,total_rows,1))
    act_year_ids = np.transpose(act_year_ids,[1,0,2])

    act_reach_ids = np.asarray(act_reach_ids)
    act_reach_ids = np.transpose(act_reach_ids,[1,0,2,3])
    act_reach_ids = np.reshape(act_reach_ids, (num_val_img,total_rows,1))
    act_reach_ids = np.transpose(act_reach_ids,[1,0,2]) """
    if extra_samples != 0 :
        arr_list = arr_list[:-extra_samples,:,:]


    """ if output_subtracted == True :
        arr_list = np.add(arr_list,prev_actual_list)
    elif out_mode == 'act':
        pass """

    #print(arr_list.shape)
    #print(asd)
    return arr_list



    

def model_save(model, optimizer, model_name):
    print('saving model....')
    model_path = os.path.join('./data/model/'+model_name +'.pt')
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, model_path)

def log_performance_metrics(pred_list,actual_list,prev_actual_list,num_val_img, epoch, 
                val_img_ids,writer,erosion_thresh):
    print('logging performance metrics........')
    imp_val_logs = {}
    for i in range(num_val_img):
        log_dic_lef_rght, log_dic_scores, imp_val_log_ind = calc_fscore(i, actual_list, prev_actual_list, pred_list,
                                             epoch,writer,val_img_ids,erosion_thresh)
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


def write_inp_tf(input_tensor,reach_id,year_id,batch_size,time_step,inp_std,inp_mean):
    

    temp_inp = input_tensor.copy()
    temp_rch = reach_id.copy()
    temp_yd = year_id.copy()
    temp_ims = os.path.join('./data/img/temp/')

    for m in range(batch_size):
        for n in range(time_step-1):
            input_tensor = temp_inp[m,n,:,:].copy()
            reach_id = temp_rch[m,n,:,:].copy()
            year_id = temp_yd[m,n,:,:].copy()

            input_tensor = np.add(np.multiply(input_tensor,np.transpose(inp_std)[int(reach_id)]), np.transpose(inp_mean)[int(reach_id)])
            print(round(input_tensor[0,0]),round(input_tensor[0,1]))
            img = cv2.imread(temp_ims+str(int(year_id))+'.png')
            img[int(reach_id),round(input_tensor[0,0]),:] = [255,255,255]
            img[int(reach_id),round(input_tensor[0,1]),:] = [255,255,255]
            cv2.imwrite(temp_ims+str(int(year_id))+'.png', img)


def process_tf_dataset(input_tensor_org, year_id, reach_id, sdd_output,inp_mode,inp_lr_flag,out_mode,out_lr_tag,
    flag_standardize_actual,flag_reach_use,batch_size,time_step,vert_img_hgt,flag_sdd_act_data,output_subtracted,
    out_use_mid,concat_img):

    #year_id = np.reshape(year_id, (batch_size,time_step,vert_img_hgt,1))
    reach_id = np.reshape(reach_id, (batch_size,time_step,vert_img_hgt,1))

    
    #input_type = 'img' 
    if inp_lr_flag == 'img' :
        input_tensor = np.reshape(concat_img, (batch_size,time_step,vert_img_hgt,745,7))
    else :
        input_tensor = np.reshape(input_tensor_org, (batch_size,time_step,vert_img_hgt,2))

    

    if inp_lr_flag == 'left' :
        input_tensor = input_tensor[:,0:time_step-1,:,0:1]
        input_tensor_sub = np.reshape(input_tensor[:,-1,:,:],(batch_size,1,vert_img_hgt,1))

        if flag_reach_use :
            reach_id = reach_id[:,0,:,:]

    elif inp_lr_flag == 'right' :
        input_tensor = input_tensor[:,0:time_step-1,:,1:2]
        input_tensor_sub = np.reshape(input_tensor[:,-1,:,:],(batch_size,1,vert_img_hgt,1))

        if flag_reach_use :
            #print(reach_id)
            reach_id = reach_id[:,0,:,:]
            #print(reach_id)
            #print(asd)

    elif inp_lr_flag == 'both' :
        input_tensor = input_tensor[:,0:time_step-1,:,:]
        input_tensor_sub = np.reshape(input_tensor[:,-1,:,:],(batch_size,1,vert_img_hgt,2))

        if flag_reach_use :
            reach_id = reach_id[:,0,:,:]

    elif (inp_lr_flag == 'img') and (output_subtracted == True) and (out_lr_tag == 'both'):
        input_tensor = input_tensor[:,0:time_step-1,:,:,:]

        input_tensor_sub = np.reshape(input_tensor_org, (batch_size,time_step,vert_img_hgt,2))
        input_tensor_sub = input_tensor_sub[:,0:time_step-1,:,:]
        input_tensor_sub = np.reshape(input_tensor_sub[:,-1,:,:],(batch_size,1,vert_img_hgt,2))




    if out_mode == 'diff_sdd' :
        sdd_output = np.reshape(sdd_output, (batch_size,time_step,vert_img_hgt,2))
    elif out_mode == 'act_sdd' or out_mode == 'act':
        sdd_output = np.reshape(input_tensor_org, (batch_size,time_step,vert_img_hgt,2))
    

    if out_lr_tag == 'left':
        sdd_output = sdd_output[:,time_step-1:time_step,:,0:1]
    elif out_lr_tag == 'right':
        sdd_output = sdd_output[:,time_step-1:time_step,:,1:2]
    elif out_lr_tag == 'both':
        sdd_output = sdd_output[:,time_step-1:time_step,:,:]

    output_subtracted = output_subtracted
    if output_subtracted == True :
        sdd_output = np.subtract(sdd_output,input_tensor_sub)

    if (inp_mode == 'act') and (flag_standardize_actual == True) :
        if flag_sdd_act_data == True :
            
            if inp_lr_flag != 'img' :
                inp_flatten = np.reshape(input_tensor, (batch_size, -1))
            out_flatten = np.reshape(sdd_output, (batch_size, -1))

            #out_use_mid = True
            if out_use_mid == True :
                out_flatten = out_flatten[:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1]

            if inp_lr_flag != 'img' :
                if flag_reach_use == True :
                    reach_id = np.reshape(reach_id, (batch_size, -1))
                    if vert_img_hgt > 1 :
                        reach_id = reach_id[:,0]
                        reach_id = np.reshape(reach_id, (batch_size, -1))
                        inp_flatten = np.concatenate((inp_flatten,reach_id),axis=1)
                    elif vert_img_hgt == 1 :
                        inp_flatten = np.concatenate((inp_flatten,reach_id),axis=1)
    
    #print(inp_flatten[0:20,:])
    #print(out_flatten[0:20,:])
    #print(asd)

    if inp_lr_flag == 'img' :
        inp_flatten = input_tensor

    return inp_flatten, out_flatten, input_tensor, sdd_output


""" def process_py_imgs(concat_img,lines):

    concat_img = concat_img[:,:-1,:,:]
    lines = lines[:,-1:,:,:] """





def custom_mean_sdd(dataset_f,inp_mode,inp_lr_flag,out_mode,out_lr_tag,flag_standardize_actual,
                    flag_reach_use,batch_size,time_step,vert_img_hgt,flag_sdd_act_data,output_subtracted,
                    out_use_mid,dataset_type,dataset_dic,flag_use_lines):
    print('calculating standardization constants .........')
    inp_list = []
    out_list = []

    prox_counter = 0

    #dataset_type = 'pydic'

    if dataset_type == 'tfrecord':
        for input_tensor_org, year_id, reach_id, _, sdd_output, concat_img in dataset_f:

            inp_flatten, out_flatten, inp_tensor, out_tensor = process_tf_dataset(input_tensor_org, year_id, reach_id, sdd_output,inp_mode,inp_lr_flag,
                out_mode,out_lr_tag,flag_standardize_actual,flag_reach_use,batch_size,time_step,vert_img_hgt,flag_sdd_act_data,output_subtracted,out_use_mid,concat_img)

            if prox_counter == 0 :
            
                prin_count = 1
                """ print('batch,time_step,vert_hgt,banks format input/output view ........')
                print(inp_tensor[0:prin_count,:,:,:])
                print(out_tensor[0:prin_count,:,:,:]) """
                print('flatten input/output view ............')
                if inp_lr_flag != 'img' :
                    print(inp_flatten[0:prin_count,:])
                elif inp_lr_flag == 'img' :
                    print(inp_tensor[0,:,:,:,:])
                    print(inp_tensor.shape)

                print(out_flatten[0:prin_count,:])


            if inp_lr_flag != 'img' :
                inp_list.append(inp_flatten)
            out_list.append(out_flatten)

            prox_counter += 1

        
        if inp_lr_flag != 'img' :
            inp_list = np.asarray(inp_list,dtype=np.float32)
            inp_list = np.reshape(inp_list,(batch_size*prox_counter, -1))        

        out_list = np.asarray(out_list,dtype=np.float32)
        out_list = np.reshape(out_list,(batch_size*prox_counter, -1))
        
        #np.savetxt(os.path.join('./data/right_bank_inp.csv'), inp_list, delimiter=",")
        #np.savetxt(os.path.join('./data/right_bank_out.csv'), out_list, delimiter=",")
        #print(inp_list[0:4,:])
        #print(out_list[0:4,:])
        #print(inp_list.shape)
        #print(out_list.shape)
        #print(asd)

        if inp_lr_flag != 'img' :
            inp_list_mean = np.mean(inp_list,axis=0)
            inp_list_std = np.std(inp_list,axis=0)
        elif inp_lr_flag == 'img' :
            inp_list_mean = None
            inp_list_std = None

        out_list_mean = np.mean(out_list,axis=0)
        out_list_std = np.std(out_list,axis=0)

        transform_constants = {'inp_mean':inp_list_mean,'inp_std':inp_list_std,
                            'out_mean':out_list_mean,'out_std':out_list_std}

        
    elif dataset_type == 'pydic' :
        #flag_use_reachid = True
        num_batches = len(dataset_f)
        for i_batch, sample_batched in enumerate(dataset_f) :

            lines = sample_batched['lines']
            

            if (i_batch+1) == num_batches :
                last_batch_size = lines.shape[0]
            
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
    
def train_performance(dataset_tr_pr,inp_mode,inp_lr_flag,out_mode,out_lr_tag,flag_standardize_actual,flag_reach_use,
        batch_size,time_step,vert_img_hgt,flag_sdd_act_data,inp_mean,inp_std,out_mean,out_std,model,
        loss_func,transform_constants,num_val_img,output_subtracted,out_use_mid):
    
    inp_mean = transform_constants['inp_mean']
    inp_std = transform_constants['inp_std']
    out_mean = transform_constants['out_mean']
    out_std = transform_constants['out_std']

    pred_list = []
    actual_list = []
    
    model.eval()
    with torch.no_grad():
        for inp_flatten_org, year_id, reach_id, _, sdd_output, concat_img in dataset_tr_pr:

            inp_flatten_org, out_flatten,_,_ = process_tf_dataset(inp_flatten_org, year_id, reach_id, sdd_output,inp_mode,inp_lr_flag,
            out_mode,out_lr_tag,flag_standardize_actual,flag_reach_use,batch_size,time_step,vert_img_hgt,flag_sdd_act_data,output_subtracted,out_use_mid,concat_img)


            if (inp_mode == 'act') and (flag_standardize_actual == True) :
                inp_flatten = (inp_flatten_org - inp_mean) / inp_std 
                #out_flatten = (out_flatten - out_mean) / out_std 
            
            inp_flatten = torch.Tensor(inp_flatten).cuda()                
            #out_flatten = torch.Tensor(out_flatten).cuda()
            pred = model(inp_flatten)

                
            if flag_reach_use == True :
                if vert_img_hgt > 1 :
                    inp_flatten_org = inp_flatten_org[:,:-1]
                elif vert_img_hgt == 1 :
                    inp_flatten_org = inp_flatten_org[:,:-vert_img_hgt]
                if inp_lr_flag == 'left' or inp_lr_flag == 'right' :
                    inp_flatten_org = np.reshape(inp_flatten_org,(batch_size,time_step-1,vert_img_hgt,1))
                elif inp_lr_flag == 'both' :
                    inp_flatten_org = np.reshape(inp_flatten_org,(batch_size,time_step-1,vert_img_hgt,2))
                
                inp_flatten_org = inp_flatten_org[:,-1,:,:]
                prev_inp_flatten = np.reshape(inp_flatten_org,(batch_size,-1))
            elif flag_reach_use == False :
                prev_inp_flatten = inp_flatten_org[:,-1]

            pred_np = pred.cpu()
            pred_np = pred_np.numpy()            

            if output_subtracted == True :
                out_flatten = np.add(out_flatten,prev_inp_flatten)
            pred_np = np.add(np.multiply(pred_np,out_std),out_mean)
            
            if output_subtracted == True :
                pred_np = np.add(pred_np,prev_inp_flatten)

            #prev_actual_list.append(prev_inp_flatten)
            actual_list.append(out_flatten)
            pred_list.append(pred_np)

        actual_list = np.asarray(actual_list)
        pred_list = np.asarray(pred_list)
        actual_list = np.reshape(actual_list,(actual_list.shape[0]*actual_list.shape[1],-1))
        pred_list = np.reshape(pred_list,(pred_list.shape[0]*pred_list.shape[1],-1))

        train_mae = np.mean(np.absolute(actual_list-pred_list))
        
        return train_mae

def pt_train_per(dataset_tr_pr,inp_mode,inp_lr_flag,out_mode,out_lr_tag,flag_standardize_actual,flag_reach_use,
        batch_size,time_step,vert_img_hgt,flag_sdd_act_data,inp_mean,inp_std,out_mean,out_std,model,
        loss_func,transform_constants,num_val_img,output_subtracted,out_use_mid,flag_use_lines,flag_use_imgs):
    
    out_mean = transform_constants['out_mean']
    out_std = transform_constants['out_std']
    inp_lines_mean = transform_constants['inp_lines_mean']
    inp_lines_std = transform_constants['inp_lines_std']
    inp_reach_mean = transform_constants['inp_reach_mean']
    inp_reach_std = transform_constants['inp_reach_std']


    #pred_list = []
    #actual_list = []
    
    model.eval()
    #counter = 0
    with torch.no_grad():

        num_batches = len(dataset_tr_pr)
        for i_batch, sample_batched in enumerate(dataset_tr_pr) :

            if flag_use_imgs :
                inp_flatten = sample_batched['img']
                inp_flatten = inp_flatten[:,:-1,:,:,:]
                inp_flatten = inp_flatten / 255.0
                inp_flatten = np.asarray(inp_flatten, dtype=np.float32)
                inp_flatten = torch.Tensor(inp_flatten).cuda()   
            else :
                inp_flatten = None             

            out_flatten_org = sample_batched['lines']

            if (i_batch+1) == num_batches :
                last_batch_size = out_flatten_org.shape[0]

            
            lines_last = out_flatten_org[:,-1:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]
            lines_prev = out_flatten_org[:,-2:-1,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:]

            lines_last = torch.reshape(lines_last, (lines_last.shape[0], 2))
            lines_prev = torch.reshape(lines_prev, (lines_prev.shape[0], 2))

            #out_flatten = lines_last - lines_prev

            #out_flatten = (out_flatten - out_mean) / out_std 
            
            #print(inp_flatten.shape)
            #print(asd)
            #out_flatten = np.asarray(out_flatten, dtype=np.float32)
            #out_flatten = torch.Tensor(out_flatten).cuda()

            if flag_use_lines :
                lines_prev_inp = out_flatten_org[:,:-1,:,:]
                lines_prev_inp = torch.reshape(lines_prev_inp, (lines_prev_inp.size()[0],-1))
                lines_prev_inp = (lines_prev_inp - inp_lines_mean) / inp_lines_std
                lines_prev_inp = lines_prev_inp.float().cuda().requires_grad_(False)

                if flag_reach_use :
                    reach_id = sample_batched['reaches']
                    reach_id = reach_id[:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1]                        
                    reach_id = torch.reshape(reach_id, (reach_id.shape[0], 1))
                    reach_id = (reach_id - inp_reach_mean) / inp_reach_std
                    reach_id = reach_id.float().cuda().requires_grad_(False)
            else :
                lines_prev_inp = None
                reach_id = None
            
            #if flag_use_lines :
            _, pred_left, pred_right, _, _ = model(inp_flatten, lines_prev_inp, reach_id)
            #else :
            #    pred, _,_,_,_ = model(inp_flatten, lines_prev_inp, reach_id)

            #if flag_use_lines :
            pred_left = pred_left.cpu()
            pred_left = pred_left.numpy()
            pred_right = pred_right.cpu()
            pred_right = pred_right.numpy()

            pred = np.concatenate((pred_left,pred_right), axis=1)
            """ else :
                pred = pred.cpu()
                pred = pred.numpy() """
            #if output_subtracted == True :
            #    out_flatten = np.add(out_flatten,prev_inp_flatten)
            pred = np.add(np.multiply(pred,out_std),out_mean)
            
            if output_subtracted == True :
                pred = np.add(pred,lines_prev)

            lines_last = np.asarray(lines_last)
            pred = np.asarray(pred)

            #print(lines_last)
            #print(pred_np)
            
            abs_batch_mae = np.absolute(lines_last-pred)
            abs_batch_mae = np.sum(abs_batch_mae, axis=0)

            if i_batch == 0 :
                np_aggr = np.zeros(abs_batch_mae.shape)

            np_aggr = np_aggr + abs_batch_mae

    train_mae = np_aggr / ((i_batch * batch_size) + last_batch_size)


    return train_mae


def create_train_dataset(data_div_step,total_time_step,start_indx,reach_win_size,reach_shift_cons,reach_start_indx,
            vert_img_hgt,vert_step,end_indx,time_step,time_win_shift,train_shuffle,batch_size):

    #dataset_f = tf.data.TFRecordDataset(os.path.join('./data/tfrecord/lines_'+str(0)+'_'+str(5)+'.tfrecords'))
    dataset_f = tf.data.TFRecordDataset(os.path.join('./data/tfrecord/img_lines_comp.tfrecords'))


    dataset_f = dataset_f.window(size=data_div_step, shift=total_time_step, stride=1, drop_remainder=False)
    dataset_f = dataset_f.map(lambda x: x.skip(start_indx))

    dataset_f = dataset_f.window(size=reach_win_size, shift=reach_shift_cons, stride=1, drop_remainder=False)
    dataset_f = dataset_f.map(lambda x: x.skip(reach_start_indx))
    dataset_f = dataset_f.flat_map(lambda x: x)

    dataset_f = dataset_f.window(size=vert_img_hgt, shift=vert_step, stride=vert_step,drop_remainder=True)
    dataset_f = dataset_f.map(lambda x: x.flat_map(lambda x1: x1))
    dataset_f = dataset_f.map(lambda x: x.window(size=vert_img_hgt,shift=1,stride=end_indx-start_indx+1,drop_remainder=True))
    dataset_f = dataset_f.map(lambda x: x.window(size=time_step, shift=time_win_shift, stride=1,drop_remainder=True))
    dataset_f = dataset_f.flat_map(lambda x: x)
    dataset_f = dataset_f.flat_map(lambda x: x.flat_map(lambda x1: x1))

    dataset_f = dataset_f.map(_parse_function_org_,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(vert_img_hgt).batch(time_step)
    
    if train_shuffle == True :
        #print(dataset_f.cardinality().numpy())
        #rint(asd)
        dataset_f = dataset_f.cache()
        dataset_f = dataset_f.shuffle(10000, reshuffle_each_iteration=True)
    elif train_shuffle == False :
        pass
    dataset_f = dataset_f.batch(batch_size, drop_remainder=True)
    dataset_f = dataset_f.prefetch(int(10*batch_size))

    return dataset_f

def create_val_dataset(total_time_step,val_split,val_skip,reach_win_size,reach_shift_cons,reach_start_indx,vert_img_hgt,vert_step,
        val_img_range,time_step,val_batch_size):

    #dataseti1 = tf.data.TFRecordDataset(os.path.join('./data/tfrecord/lines_'+str(0)+'_'+str(5)+'.tfrecords'))
    dataseti1 = tf.data.TFRecordDataset(os.path.join('./data/tfrecord/img_lines_comp.tfrecords'))

    dataseti1 = dataseti1.window(size=total_time_step, shift=total_time_step, stride=1, drop_remainder=False)
    dataseti1 = dataseti1.map(lambda x: x.skip(total_time_step-(val_split+1)))
    dataseti1 = dataseti1.map(lambda x: x.take(val_split-val_skip+1))
    
    dataseti1 = dataseti1.window(size=reach_win_size, shift=reach_shift_cons, stride=1, drop_remainder=False)
    dataseti1 = dataseti1.map(lambda x: x.skip(reach_start_indx))
    dataseti1 = dataseti1.flat_map(lambda x: x)

    dataseti1 = dataseti1.window(size=vert_img_hgt, shift=vert_step, stride=1,drop_remainder=True)
    dataseti1 = dataseti1.map(lambda x: x.flat_map(lambda x1: x1))
    dataseti1 = dataseti1.map(lambda x: x.window(size=vert_img_hgt,shift=1,stride=val_img_range,drop_remainder=True))
    dataseti1 = dataseti1.map(lambda x: x.window(size=time_step, shift=1, stride=1,drop_remainder=True))
    dataseti1 = dataseti1.flat_map(lambda x: x)
    dataseti1 = dataseti1.flat_map(lambda x: x.flat_map(lambda x1: x1))

    dataseti1 = dataseti1.map(_parse_function_org_,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(vert_img_hgt).batch(time_step)

    dataseti1 = dataseti1.batch(val_batch_size, drop_remainder=True)
    #dataseti1 = dataseti1.prefetch(int(25*val_batch_size))

    return dataseti1


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

    """ else :
        if loss_func == 'l1_loss' :
            loss = F.l1_loss(pred, out_flatten,reduction='mean')
        elif loss_func == 'mse_loss' :
            loss = F.mse_loss(pred, out_flatten,reduction='mean')
        elif loss_func == 'huber_loss' :
            loss = F.smooth_l1_loss(pred, out_flatten,reduction='mean') """
    #print(asd)
    return loss

def pytorch_process_inp(sample_batched,vert_img_hgt,output_subtracted,flag_bin_out,out_mean,out_std,
    flag_use_lines,inp_lines_mean,inp_lines_std,flag_reach_use,inp_reach_mean,inp_reach_std,flag_use_imgs):

    if flag_use_imgs :
        inp_flatten = sample_batched['img']
        inp_flatten = inp_flatten[:,:-1,:,:,:]
        inp_flatten = inp_flatten / 255.0
        inp_flatten = inp_flatten.cuda().requires_grad_(False)
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
        bin_out_flatten_left = bin_out_flatten_left.cuda().requires_grad_(False)

        bin_out_flatten_right = lines_last[:,1] - lines_prev[:,1]
        bin_out_flatten_right = torch.reshape(torch.where(bin_out_flatten_right > 2, torch.ones(bin_out_flatten_right.size()),
                                        torch.zeros(bin_out_flatten_right.size())), (-1, 1))
        bin_out_flatten_right_c = torch.reshape(torch.where(bin_out_flatten_right == 1, torch.zeros(bin_out_flatten_right.size()),
                                        torch.ones(bin_out_flatten_right.size())), (-1,1))
        bin_out_flatten_right = torch.cat((bin_out_flatten_right_c,bin_out_flatten_right), 1)
        bin_out_flatten_right = bin_out_flatten_right.cuda().requires_grad_(False)
    else :
        bin_out_flatten_left = None
        bin_out_flatten_right = None


    out_flatten = (out_flatten - out_mean) / out_std 

    out_flatten = out_flatten.float().cuda().requires_grad_(False)

    if flag_use_lines :
        lines_prev_inp = out_flatten_org[:,:-1,:,:]
        lines_prev_inp = torch.reshape(lines_prev_inp, (lines_prev_inp.size()[0],-1))
        lines_prev_inp = (lines_prev_inp - inp_lines_mean) / inp_lines_std
        lines_prev_inp = lines_prev_inp.float().cuda().requires_grad_(False)

        if flag_reach_use :
            reach_id = sample_batched['reaches']
            reach_id = reach_id[:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1]                        
            reach_id = torch.reshape(reach_id, (reach_id.shape[0], 1))
            reach_id = (reach_id - inp_reach_mean) / inp_reach_std
            reach_id = reach_id.float().cuda().requires_grad_(False)
    else :
        lines_prev_inp = None
        reach_id = None

    return inp_flatten,lines_prev_inp,reach_id,out_flatten,bin_out_flatten_left,bin_out_flatten_right,lines_prev


def objective(tm_stp, strt, lr_pow, ad_pow, vert_hgt, vert_step_num, num_epochs,train_shuffle,get_train_mae,transform_constants,
                lstm_layers,lstm_hidden_units,batch_size,inp_bnk,out_bnk,val_split,val_skip,model_type,num_layers,
                model_optim,loss_func,save_mod,load_mod,load_file,skip_training,output_subtracted,train_val_gap,
                out_use_mid,trail_id,flag_batch_norm,dataset_dic,num_cnn_layers,flag_use_lines,pooling_layer,flag_bin_out,
                only_lstm_units,num_branch_layers,branch_layer_neurons,right_loss_weight,num_filter_choice,flag_use_imgs,
                erosion_thresh,num_lft_brn_lyrs,num_rgt_brn_lyrs,lstm_dropout):
    
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
    #model_type = 'CNN_Model_dropout_reg'
    wgt_seed_flag = True
    if wgt_seed_flag :
        torch.manual_seed(0)
    val_skip = val_skip
    #print(val_skip)
    out_use_mid = out_use_mid

    model_optim = model_optim
    
    drop_rate = []
    smooth_flag = False
    time_step = tm_stp
    #atch_size = int(int((500/time_step) - 2)/vert_img_hgt)
    #batch_size = batch_size
    
    
    log_performance = get_train_mae ###number of epochs after which performance metrics are calculated
    log_val_loss_at = get_train_mae
    #log_val_loss_at = 1
    early_stop_flag = False
    early_stop_thresh = 30
    path_to_val_img = os.path.join('./data/img/up_rgb/')
    val_img_ids = [int(f.split('.')[0]) for f in listdir(path_to_val_img) if isfile(join(path_to_val_img, f))]
    val_img_ids.sort()
    org_val_img = val_img_ids
    start_indx = strt
    val_split = val_split
    
    val_numbers_id = (val_split+1) - (time_step-1)
    #print(val_numbers_id)
    #print(val_skip)
    
    if (skip_training == False) and (val_skip == 0) :
        val_img_ids = val_img_ids[-(val_numbers_id):]
    elif skip_training == False :
        val_img_ids = val_img_ids[-(val_numbers_id):-(val_skip)]
    elif (skip_training == True) and (val_skip == 0):
        val_img_ids = val_img_ids[-(val_numbers_id):]
    elif (skip_training == True) and (val_skip > 0):
        val_img_ids = val_img_ids[-(val_numbers_id):-(val_skip)]


    #print(val_img_ids)
    #print(asd)
    total_time_step = 33    ###number of total year images
    num_val_img = len(val_img_ids)
    #print('number of val images.')
    #print(num_val_img)
    #print(asd)

    #val_batch_size = num_val_img
    tr_pr_batch_size = 2222 - (2*(int((vert_img_hgt-1)/2)))
    val_batch_size = tr_pr_batch_size
    #print(val_img_ids)
    #print(num_val_img)
    #print(asd)
    
    if train_val_gap == True :
        data_div_step = total_time_step - (val_split)
    elif train_val_gap == False :
        data_div_step = total_time_step - (val_split - time_step + 2)
    #print(data_div_step)
    #print(asd)
    end_indx = data_div_step-1
    log_hist = 6
    writer = SummaryWriter()
    model_name = writer.get_logdir().split("\\")[1]
    adm_wd = ad_pow
    #adm_wd = 0
    val_img_range = time_step+num_val_img-1
    #print(val_img_range)
    #output_vert_indx = int((vert_img_hgt-1)/2)
    time_win_shift = 1

    #reach_start_indx = 1461 
    #reach_end_num = 377
    reach_start_indx = 0 
    reach_end_num = 0

    reach_shift_cons = 2222
    reach_win_size = reach_shift_cons - reach_end_num 
    reach_end_indx = reach_win_size - 1

    flag_reach_use = True
    flag_sdd_act_data = True
    flag_standardize_actual = True
    loss_func = loss_func

    

    ###out_mode == act_sdd or diff_sdd or act
    out_mode = 'act'
    ###inp_mode == act_sdd or act 
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
        lstm_dropout = lstm_dropout
        )

    if skip_training == False :
        if data_mode == 'imgs' :
            dataset_f = create_dic_train_dataset(data_div_step,total_time_step,start_indx,reach_win_size,reach_shift_cons,reach_start_indx,
                vert_img_hgt,vert_step,end_indx,time_step,time_win_shift,train_shuffle,batch_size,dataset_dic)
        else :
            dataset_f = create_train_dataset(data_div_step,total_time_step,start_indx,reach_win_size,reach_shift_cons,reach_start_indx,
                    vert_img_hgt,vert_step,end_indx,time_step,time_win_shift,train_shuffle,batch_size)

    if skip_training == False :
        if data_mode == 'imgs' :
            time_step_start = start_indx
            time_step_end = data_div_step
            time_step_values = np.arange(time_step_start, time_step_end, 1)
            time_step_values = time_step_values[:-(time_step-1)]

            #tr_pr_batch_size = int(tr_pr_batch_size * len(time_step_values))
            #print(tr_pr_batch_size)
            #print(asd)
            dataset_tr_pr = create_dic_train_dataset(data_div_step,total_time_step,start_indx,reach_win_size,reach_shift_cons,reach_start_indx,
                vert_img_hgt,vert_step,end_indx,time_step,time_win_shift,train_shuffle=False,batch_size=batch_size,dataset_dic=dataset_dic)
        else :
            dataset_tr_pr = create_train_dataset(data_div_step,total_time_step,start_indx,reach_win_size,reach_shift_cons,reach_start_indx,
                    vert_img_hgt,vert_step,end_indx,time_step,time_win_shift,train_shuffle=False,batch_size=batch_size)

    if data_mode == 'imgs' :
        dataseti1 = create_dic_val_dataset(total_time_step,val_split,val_skip,reach_win_size,reach_shift_cons,reach_start_indx,
                vert_img_hgt,vert_step,val_img_range,time_step,batch_size,dataset_dic)
    else:
        dataseti1 = create_val_dataset(total_time_step,val_split,val_skip,reach_win_size,reach_shift_cons,reach_start_indx,vert_img_hgt,vert_step,
                val_img_range,time_step,batch_size)


    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                branch_layer_neurons,num_filter_choice,flag_use_imgs,num_lft_brn_lyrs,num_rgt_brn_lyrs,lstm_dropout)


    
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

    out_mean = transform_constants['out_mean']
    out_std = transform_constants['out_std']

    #print(inp_mean)
    #print(out_mean)

    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    print('started training ..........')
    for epoch in range(EPOCHS):

        model.train()
        counter = 0
        epoch_loss = 0
        batch_loss_counter = 0
        batch_loss = 0
        batch_counter = 0
        #print(dataset_f)

        if skip_training == False :
            for i_batch, sample_batched in enumerate(dataset_f) :
                #print(i_batch)
                #print(out_flatten)
                #print(asd)
                """ print(concat_img.shape)
                print(lines.shape)
                print(year_id.shape)
                print(reach_id.shape)
                print(asd) """
                """ inp_flatten, out_flatten, _, _ = process_tf_dataset(inp_flatten, year_id, reach_id, 
                            sdd_output,inp_mode,inp_lr_flag, out_mode,out_lr_tag,flag_standardize_actual,
                            flag_reach_use,batch_size,time_step,vert_img_hgt,flag_sdd_act_data,
                            output_subtracted,out_use_mid,concat_img) """
        

                """ if inp_mode == 'act' and flag_standardize_actual == True :
                    if inp_lr_flag != 'img' :
                        inp_flatten = (inp_flatten - inp_mean) / inp_std 
                    elif inp_lr_flag == 'img' : """
                
                #print(inp_flatten[0:prin_count,:])
                #print(out_flatten[0:prin_count,:])
                #print(inp_flatten[i,:,:,:])

                #print(asd)

                #print(inp_flatten.shape)
                #print(out_flatten.shape)

                #print(asd)

                """ if i_batch == 0 :
                    print(inp_flatten[0,:,:,:,:])
                    print(out_flatten[0,:]) """
                #print(asd)


                #inp_flatten = np.asarray(inp_flatten, dtype=np.float32)
                #out_flatten = np.asarray(out_flatten, dtype=np.float32)
                #print(inp_flatten.shape)
                #print(out_flatten.shape)
                #print(asd)

                #out_flatten = torch.reshape(out_flatten, (batch_size,-1))
                

                """ print(inp_flatten.shape)
                print(lines_prev_inp.shape)
                print(reach_id.shape)
                print(out_flatten.shape)
                print(bin_out_flatten_left.shape)
                print(bin_out_flatten_right.shape)

                print(asd) """

                inp_flatten,lines_prev_inp,reach_id,out_flatten,bin_out_flatten_left, \
                bin_out_flatten_right,_ = pytorch_process_inp(sample_batched,vert_img_hgt,
                                            output_subtracted,flag_bin_out,out_mean,out_std,
                                            flag_use_lines,inp_lines_mean,inp_lines_std,
                                            flag_reach_use,inp_reach_mean,inp_reach_std,flag_use_imgs)

                if i_batch == 0 :
                    #print(inp_flatten[i_batch,:,:,:,:])
                    if flag_use_imgs :
                        print(inp_flatten.shape)
                        print(inp_flatten.dtype)
                    print(out_flatten.shape)
                    print(out_flatten.dtype)

                    if lines_prev_inp != None :
                        #print(lines_prev_inp)
                        print(lines_prev_inp.shape)
                        #print(reach_id)
                        print(lines_prev_inp.dtype)
                        print(reach_id.shape)
                        print(reach_id.dtype)
                    
                    #print(bin_out_flatten_left.dtype)
                    #print(bin_out_flatten_right.dtype)


                optimizer.zero_grad()

                #if flag_use_lines :
                _, pred_left, pred_right, pred_binl, pred_binr = model(inp_flatten, lines_prev_inp, reach_id)
                #else :
                #    pred, _,_,_,_ = model(inp_flatten, lines_prev_inp, reach_id)

                
                
                loss = calculate_loss(pred_left, pred_right, pred_binl, pred_binr, out_flatten, bin_out_flatten_left,
                        bin_out_flatten_right,flag_use_lines,flag_bin_out,loss_func,right_loss_weight)

                loss.backward()
                optimizer.step()

                epoch_loss = epoch_loss+loss
                counter += 1

                batch_loss = batch_loss + loss
                
                batch_counter += 1

                if ((i_batch+1)*batch_size) <= ((batch_loss_counter+1)*100) < ((i_batch+2)*batch_size) :
                    
                    batch_loss_counter += 1
                    batch_loss = batch_loss / batch_counter

                    batch_template = 'Epoch {}, {} batch Loss: {}'
                    print(batch_template.format(epoch+1, i_batch+1, batch_loss))
                    batch_counter = 0
                    batch_loss = 0

                """ if i_batch == 0 :
                    inp_tuple = (inp_flatten, lines_prev, reach_id)
                    writer.add_graph(model, inp_tuple) """
        
            flag_sdd_act_data == False
            #print(asd)
            avg_epoch_loss = epoch_loss / counter
            template = 'Epoch {}, Train Loss: {}'
            print(template.format(epoch+1,avg_epoch_loss))

            #print(asd)
            writer.add_scalar('Loss/train', avg_epoch_loss, epoch+1)
            train_losses.append(avg_epoch_loss.item())

            #if epoch == 0 :
                #if model_type == 'ANN' :
            #    writer.add_graph(model, (inp_flatten, lines_prev, reach_id) )
                #elif model_type == 'LSTM' :
                #inp_flatten = inp_flatten[:,:-1]
                #inp_flatten = torch.reshape(inp_flatten, (-1,(time_step-1),int(vert_img_hgt * 1)))
                #writer.add_graph(model, inp_flatten)
                #    pass

            if epoch % log_hist == log_hist-1:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram('parameters_'+str(name), param.data, epoch + 1)
                        writer.add_histogram('gradients'+str(name), param.grad, epoch + 1)

            """ if (save_mod == True) and (epoch % model_save_at == 0) :
                model_save(model, optimizer, model_name) """

        
        #print(asd)
        val_epoch_loss = 0
        counter_val = 0

        #torch.cuda.empty_cache()
        #print(asd)
        model.eval()
        #if epoch % log_val_loss_at == log_val_loss_at-1 :
        if True :
            print('starting validation .......')
            with torch.no_grad():
                
                #print('hel')
                #if epoch % log_performance == log_performance-1:
                pred_list = []
                actual_list = []
                prev_actual_list = []
                prev_reach_ids = []
                prev_year_ids = []
                act_reach_ids = []
                act_year_ids = []

                counter_temp = 0

                prev_sum_temp = []

                for i_batch, sample_batched in enumerate(dataseti1) :
                    #print(i_batch)
                    #for inp_flatten_org, year_id, reach_id, _, sdd_output in dataseti1:
                    #year_id = np.reshape(year_id, (val_batch_size,time_step,vert_img_hgt,1))
                    #reach_id = np.reshape(reach_id, (val_batch_size,time_step,vert_img_hgt,1))
                    
                    """ if counter_temp == 0:
                        prin_count = 30
                        print(inp_flatten_org[:,:,:,:])
                        print(year_id[:,:,:,:])
                        print(reach_id[:,:,:,:])
                    #print(asd)
                    counter_temp += 1 """
                    
                    """ inp_flatten_org, out_flatten, _, _ = process_tf_dataset(inp_flatten_org, year_id, reach_id, sdd_output,inp_mode,inp_lr_flag,
                        out_mode,out_lr_tag,flag_standardize_actual,flag_reach_use,val_batch_size,time_step,vert_img_hgt,flag_sdd_act_data,output_subtracted,out_use_mid) """

                    """ prin_count = 30
                    print(inp_flatten_org[0:prin_count,:])
                    print(out_flatten[0:prin_count,:])
                    print(asd) """

                    """ if (inp_mode == 'act') and (flag_standardize_actual == True) :
                        inp_flatten = (inp_flatten_org - inp_mean) / inp_std 
                        out_flatten = (out_flatten - out_mean) / out_std  """

                    

                    """ print(inp_flatten.shape)
                    print(lines_prev_inp.shape)
                    print(reach_id.shape)
                    print(out_flatten.shape)
                    print(bin_out_flatten_left.shape)
                    print(bin_out_flatten_right.shape)

                    print(inp_flatten.dtype)
                    print(lines_prev_inp.dtype)
                    print(reach_id.dtype)
                    print(out_flatten.dtype)
                    print(bin_out_flatten_left.dtype)
                    print(bin_out_flatten_right.dtype) """

                    inp_flatten,lines_prev_inp,reach_id,out_flatten,bin_out_flatten_left, \
                    bin_out_flatten_right,lines_prev = pytorch_process_inp(sample_batched,vert_img_hgt,
                                            output_subtracted,flag_bin_out,out_mean,out_std,
                                            flag_use_lines,inp_lines_mean,inp_lines_std,
                                            flag_reach_use,inp_reach_mean,inp_reach_std,flag_use_imgs)

                    if i_batch == 0 :
                        #print(inp_flatten[i_batch,:,:,:,:])
                        if flag_use_imgs :
                            print(inp_flatten.shape)
                            print(inp_flatten.dtype)
                        print(out_flatten.shape)
                        print(out_flatten.dtype)

                        if lines_prev_inp != None :
                            #print(lines_prev_inp)
                            print(lines_prev_inp.shape)
                            #print(reach_id)
                            print(lines_prev_inp.dtype)
                            print(reach_id.shape)
                            print(reach_id.dtype)

                    #print(asd)

                    #if flag_use_lines :
                    _, pred_left, pred_right, pred_binl, pred_binr = model(inp_flatten, lines_prev_inp, reach_id)
                    #else :
                    #    pred, _,_,_,_ = model(inp_flatten, lines_prev_inp, reach_id)
                    

                    loss = calculate_loss(pred_left, pred_right, pred_binl, pred_binr, out_flatten, bin_out_flatten_left,
                        bin_out_flatten_right,flag_use_lines,flag_bin_out,loss_func,right_loss_weight)


                    val_epoch_loss = val_epoch_loss+loss
                    counter_val += 1

                    #print(val_epoch_loss)
                    #print(asd)
                    if inp_lr_flag == 'img' :
                        #print(lines_last.shape)
                        #print(lines_prev.shape)
                        out_flatten = out_flatten.cpu()
                        out_flatten = out_flatten.numpy()
                        
                        #if flag_use_lines :
                        pred_left = pred_left.cpu()
                        pred_left = pred_left.numpy()
                        pred_right = pred_right.cpu()
                        pred_right = pred_right.numpy()

                        pred = np.concatenate((pred_left,pred_right), axis=1)
                            
                        """ else :
                            pred = pred.cpu()
                            pred = pred.numpy() """

                        #lines_prev = lines_prev.cpu()
                        #lines_prev = lines_prev.numpy()

                        out_flatten = np.add(np.multiply(out_flatten, out_std), out_mean)
                        pred = np.add(np.multiply(pred, out_std), out_mean)
                        extra_samples = 0

                        if ((i_batch+1) == len(dataseti1)) and (batch_size != lines_prev.shape[0]) :
                            extra_samples = int(batch_size -lines_prev.shape[0])
                            lines_prev = np.pad(lines_prev, ((0,extra_samples),(0,0)) )
                            out_flatten = np.pad(out_flatten, ((0,extra_samples),(0,0)) )
                            pred = np.pad(pred, ((0,extra_samples),(0,0)) )

                            """ print(lines_prev.shape)
                            #print(lines_prev)
                            print(out_flatten.shape)
                            print(pred.shape)
                            print(asd) """

                        lines_prev = np.asarray(lines_prev)
                        """ print(lines_prev.shape)
                        print(out_flatten.shape)
                        print(pred.shape) """

                        prev_actual_list.append(lines_prev)
                        actual_list.append(out_flatten)
                        pred_list.append(pred) 

                        #print(out_flatten.shape)
                        #print(pred.shape)
                        #print(lines_prev.shape)

                        #print(asd)


                    """ if flag_reach_use == True :
                        #print(inp_flatten_org)
                        if vert_img_hgt > 1 :
                            inp_flatten_org = inp_flatten_org[:,:-1]
                        elif vert_img_hgt == 1 :
                            inp_flatten_org = inp_flatten_org[:,:-vert_img_hgt]
                        #print(inp_flatten_org[0:2,:])
                        #print(inp_flatten_org)
                        #print(asd)
                        if inp_lr_flag == 'left' or inp_lr_flag == 'right' :
                            inp_flatten_org = np.reshape(inp_flatten_org,(val_batch_size,time_step-1,vert_img_hgt,1))
                        elif inp_lr_flag == 'both' :
                            inp_flatten_org = np.reshape(inp_flatten_org,(val_batch_size,time_step-1,vert_img_hgt,2))
                        
                        #print(inp_flatten_org)
                        #print(asd)
                        if int(vert_img_hgt/2) == 0 :
                            inp_flatten_org = inp_flatten_org[:,-2,0:int(vert_img_hgt/2)+1,:]
                        else :
                        inp_flatten_org = inp_flatten_org[:,-1,:,:]
                        #print(inp_flatten_org)
                        
                        #prev_inp_flatten = np.reshape(inp_flatten_org,(val_batch_size,-1))
                        prev_inp_flatten = inp_flatten_org
                        #print(prev_inp_flatten)
                        #print(asd)
                        #prev_inp_flatten = inp_flatten_org[:,-2]
                    elif flag_reach_use == False :
                        prev_inp_flatten = inp_flatten_org[:,-1]

                    #print(prev_inp_flatten.shape)

                    #print(asd)

                    out_flatten = out_flatten.cpu()
                    out_flatten = out_flatten.numpy()
    
                    pred_np = pred.cpu()
                    pred_np = pred_np.numpy()
                    #print(year_id)
                    #print(reach_id)
                    year_id = np.reshape(year_id, (val_batch_size,time_step,vert_img_hgt,1))
                    reach_id = np.reshape(reach_id, (val_batch_size,time_step,vert_img_hgt,1))
                    #print(year_id)

                    year_id_prev = year_id[:,time_step-2:time_step-1,:,:]
                    reach_id_prev = reach_id[:,time_step-2:time_step-1,:,:]
                    year_id_act = year_id[:,time_step-1:time_step,:,:]
                    reach_id_act = reach_id[:,time_step-1:time_step,:,:]

                    if out_lr_tag == 'left' or out_lr_tag == 'right':
                        val_extra_dim = 1
                    elif out_lr_tag == 'both' :
                        val_extra_dim = 2
                    #print(year_id_prev)
                    #print(asd)

                    prev_time_step = np.reshape(prev_inp_flatten,(val_batch_size,-1,val_extra_dim))
                    year_id_prev = np.reshape(year_id_prev,(val_batch_size,-1,1))
                    reach_id_prev = np.reshape(reach_id_prev,(val_batch_size,-1,1))
                    year_id_act = np.reshape(year_id_act,(val_batch_size,-1,1))
                    reach_id_act = np.reshape(reach_id_act,(val_batch_size,-1,1))


                    out_flatten = np.reshape(out_flatten,(val_batch_size,-1,val_extra_dim))
                    pred_np = np.reshape(pred_np,(val_batch_size,-1,val_extra_dim))

                    #print(out_flatten.shape)
                    #print(prev_time_step.shape)
                    #print(asd)

                    if out_lr_tag == 'left' :
                        if out_use_mid == True :
                            np_zero_out = np.zeros((val_batch_size,1,val_extra_dim))

                        np_zero = np.zeros((val_batch_size,vert_img_hgt,val_extra_dim))
                        
                        prev_time_step = np.concatenate((prev_time_step,np_zero),axis=2) 
                        if out_use_mid == True :
                            out_flatten = np.concatenate((out_flatten,np_zero_out),axis=2)
                            pred_np = np.concatenate((pred_np,np_zero_out),axis=2)
                        elif out_use_mid == False :
                            out_flatten = np.concatenate((out_flatten,np_zero),axis=2)
                            pred_np = np.concatenate((pred_np,np_zero),axis=2)

                    elif out_lr_tag == 'right' :
                        if out_use_mid == True :
                            np_zero_out = np.zeros((val_batch_size,1,val_extra_dim))

                        np_zero = np.zeros((val_batch_size,vert_img_hgt,val_extra_dim))


                        prev_time_step = np.concatenate((np_zero,prev_time_step),axis=2) 

                        if out_use_mid == True :
                            out_flatten = np.concatenate((np_zero_out,out_flatten),axis=2)
                            pred_np = np.concatenate((np_zero_out,pred_np),axis=2)
                        elif out_use_mid == False :
                            out_flatten = np.concatenate((np_zero,out_flatten),axis=2)  
                            pred_np = np.concatenate((np_zero,pred_np),axis=2)

                    #print(prev_time_step.shape)
                    #print(out_flatten.shape)
                    #print(asd)
                    #print(out_flatten)
                    #out_mean = np.reshape(out_mean,(out_mean.shape[0],1))
                    #print(out_mean.shape)

                    rescale_out = True
                    if rescale_out == True :
                        if out_lr_tag == 'right' :  
                            out_flatten[:,:,1] = np.add(np.multiply(out_flatten[:,:,1], out_std), out_mean)
                            pred_np[:,:,1] = np.add(np.multiply(pred_np[:,:,1], out_std), out_mean)
                        elif out_lr_tag == 'left' :
                            out_flatten[:,:,0] = np.add(np.multiply(out_flatten[:,:,0], out_std), out_mean)
                            pred_np[:,:,0] = np.add(np.multiply(pred_np[:,:,0], out_std), out_mean)
                        elif out_lr_tag == 'both' :
                            out_mean = np.reshape(out_mean,(vert_img_hgt,2))
                            out_std = np.reshape(out_std,(vert_img_hgt,2))
                            out_flatten = np.add(np.multiply(out_flatten, out_std), out_mean)
                            pred_np = np.add(np.multiply(pred_np, out_std), out_mean)
                    #print(out_flatten.shape)
                    #print(out_flatten)
                    #print(asd)
                    running_sum = False

                    if out_use_mid == True :
                        prev_actual_list.append(prev_time_step[:,int((vert_img_hgt-1)/2):int((vert_img_hgt-1)/2)+1,:])
                    elif out_use_mid == False :
                        prev_actual_list.append(prev_time_step[:,0:vert_step,:])
                    actual_list.append(out_flatten[:,0:vert_step,:])
                    pred_list.append(pred_np[:,0:vert_step,:]) """

                    """ prev_year_ids.append(year_id_prev[:,0:vert_step,:])
                    prev_reach_ids.append(reach_id_prev[:,0:vert_step,:])
                    act_year_ids.append(year_id_act[:,0:vert_step,:])
                    act_reach_ids.append(reach_id_act[:,0:vert_step,:]) """


                    #print(prev_time_step)
                    #print(asd)
                #print(prev_time_step.shape)

                avg_val_epoch_loss = val_epoch_loss / counter_val
                template = 'Epoch {}, Val_Loss: {}'
                print(template.format(epoch+1,avg_val_epoch_loss))

                writer.add_scalar('Loss/Val', avg_val_epoch_loss, epoch+1)
                val_losses.append(avg_val_epoch_loss.item())

                #writer.add_scalar('Loss/variance', (avg_epoch_loss - avg_val_epoch_loss), epoch+1)
                if skip_training == False :
                    writer.add_scalars('Loss/train_val', {'train':avg_epoch_loss,
                                            'val':avg_val_epoch_loss}, epoch+1)
                elif skip_training == True :
                    pass
            #writer.add_scalars('Loss/bias_variance', {'bias':avg_epoch_loss,
            #                        'variance':(avg_epoch_loss - avg_val_epoch_loss)}, epoch+1)
            """ if epoch == 0 :
                best_val_loss = avg_val_epoch_loss
            else :
                if avg_val_epoch_loss < best_val_loss :
                    best_val_loss = avg_val_epoch_loss
                    early_stop_counter = 0
                else :
                    early_stop_counter += 1 """
            
            #torch.cuda.empty_cache()
            
            ###logging performance metrics
            if epoch % log_performance == log_performance-1 :
                #print(prev_actual_list)
                #print(actual_list)
                #print(pred_list)
                #print(actual_list.shape)
                #print(asd)
                prev_actual_list = process_prev(prev_actual_list,num_val_img,prev_year_ids,prev_reach_ids,vert_img_hgt,vert_step,inp_mode,
                                    flag_standardize_actual,transform_constants,extra_samples)
                actual_list = process_diffs(actual_list,num_val_img, prev_actual_list,act_year_ids,act_reach_ids,vert_img_hgt,out_mode,
                                    flag_standardize_actual,transform_constants,output_subtracted,extra_samples)
                pred_list = process_diffs_pred(pred_list,num_val_img, prev_actual_list,act_year_ids,act_reach_ids,vert_img_hgt,out_mode,
                                    flag_standardize_actual,transform_constants,output_subtracted,extra_samples)
                
                #print(prev_actual_list.shape)
                #print(actual_list.shape)
                #print(pred_list.shape)
                #print(asd)

                if output_subtracted == False :
                    if inp_lr_flag == 'right' :
                        actual_list[:,:,0] = 0
                        pred_list[:,:,0] = 0 
                    elif inp_lr_flag == 'left' :
                        actual_list[:,:,1] = 0
                        pred_list[:,:,1] = 0 
                
                #print(prev_actual_list.shape)
                #print(actual_list)
                #print(pred_list.shape)
                #print(actual_list.shape)
                #print(asd)
                test_logs, test_logs_scores, imp_val_logs = log_performance_metrics(pred_list,actual_list,prev_actual_list,
                                                    num_val_img, epoch, val_img_ids,writer,erosion_thresh)
                print('validation reach MAE ......')
                print(test_logs['reach_mae'])
                val_maes.append(test_logs['reach_mae'])

        """ if early_stop_flag:
            if early_stop_counter > early_stop_thresh :
                print('early stopping as val loss is not improving ........')
                #model_save(model, optimizer, model_name)
                break """
        
        """ if epoch % get_train_mae == get_train_mae-1 :
            train_mae = train_performance(dataset_tr_pr,inp_mode,inp_lr_flag,out_mode,out_lr_tag,flag_standardize_actual,flag_reach_use,
                batch_size,time_step,vert_img_hgt,flag_sdd_act_data,inp_mean,inp_std,out_mean,out_std,model,loss_func,
                transform_constants,num_val_img,output_subtracted,out_use_mid)
            writer.add_scalar('Loss/Train_MAE', train_mae, epoch+1)
            train_maes.append(train_mae) """

        
    #print(actual_list[0:20,0,:])
    #print(actual_list[0:20,1,:])


    #print(asd)
    #if skip_training == True :
    for iter_num in range((num_val_img)):
        temp_conf = wrt_img(iter_num, actual_list, prev_actual_list, pred_list, val_img_ids, writer,smooth_flag,
                    reach_start_indx,out_use_mid,vert_img_hgt)

        if iter_num == 0 :
            final_conf = temp_conf
        else :
            final_conf = final_conf + temp_conf
    
    plt_conf_mat(final_conf, 'total_test_confusion_matrix', writer) #'trial_name':str(model_name),
    
    hyperparameter_defaults.update(epochs=epoch+1)
    hyperparameter_defaults.update(trial_name=model_name)
    
    """     hparam_logs = {'hparam/train_loss':avg_epoch_loss,'hparam/val_loss':avg_val_epoch_loss,
        'hparam/augmented_metric':test_logs_scores['AM_score'], 'hparam/lr_f1_score':test_logs_scores['lr_f1_score'], 'hparam/lr_reach_mae':test_logs_scores['lr_reach_mae'],
        'hparam/left_reach_mae':test_logs['reach_mae'][0],'hparam/right_reach_mae':test_logs['reach_mae'][1],
        'hparam/left_f1score':test_logs_scores['left_f1'],'hparam/right_f1score':test_logs_scores['right_f1'],
        'hparam/left_pos_mae':test_logs['pos_mae'][0],'hparam/right_pos_mae':test_logs['pos_mae'][1],
        'hparam/left_non_erosion_mae':test_logs['full_non_erosion_mae'][0],'hparam/right_non_erosion_mae':test_logs['full_non_erosion_mae'][1],
        'hparam/left_neg_mae':test_logs['neg_mae'][0],'hparam/right_neg_mae':test_logs['neg_mae'][1],
        'hparam/left_pos_neg_mae':test_logs['full_act_mae'][0],'hparam/right_pos_neg_mae':test_logs['full_act_mae'][1],
        'hparam/left_act_ero_diff':test_logs['mean_ero_diff'][0],'hparam/right_act_ero_diff':test_logs['mean_ero_diff'][1],
        'hparam/left_act_overfit_dif':test_logs['act_overfit_diff'][0],'hparam/right_act_overfit_dif':test_logs['act_overfit_diff'][1]}
    """    
    if skip_training == False :
        #torch.cuda.empty_cache()
        print('calculating training MAE performance ........')
        train_mae = pt_train_per(dataset_tr_pr,inp_mode,inp_lr_flag,out_mode,out_lr_tag,flag_standardize_actual,flag_reach_use,
            batch_size,time_step,vert_img_hgt,flag_sdd_act_data,inp_lines_mean,inp_lines_std,out_mean,out_std,model,loss_func,
            transform_constants,num_val_img,output_subtracted,out_use_mid,flag_use_lines,flag_use_imgs)
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
    


    """ hparam_logs = {'hparam/val_loss':avg_val_epoch_loss,
        'hparam/augmented_metric':test_logs_scores['AM_score'],
        'hparam/Left_AM_Metric':test_logs_scores['Left_AM_Score'],
        'hparam/Right_AM_Metric':test_logs_scores['Right_AM_Score'],
        'hparam/lr_reach_mae':test_logs_scores['lr_reach_mae'],
        'hparam/left_reach_mae':test_logs['reach_mae'][0],'hparam/right_reach_mae':test_logs['reach_mae'][1],
        'hparam/left_precision':test_logs_scores['left_precision'],'hparam/right_precision':test_logs_scores['right_precision'],
        'hparam/left_recall':test_logs_scores['left_recall'],'hparam/right_recall':test_logs_scores['right_recall'],
        'hparam/left_right_precision':test_logs_scores['lft_rgt_precision'],'hparam/left_right_recall':test_logs_scores['lft_rgt_recall'],
        'hparam/left_pos_mae':test_logs['pos_mae'][0],'hparam/right_pos_mae':test_logs['pos_mae'][1],
        'hparam/left_non_erosion_mae':test_logs['full_non_erosion_mae'][0],'hparam/right_non_erosion_mae':test_logs['full_non_erosion_mae'][1],
        'hparam/left_neg_mae':test_logs['neg_mae'][0],'hparam/right_neg_mae':test_logs['neg_mae'][1],
        'hparam/left_pos_neg_mae':test_logs['full_act_mae'][0],'hparam/right_pos_neg_mae':test_logs['full_act_mae'][1],
        'hparam/left_act_ero_diff':test_logs['mean_ero_diff'][0],'hparam/right_act_ero_diff':test_logs['mean_ero_diff'][1],
        'hparam/left_act_overfit_dif':test_logs['act_overfit_diff'][0],'hparam/right_act_overfit_dif':test_logs['act_overfit_diff'][1]} """
    
    hparam_logs = {'hparam/val_loss':avg_val_epoch_loss,
        'hparam/lr_reach_mae':test_logs_scores['lr_reach_mae'],
        'hparam/left_reach_mae':test_logs['reach_mae'][0],'hparam/right_reach_mae':test_logs['reach_mae'][1],
        'hparam/left_precision':test_logs_scores['left_precision'],'hparam/right_precision':test_logs_scores['right_precision'],
        'hparam/left_recall':test_logs_scores['left_recall'],'hparam/right_recall':test_logs_scores['right_recall']}

    """ 'hparam/left_F1Score':float(2*((test_logs_scores['left_precision']*test_logs_scores['left_recall'])/(test_logs_scores['left_precision']+test_logs_scores['left_recall']))),
        'hparam/Right_F1Score':float(2*((test_logs_scores['right_precision']*test_logs_scores['right_recall'])/(test_logs_scores['right_precision']+test_logs_scores['right_recall']))), """
    
    if skip_training == False :
        hparam_logs = {**temp_hparam,**hparam_logs}
    elif skip_training == True :
        pass

    #print(hyperparameter_defaults)
    #print(imp_val_logs)
    #print(hparam_logs)
    
    hparam_logs.update(imp_val_logs)
    writer.add_hparams(hyperparameter_defaults, hparam_logs)
    writer.close()
    if save_mod == True:
        model_save(model, optimizer, model_name)

    #avg_act_ero_mae = (test_logs['full_act_mae'][0] + test_logs['full_act_mae'][1] ) / 2 
    #torch.cuda.empty_cache()

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
        #if i == 0 :
        #    break


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



if __name__ == "__main__":

    super_epochs = 25
    num_epochs = 4

    def objtv(trial):

        dataset_dic = get_all_data()
        #print(asd)

        super_epochs = 25
        num_epochs = 4

        trail_id = trial.number
        load_models_list = []
        transform_constants_list = []
        #tm_stp=trial.suggest_int('time_step', 3, 6, 1)
        tm_stp = 6
        #lr_pow = trial.suggest_discrete_uniform('learning_rate', -5.0, -3.0, 0.5)
        lr_pow = -4.0
        #lstm_hidden_units = trial.suggest_int('neurons_per_layer', 200, 500, 50 )
        lstm_hidden_units = 100
        #batch_size_pow = trial.suggest_int('batch_size_power', 2, 6 , 1)
        batch_size_pow = 4
        #num_layers = trial.suggest_int('num_of_layers', 3, 5, 1)
        num_layers = 0
        num_cnn_layers = 6
        #strt = trial.suggest_int('starting_year', 0, 20, 5)
        strt = 0
        #vert_hgt = trial.suggest_int('vertical_window_size', 128, 256, 128)
        vert_hgt = 96
        #loss_func = trial.suggest_categorical('loss_function', ['mse_loss', 'l1_loss', 'huber_loss','log_cosh])
        loss_func = 'mse_loss'
        #output_subtracted = trial.suggest_categorical('output_subtracted', [0,False])
        #lstm_layers = trial.suggest_int('lstm_depth_layers', 1, 3, 1)
        lstm_layers = 1
        #model_type = trial.suggest_categorical('model_type', ['ANN', 'LSTM'])
        model_type = 'CNN_LSTM'
        #flag_batch_norm_bin = trial.suggest_int('batch_norm', 0, 1, 1)
        #flag_batch_norm_bin = 0
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
        branch_layer_neurons = 50
        #right_loss_weight = trial.suggest_discrete_uniform('right_loss_weight', 0.5, 0.95, 0.05)
        right_loss_weight = 0.4
        #num_filter_choice = trial.suggest_int('num_filter_choice', 0, 1, 1)
        num_filter_choice = 1
        model_optim = 'Adam'
        #ad_pow = 1*(10**-1.0)
        ad_pow = 0
        erosion_thresh = 1

        for j in range(super_epochs):

            cross_val_nums = 1
            val_split_org = 3
            val_skip = 2
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
            inp_bnk='img'
            out_bnk='both'
            
            #loss_func='mse_loss'
            
            train_shuffle = True
            train_val_gap = False
            #flg_btch_list = [False, True]
            #flag_batch_norm = flg_btch_list[int(flag_batch_norm_bin)]
            flag_batch_norm = True
            
            #model_type = 'ANN'
            #num_layers_list = [1,3,5,7,9,12,14]
    

            crs_train_ls = []
            crs_val_ls = []
            crs_train_maes = []
            crs_val_maes = []
            crs_test_maes = []

            for i in range(cross_val_nums) :
                val_split = val_split_org + (tm_stp - 2)

                if j == 0 :
                    model_name, train_losses, val_losses, train_maes, val_maes, hparam_def, transform_constants = objective(tm_stp=tm_stp,strt=strt,lr_pow=lr_pow,ad_pow=ad_pow,vert_hgt=vert_hgt,vert_step_num=vert_step_num,num_epochs=num_epochs,train_shuffle=train_shuffle,
                    get_train_mae=get_train_mae,transform_constants=None,lstm_layers=lstm_layers,lstm_hidden_units=lstm_hidden_units,batch_size=batch_size,inp_bnk=inp_bnk,out_bnk=out_bnk,val_split=val_split,val_skip=val_skip,model_type=model_type,num_layers=num_layers,
                    model_optim=model_optim,loss_func=loss_func,save_mod=True,load_mod=False,load_file=None,skip_training=False,output_subtracted=output_subtracted,train_val_gap=train_val_gap,out_use_mid=out_use_mid,trail_id=trail_id,flag_batch_norm=flag_batch_norm,dataset_dic=dataset_dic,
                    num_cnn_layers=num_cnn_layers,flag_use_lines=flag_use_lines,pooling_layer=pooling_layer,flag_bin_out=flag_bin_out,only_lstm_units=only_lstm_units,num_branch_layers=num_branch_layers,branch_layer_neurons=branch_layer_neurons,right_loss_weight=right_loss_weight,
                    num_filter_choice=num_filter_choice,flag_use_imgs=flag_use_imgs,erosion_thresh=erosion_thresh,num_lft_brn_lyrs=num_lft_brn_lyrs,num_rgt_brn_lyrs=num_rgt_brn_lyrs,lstm_dropout=lstm_dropout)
                elif j > 0 :
                    model_name, train_losses, val_losses, train_maes, val_maes, hparam_def, transform_constants = objective(tm_stp=tm_stp,strt=strt,lr_pow=lr_pow,ad_pow=ad_pow,vert_hgt=vert_hgt,vert_step_num=vert_step_num,num_epochs=num_epochs,train_shuffle=train_shuffle,
                    get_train_mae=get_train_mae,transform_constants=transform_constants_list[i],lstm_layers=lstm_layers,lstm_hidden_units=lstm_hidden_units,batch_size=batch_size,inp_bnk=inp_bnk,out_bnk=out_bnk,val_split=val_split,val_skip=val_skip,model_type=model_type,num_layers=num_layers,
                    model_optim=model_optim,loss_func=loss_func,save_mod=True,load_mod=True,load_file=load_models_list[0],skip_training=False,output_subtracted=output_subtracted,train_val_gap=train_val_gap,out_use_mid=out_use_mid,trail_id=trail_id,flag_batch_norm=flag_batch_norm,dataset_dic=dataset_dic,
                    num_cnn_layers=num_cnn_layers,flag_use_lines=flag_use_lines,pooling_layer=pooling_layer,flag_bin_out=flag_bin_out,only_lstm_units=only_lstm_units,num_branch_layers=num_branch_layers,branch_layer_neurons=branch_layer_neurons,right_loss_weight=right_loss_weight,
                    num_filter_choice=num_filter_choice,flag_use_imgs=flag_use_imgs,erosion_thresh=erosion_thresh,num_lft_brn_lyrs=num_lft_brn_lyrs,num_rgt_brn_lyrs=num_rgt_brn_lyrs,lstm_dropout=lstm_dropout)

                    load_models_list.pop(0)

                load_models_list.append(model_name)
                if j == 0 :
                    transform_constants_list.append(transform_constants)
                #print(val_losses)
                crs_train_ls.append(train_losses)
                crs_val_ls.append(val_losses)
                crs_train_maes.append(train_maes)
                crs_val_maes.append(val_maes)

                _, _, _, _, test_val_maes, _, _ = objective(tm_stp=tm_stp,strt=strt,lr_pow=lr_pow,ad_pow=ad_pow,vert_hgt=vert_hgt,vert_step_num=vert_step_num,num_epochs=1,train_shuffle=train_shuffle,get_train_mae=1,transform_constants=transform_constants,
                    lstm_layers=lstm_layers,lstm_hidden_units=lstm_hidden_units,batch_size=batch_size,inp_bnk=inp_bnk,out_bnk=out_bnk,val_split=val_split-(val_split_org-val_skip),val_skip=(val_skip-1),model_type=model_type,num_layers=num_layers,
                    model_optim=model_optim,loss_func=loss_func,save_mod=False,load_mod=True,load_file=model_name,skip_training=True,output_subtracted=output_subtracted,train_val_gap=train_val_gap,out_use_mid=out_use_mid,trail_id=trail_id,flag_batch_norm=flag_batch_norm,
                    dataset_dic=dataset_dic,num_cnn_layers=num_cnn_layers,flag_use_lines=flag_use_lines,pooling_layer=pooling_layer,flag_bin_out=flag_bin_out,only_lstm_units=only_lstm_units,num_branch_layers=num_branch_layers,branch_layer_neurons=branch_layer_neurons,
                    right_loss_weight=right_loss_weight,num_filter_choice=num_filter_choice,flag_use_imgs=flag_use_imgs,erosion_thresh=erosion_thresh,num_lft_brn_lyrs=num_lft_brn_lyrs,num_rgt_brn_lyrs=num_rgt_brn_lyrs,lstm_dropout=lstm_dropout)
                
                crs_test_maes.append(test_val_maes)

                val_split_org = val_split_org + 1
                val_skip = val_split_org - 1
                if change_start == True :
                    strt = strt - 1

                #print(asd)
                """ print(crs_train_ls)
                print(crs_val_ls)
                print(crs_train_maes)
                print(crs_val_maes) """

                


            crs_train_ls = np.mean(np.asarray(crs_train_ls),axis=0)
            crs_val_ls = np.mean(np.asarray(crs_val_ls),axis=0)
            crs_train_maes = np.mean(np.asarray(crs_train_maes),axis=0)
            crs_val_maes = np.mean(np.asarray(crs_val_maes),axis=0)
            crs_test_mae = np.mean(np.asarray(crs_test_maes),axis=0)

            #print(crs_val_ls)

            writer = SummaryWriter()

            for i in range(crs_train_ls.shape[0]) :
                
                writer.add_scalar('cross_val/train',crs_train_ls[i], i+1)
            
            for i in range(crs_val_ls.shape[0]) :
                
                writer.add_scalar('cross_val/val', crs_val_ls[i], (i+1)*get_train_mae)

            if out_bnk == 'right' :
                crs_val_maes = crs_val_maes[:,1]
                crs_test_mae = crs_test_mae[:,1]
                crs_train_maes = crs_train_maes[:,1]

            elif out_bnk == 'left' :
                crs_val_maes = crs_val_maes[:,0]
                crs_test_mae = crs_test_mae[:,0]
                crs_train_maes = crs_train_maes[:,0]

            elif out_bnk == 'both' :
                crs_val_maes = (crs_val_maes[:,0] + crs_val_maes[:,1]) / 2
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


            for i in range(crs_val_maes.shape[0]) :
                writer.add_scalar('cross_val/Val_Reach_MAEs', crs_val_maes[i], i+1)

            counter = 1
            for i in range(crs_train_maes.shape[0]) :
                writer.add_scalar('cross_val/Train_MAEs', crs_train_maes[i], counter)
                counter += get_train_mae

            crs_hparam_logs = {'cross_val/crs_train_loss':crs_train_ls[-1],'cross_val/crs_val_loss':crs_val_ls[-1],
                        'cross_val/crs_train_MAE':crs_train_maes[-1],'cross_val/crs_val_MAE':crs_val_maes[-1],
                        'cross_val/crs_test_MAE':crs_test_mae}

            writer.add_hparams(hparam_def, crs_hparam_logs)
            writer.close()

            trial.report(crs_val_maes[-1], ((j+1)*num_epochs))

            if trial.should_prune():
                raise optuna.TrialPruned()


        return crs_val_maes[-1]


    #study = optuna.create_study(study_name='batch_norm',storage='sqlite:///data\\sqdb\\lin_both_fls_man.db',load_if_exists=True,direction='minimize',sampler=RandomSampler(),
    #        pruner=HyperbandPruner(min_resource=1, max_resource=int(super_epochs*num_epochs), reduction_factor=3))

    study = optuna.create_study(study_name='batch_norm',storage='sqlite:///data\\sqdb\\lin_imgs_both_fls_man.db',
            load_if_exists=True,direction='minimize',pruner=NopPruner())

    study.optimize(objtv,n_trials=1)
    #study.optimize(objtv)
    pass