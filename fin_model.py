import tensorflow as tf
import torch.nn as nn
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from os import listdir
from os.path import isfile, join
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from models import CNN_Model
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import itertools
from scipy.signal import savgol_filter
import optuna


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

def regress_erro(act_err_bin, act_reg, pred_reg, iter_num, side,writer, val_img_ids,epoch):
    temp_arr = pred_reg - act_reg
    if side == 'left' :
        temp_arr = temp_arr
    elif side == 'right' :
        temp_arr = -temp_arr
    counter_pos = 0
    counter_neg = 0 
    pos_deviation = 0
    neg_deviation = 0

    pos_list = []
    neg_list = []

    for i in range(act_err_bin.shape[0]):
        if act_err_bin[i] == 1 and temp_arr[i]>=0 :
            pos_list.append(temp_arr[i])
            pos_deviation = pos_deviation + temp_arr[i]
            counter_pos += 1
        elif act_err_bin[i] == 1 and temp_arr[i]<0 :
            neg_list.append(-temp_arr[i])
            neg_deviation = neg_deviation + (-temp_arr[i])
            counter_neg += 1

    pos_std = np.std(np.asarray(pos_list))
    neg_std = np.std(np.asarray(neg_list))

    pos_max = np.amax(np.asarray(pos_list))
    neg_max = np.amax(np.asarray(neg_list))

    mean_pos_dev = pos_deviation/counter_pos
    mean_neg_dev = neg_deviation/counter_neg

    writer.add_scalar('mean_abs_pos_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), mean_pos_dev, epoch+1)
    writer.add_scalar('mean_abs_neg_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), mean_neg_dev, epoch+1)

    writer.add_scalar('std_of_pos_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), pos_std, epoch+1)
    writer.add_scalar('std_of_neg_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), neg_std, epoch+1)

    writer.add_scalar('max_pos_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), pos_max, epoch+1)
    writer.add_scalar('max_neg_error_for_actual_'+side+'_erosion'+str(val_img_ids[iter_num]), neg_max, epoch+1)

    return mean_pos_dev, pos_std, mean_neg_dev, neg_std

def calc_fscore(iter_num, actual_list, prev_actual_list, pred_list, epoch,writer,val_img_ids):
    act_left = actual_list[:,iter_num,0]
    act_right = actual_list[:,iter_num,1]

    prev_left = prev_actual_list[:, iter_num, 0]
    prev_right = prev_actual_list[:, iter_num, 1]

    pred_left = pred_list[:, iter_num, 0]
    pred_right = pred_list[:, iter_num, 1]

    actual_ers_lft = np.reshape(np.where(act_left<prev_left, 1, 0),(act_left.shape[0],1))
    actual_ers_rht = np.reshape(np.where(act_right>prev_right, 1, 0),(act_right.shape[0],1))
    
    left_mae_pos,left_std_pos,left_mae_neg,left_std_neg = regress_erro(actual_ers_lft, act_left, pred_left, iter_num, 'left',writer,val_img_ids,epoch)
    right_mae_pos,right_std_pos,right_mae_neg,right_std_neg = regress_erro(actual_ers_rht, act_right, pred_right, iter_num, 'right',writer,val_img_ids,epoch)

    avg_mae_pos = (left_mae_pos + right_mae_pos)/2
    writer.add_scalar('pos_mae_for_actual_erosion'+str(val_img_ids[iter_num]), avg_mae_pos, epoch+1)
    avg_std_pos = (left_std_pos + right_std_pos)/2
    writer.add_scalar('pos_std_for_actual_erosion'+str(val_img_ids[iter_num]), avg_std_pos, epoch+1)

    avg_mae_neg = (left_mae_neg + right_mae_neg)/2
    writer.add_scalar('neg_mae_for_actual_erosion'+str(val_img_ids[iter_num]), avg_mae_neg, epoch+1)
    avg_std_neg = (left_std_neg + right_std_neg)/2
    writer.add_scalar('neg_std_for_actual_erosion'+str(val_img_ids[iter_num]), avg_std_neg, epoch+1)

    pred_ers_lft = np.reshape(np.where(pred_left<prev_left, 1, 0),(pred_left.shape[0],1))
    pred_ers_rht = np.reshape(np.where(pred_right>prev_right, 1, 0),(pred_right.shape[0],1))

    """ conf_mat_lft = confusion_matrix(actual_ers_lft, pred_ers_lft)
    conf_mat_rht = confusion_matrix(actual_ers_rht, pred_ers_rht)
    combined_conf = conf_mat_lft + conf_mat_rht """

    y_true = np.concatenate((actual_ers_lft,actual_ers_rht), axis = 0)
    y_pred = np.concatenate((pred_ers_lft,pred_ers_rht), axis = 0)

    precision_comb = precision_score(y_true, y_pred, average='binary')
    recall_comb = recall_score(y_true, y_pred, average='binary')
    f1_comb = f1_score(y_true, y_pred, average='binary')
    
    writer.add_scalar('precision_'+ str(val_img_ids[iter_num]), precision_comb, epoch+1)
    writer.add_scalar('recall_'+ str(val_img_ids[iter_num]), recall_comb, epoch+1)
    writer.add_scalar('f1_score_'+ str(val_img_ids[iter_num]), f1_comb, epoch+1)

    return avg_mae_pos, avg_std_pos, avg_mae_neg, avg_std_neg, precision_comb, recall_comb, f1_comb



def wrt_img(iter_num, actual_list, prev_actual_list, pred_list, val_img_ids,writer):
    
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

    denoising = True
    window = 99
    poly = 2

    if denoising :
        pred_left_den = savgol_filter(pred_list[:,iter_num,0], window, poly)
        pred_right_den = savgol_filter(pred_list[:,iter_num,1], window, poly)

    img = cv2.imread(os.path.join('./data/img/up_rgb/'+str(val_img_ids[iter_num])+'.png'), 1)
    for i in range(num_rows):

        img[i,int(actual_list[i,iter_num,0]),:] = [255,255,255]
        img[i,int(actual_list[i,iter_num,1]),:] = [255,255,255]

        if 0<=int(pred_list[i,iter_num,0])<=745 :
            pass
        else :
            pred_list[i,iter_num,0] = 0
        if 0<=int(pred_left_den[i])<=745 :
            pass
        else :
            pred_left_den[i] = 0 
        
        if 0<=int(pred_list[i,iter_num,1])<=745 :
            pass
        else :
            pred_list[i,iter_num,1] = 744
        if 0<=int(pred_right_den[i])<=745 : 
            pass
        else :
            pred_right_den[i] = 0


        if actual_ers_lft[i] == 1 :
            img[i,int(prev_actual_list[i,iter_num,0]),:] = [255,0,0]
            img[i,int(pred_list[i,iter_num,0]),:] = [0,255,0]
            if denoising:
                img[i,int(pred_left_den[i]),:] = [0,0,255]
            

        if actual_ers_rht[i] == 1 :
            img[i,int(prev_actual_list[i,iter_num,1]),:] = [255,0,0]
            img[i,int(pred_list[i,iter_num,1]),:] = [0,255,0]
            if denoising:
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
    model_path = os.path.join('./data/model/'+model_name.split("\\")[-1]+'.pt')
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, model_path)

def log_performance_metrics(pred_list,actual_list,prev_actual_list,num_val_img, epoch, msk_mean, msk_std, val_img_ids,writer):
    print('logging performance metrics........')
    
    for i in range(num_val_img):
        pos_mae, pos_std,neg_mae,neg_std, precision_comb,recall_comb,f1_comb = calc_fscore(i, actual_list, prev_actual_list, pred_list, epoch,writer,val_img_ids)
        if i == 0 :
            test_pos_mae = pos_mae
            test_pos_std = pos_std
            test_neg_mae = neg_mae
            test_neg_std = neg_std
            final_prec = precision_comb
            final_recall = recall_comb
            final_f1 = f1_comb
        else :
            test_pos_mae = test_pos_mae + pos_mae
            test_pos_std = test_pos_std + pos_std
            test_neg_mae = test_neg_mae + neg_mae
            test_neg_std = test_neg_std + neg_std
            final_prec = final_prec + precision_comb
            final_recall = final_recall + recall_comb
            final_f1 = final_f1 + f1_comb

    test_pos_mae = test_pos_mae/num_val_img
    test_pos_std = test_pos_std/num_val_img
    test_neg_mae = test_neg_mae/num_val_img
    test_neg_std = test_neg_std/num_val_img
    test_prec = final_prec/num_val_img
    test_recall = final_recall/num_val_img
    test_f1_score = final_f1/num_val_img

    writer.add_scalar("test_set_pos_mae", test_pos_mae, epoch+1)
    writer.add_scalar("test_set_pos_std", test_pos_std, epoch+1)
    writer.add_scalar("test_set_neg_mae", test_neg_mae, epoch+1)
    writer.add_scalar("test_set_neg_std", test_neg_std, epoch+1)
    writer.add_scalar("test_set_precision", test_prec, epoch+1)
    writer.add_scalar("test_set_recall", test_recall, epoch+1)
    writer.add_scalar("test_set_f1_score", test_f1_score, epoch+1)

    return test_pos_mae, test_pos_std,test_neg_mae,test_neg_std, test_prec, test_recall, test_f1_score


def objective(trial):
    load_mod = False
    save_mod = False
    total_window = 52
    num_lstm_layers = 1
    num_channels = 7
    batch_size = 100
    EPOCHS = 52
    lr_rate = trial.suggest_loguniform('lr_rate', .0000001, 1)                       #.0001
    vertical_image_window = 1
    model_type = 'CNN_Model_vanilla'
    drop_rate = 0
    time_step = 5
    val_batch_size = batch_size
    total_time_step = 33
    log_performance = 5 ###number of epochs after which performance metrics are calculated
    model_save_at = 50     ###number of epochs after which to save model
    early_stop_thresh = 30
    val_img_ids = [201701, 201801, 201901, 202001]
    num_val_img = len(val_img_ids)
    data_div_step = total_time_step - num_val_img
    log_hist = 5
    writer = SummaryWriter()
    model_name = writer.log_dir

    hyperparameter_defaults = dict(
        dropout = str(drop_rate),
        num_channels = num_channels,
        batch_size = batch_size,
        learning_rate = lr_rate,
        time_step = time_step,
        num_lstm_layers = num_lstm_layers,
        total_window = total_window,
        dataset='7_chann',
        model_type=model_type,
        vertical_image_window = vertical_image_window
        )

    dataset_f = tf.data.TFRecordDataset(os.path.join('./data/tfrecord/comp_tf.tfrecords'))
    dataset_f = dataset_f.window(size=data_div_step, shift=total_time_step, stride=1, drop_remainder=False)
    dataset_f = dataset_f.map(lambda x: x.window(size=time_step, shift=1, stride=1,drop_remainder=True))
    dataset_f = dataset_f.flat_map(lambda x2: x2.flat_map(lambda x1: x1))
    dataset_f = dataset_f.map(_parse_function_).batch(time_step)
    dataset_f = dataset_f.shuffle(10000)
    dataset_f = dataset_f.batch(batch_size, drop_remainder=True)

    dataseti1 = tf.data.TFRecordDataset(os.path.join('./data/tfrecord/comp_tf.tfrecords'))
    dataseti1 = dataseti1.window(size=total_time_step, shift=total_time_step, stride=1, drop_remainder=False)
    dataseti1 = dataseti1.map(lambda x: x.skip(total_time_step-(time_step+(num_val_img-1))).window(size=time_step, shift=1, stride=1,drop_remainder=True))
    dataseti1 = dataseti1.flat_map(lambda x2: x2.flat_map(lambda x1: x1))
    dataseti1 = dataseti1.map(_parse_function_).batch(time_step)
    dataseti1 = dataseti1.batch(val_batch_size, drop_remainder=True)

    model = CNN_Model(num_channels, batch_size, val_batch_size,time_step, num_lstm_layers, drop_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

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

            input_tensor = np.reshape(input_tensor, (batch_size,time_step,745,num_channels))
            reg_coor = np.reshape(reg_coor, (batch_size,time_step,2))

            input_tensor = input_tensor[:,0:time_step-1,:,:]
            reg_coor = reg_coor[:,time_step-1:time_step,:]
            
            input_tensor = torch.Tensor(input_tensor).cuda()
            reg_coor = torch.Tensor(reg_coor).cuda()
            reg_coor = torch.reshape(reg_coor, (batch_size,-1))
            
            optimizer.zero_grad()
            pred = model(input_tensor)

            loss = F.mse_loss(pred, reg_coor,reduction='mean')
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss+loss
            counter += 1
            global_train_counter += 1

        avg_epoch_loss = epoch_loss / counter
        template = 'Epoch {}, Train Loss: {}'
        print(template.format(epoch+1,avg_epoch_loss))

        writer.add_scalar('Loss/train', avg_epoch_loss, epoch+1)

        if epoch == 0 :
            writer.add_graph(model, input_tensor)

        if epoch % log_hist == log_hist-1:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram('parameters_'+str(name), param.data, epoch + 1)
                    writer.add_histogram('gradients'+str(name), param.grad, epoch + 1)

        if save_mod == True :
            if epoch % model_save_at == 0:
                model_save(model, optimizer, model_name)
            
        model.eval()

        val_epoch_loss = 0
        counter_val = 0
        
        with torch.no_grad():
            
            if epoch % log_performance == log_performance-1:
                pred_list = []
                actual_list = []
                prev_actual_list = []

            for input_tensor, reg_coor, _ , year_id in dataseti1:

                input_tensor = np.reshape(input_tensor, (batch_size,time_step,745,num_channels))
                reg_coor = np.reshape(reg_coor, (batch_size,time_step,2))

                input_tensor = input_tensor[:,0:time_step-1,:,:]
                prev_time_step = reg_coor[:,time_step-2:time_step-1,:]
                reg_coor = reg_coor[:,time_step-1:time_step,:]
                
                input_tensor = torch.Tensor(input_tensor).cuda()
                reg_coor = torch.Tensor(reg_coor).cuda()
                reg_coor = torch.reshape(reg_coor, (batch_size,-1))

                pred = model(input_tensor)
                
                loss = F.mse_loss(pred, reg_coor,reduction='mean')

                val_epoch_loss = val_epoch_loss+loss
                counter_val += 1
                
                if epoch % log_performance == log_performance-1:
                    prev_actual_list.append(prev_time_step)
                    reg_coor = reg_coor.cpu()
                    actual_list.append(reg_coor.numpy())
                    pred_np = pred.cpu()
                    pred_list.append(pred_np.numpy())

            avg_val_epoch_loss = val_epoch_loss / counter_val
            template = 'Epoch {}, Val_Loss: {}'
            print(template.format(epoch+1,avg_val_epoch_loss))

            writer.add_scalar('Loss/Val', avg_val_epoch_loss, epoch+1)

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
                test_pos_mae, test_pos_std,test_neg_mae,test_neg_std, test_prec, test_recall, test_f1_score = log_performance_metrics(pred_list,actual_list,prev_actual_list,
                                                    num_val_img, epoch, msk_mean, msk_std, val_img_ids,writer)
                
        if early_stop_counter > early_stop_thresh :
            print('early stopping as val loss is not improving ........')
            model_save(model, optimizer, model_name)
            break

    for iter_num in range(num_val_img):
        temp_conf = wrt_img(iter_num, actual_list, prev_actual_list, pred_list, val_img_ids, writer)

        if iter_num == 0 :
            final_conf = temp_conf
        else :
            final_conf = final_conf + temp_conf
        
    plt_conf_mat(final_conf, 'total_test_confusion_matrix', writer)
    hyperparameter_defaults.update('epochs'=epoch+1)
    writer.add_hparams(hyperparameter_defaults,{'hparam/train_loss':avg_epoch_loss,'hparam/val_loss':avg_val_epoch_loss,
        'hparam/pos_mae':test_pos_mae, 'hparam/precision':test_prec,'hparam/recall':test_recall,'hparam/f1_score':test_f1_score,
        'hparam/pos_std':test_pos_std, 'hparam/neg_mae':test_neg_mae,'hparam/neg_std':test_neg_std})
    writer.close()
    model_save(model, optimizer, model_name)

    return avg_val_epoch_loss


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=None) 
    pass