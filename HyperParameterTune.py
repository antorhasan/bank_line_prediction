import numpy as np
np.seterr(all='raise')
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.samplers import TPESampler,RandomSampler
from optuna.pruners import HyperbandPruner,NopPruner

super_epochs = 9
num_epochs = 4


def objtv(trial):
    dataset_dic = get_all_data()
    # print(asd)

    super_epochs = 9
    num_epochs = 4

    trail_id = trial.number
    load_models_list = []
    transform_constants_list = []
    # tm_stp=trial.suggest_int('time_step', 3, 6, 1)
    tm_stp = 5
    # lr_pow = trial.suggest_discrete_uniform('learning_rate', -5.0, -3.0, 0.5)
    lr_pow = -4.5
    # lstm_hidden_units = trial.suggest_int('neurons_per_layer', 200, 500, 50 )
    lstm_hidden_units = 50
    # batch_size_pow = trial.suggest_int('batch_size_power', 2, 6 , 1)
    batch_size_pow = 2
    # num_layers = trial.suggest_int('num_of_layers', 3, 5, 1)
    num_layers = 0
    num_cnn_layers = 6
    # strt = trial.suggest_int('starting_year', 0, 20, 5)
    strt = 0
    # vert_hgt = trial.suggest_int('vertical_window_size', 128, 256, 128)
    vert_hgt = 128
    # loss_func = trial.suggest_categorical('loss_function', ['mse_loss', 'l1_loss', 'huber_loss','log_cosh])
    loss_func = 'huber_loss'
    # output_subtracted = trial.suggest_categorical('output_subtracted', [0,False])
    # lstm_layers = trial.suggest_int('lstm_depth_layers', 1, 3, 1)
    lstm_layers = 1
    # model_type = trial.suggest_categorical('model_type', ['ANN', 'LSTM'])
    model_type = 'CNN_LSTM'
    # flag_batch_norm_bin = trial.suggest_int('batch_norm', 0, 1, 1)
    # flag_batch_norm_bin = 0
    flag_dilated_cov = False
    flag_use_lines = True
    flag_use_imgs = True
    flag_bin_out = False
    output_subtracted = False
    lstm_dropout = 0.0
    # pooling_layer = trial.suggest_categorical('pooling_layer', ['MaxPool', 'AvgPool'])
    pooling_layer = 'AvgPool'
    # only_lstm_units = trial.suggest_int('only_lstm_units', 200, 500, 50 )
    only_lstm_units = 150
    # num_branch_layers = trial.suggest_int('num_branch_layers', 2, 10, 2)
    num_branch_layers = 1
    num_lft_brn_lyrs = 0
    num_rgt_brn_lyrs = 0
    # branch_layer_neurons = trial.suggest_int('branch_layer_neurons', 50, 150, 50)
    branch_layer_neurons = 100
    # right_loss_weight = trial.suggest_discrete_uniform('right_loss_weight', 0.5, 0.95, 0.05)
    right_loss_weight = 0.4
    # num_filter_choice = trial.suggest_int('num_filter_choice', 0, 1, 1)
    # num_filter_choice = 2
    # num_filter_list = [4, 8, 16, 32]
    strtn_num_chanls = 16
    model_optim = 'Adam'
    ad_pow = 1 * (10 ** -3.0)
    # ad_pow = 0
    erosion_thresh = 1
    # temp_model_list = ['Nov27_01-50-46_DESKTOP-8SUO90F','Nov27_03-09-25_DESKTOP-8SUO90F',
    #    'Nov27_04-26-01_DESKTOP-8SUO90F','Nov27_05-34-13_DESKTOP-8SUO90F','Nov27_06-38-34_DESKTOP-8SUO90F']

    for j in range(super_epochs):

        cross_val_nums = 1
        val_split_org = 0
        val_skip = 0
        out_use_mid = True
        # strt=20
        batch_size = 2 ** batch_size_pow
        change_start = False
        # batch_size = 2**batch_size_pow
        get_train_mae = num_epochs
        # lr_pow=-3.0
        # ad_pow=1*(10**-1.0)

        # vert_hgt=1
        vert_step_num = 1
        # num_epochs=num_epochs
        # lstm_layers=1
        # neurons_per_layer_list = [20,50,70,]
        inp_bnk = 'img'
        out_bnk = 'both'

        # loss_func='mse_loss'

        train_shuffle = True
        train_val_gap = False
        # flg_btch_list = [False, True]
        # flag_batch_norm = flg_btch_list[int(flag_batch_norm_bin)]
        flag_batch_norm = True

        # model_type = 'ANN'
        # num_layers_list = [1,3,5,7,9,12,14]

        crs_train_ls = []
        crs_val_ls = []
        crs_train_maes = []
        crs_val_maes = []
        crs_test_maes = []

        for i in range(cross_val_nums):
            val_split = val_split_org + (tm_stp - 2)

            if j == 0:
                model_name, train_losses, val_losses, train_maes, val_maes, hparam_def, transform_constants = objective(
                    tm_stp=tm_stp, strt=strt, lr_pow=lr_pow, ad_pow=ad_pow, vert_hgt=vert_hgt,
                    vert_step_num=vert_step_num, num_epochs=num_epochs, train_shuffle=train_shuffle,
                    get_train_mae=get_train_mae, transform_constants=None, lstm_layers=lstm_layers,
                    lstm_hidden_units=lstm_hidden_units, batch_size=batch_size, inp_bnk=inp_bnk, out_bnk=out_bnk,
                    val_split=val_split, val_skip=val_skip, model_type=model_type, num_layers=num_layers,
                    model_optim=model_optim, loss_func=loss_func, save_mod=True, load_mod=False, load_file=None,
                    skip_training=False, output_subtracted=output_subtracted, train_val_gap=train_val_gap,
                    out_use_mid=out_use_mid, trail_id=trail_id, flag_batch_norm=flag_batch_norm,
                    dataset_dic=dataset_dic,
                    num_cnn_layers=num_cnn_layers, flag_use_lines=flag_use_lines, pooling_layer=pooling_layer,
                    flag_bin_out=flag_bin_out, only_lstm_units=only_lstm_units, num_branch_layers=num_branch_layers,
                    branch_layer_neurons=branch_layer_neurons, right_loss_weight=right_loss_weight,
                    strtn_num_chanls=strtn_num_chanls, flag_use_imgs=flag_use_imgs, erosion_thresh=erosion_thresh,
                    num_lft_brn_lyrs=num_lft_brn_lyrs, num_rgt_brn_lyrs=num_rgt_brn_lyrs, lstm_dropout=lstm_dropout,
                    flag_dilated_cov=flag_dilated_cov)
            elif j > 0:
                model_name, train_losses, val_losses, train_maes, val_maes, hparam_def, transform_constants = objective(
                    tm_stp=tm_stp, strt=strt, lr_pow=lr_pow, ad_pow=ad_pow, vert_hgt=vert_hgt,
                    vert_step_num=vert_step_num, num_epochs=num_epochs, train_shuffle=train_shuffle,
                    get_train_mae=get_train_mae, transform_constants=transform_constants_list[i],
                    lstm_layers=lstm_layers, lstm_hidden_units=lstm_hidden_units, batch_size=batch_size,
                    inp_bnk=inp_bnk, out_bnk=out_bnk, val_split=val_split, val_skip=val_skip, model_type=model_type,
                    num_layers=num_layers,
                    model_optim=model_optim, loss_func=loss_func, save_mod=True, load_mod=True,
                    load_file=load_models_list[0], skip_training=False, output_subtracted=output_subtracted,
                    train_val_gap=train_val_gap, out_use_mid=out_use_mid, trail_id=trail_id,
                    flag_batch_norm=flag_batch_norm, dataset_dic=dataset_dic,
                    num_cnn_layers=num_cnn_layers, flag_use_lines=flag_use_lines, pooling_layer=pooling_layer,
                    flag_bin_out=flag_bin_out, only_lstm_units=only_lstm_units, num_branch_layers=num_branch_layers,
                    branch_layer_neurons=branch_layer_neurons, right_loss_weight=right_loss_weight,
                    strtn_num_chanls=strtn_num_chanls, flag_use_imgs=flag_use_imgs, erosion_thresh=erosion_thresh,
                    num_lft_brn_lyrs=num_lft_brn_lyrs, num_rgt_brn_lyrs=num_rgt_brn_lyrs, lstm_dropout=lstm_dropout,
                    flag_dilated_cov=flag_dilated_cov)

                load_models_list.pop(0)

            load_models_list.append(model_name)
            if j == 0:
                transform_constants_list.append(transform_constants)
            # print(val_losses)
            crs_train_ls.append(train_losses)
            crs_val_ls.append(val_losses)
            crs_train_maes.append(train_maes)
            crs_val_maes.append(val_maes)

            if val_skip > 0:
                _, _, _, _, test_val_maes, _, _ = objective(tm_stp=tm_stp, strt=strt, lr_pow=lr_pow, ad_pow=ad_pow,
                                                            vert_hgt=vert_hgt, vert_step_num=vert_step_num,
                                                            num_epochs=1, train_shuffle=train_shuffle, get_train_mae=1,
                                                            transform_constants=transform_constants,
                                                            lstm_layers=lstm_layers,
                                                            lstm_hidden_units=lstm_hidden_units, batch_size=batch_size,
                                                            inp_bnk=inp_bnk, out_bnk=out_bnk,
                                                            val_split=val_split - (val_split_org - val_skip),
                                                            val_skip=(val_skip - 1), model_type=model_type,
                                                            num_layers=num_layers,
                                                            model_optim=model_optim, loss_func=loss_func,
                                                            save_mod=False, load_mod=True, load_file=model_name,
                                                            skip_training=True, output_subtracted=output_subtracted,
                                                            train_val_gap=train_val_gap, out_use_mid=out_use_mid,
                                                            trail_id=trail_id, flag_batch_norm=flag_batch_norm,
                                                            dataset_dic=dataset_dic, num_cnn_layers=num_cnn_layers,
                                                            flag_use_lines=flag_use_lines, pooling_layer=pooling_layer,
                                                            flag_bin_out=flag_bin_out, only_lstm_units=only_lstm_units,
                                                            num_branch_layers=num_branch_layers,
                                                            branch_layer_neurons=branch_layer_neurons,
                                                            right_loss_weight=right_loss_weight,
                                                            strtn_num_chanls=strtn_num_chanls,
                                                            flag_use_imgs=flag_use_imgs, erosion_thresh=erosion_thresh,
                                                            num_lft_brn_lyrs=num_lft_brn_lyrs,
                                                            num_rgt_brn_lyrs=num_rgt_brn_lyrs,
                                                            lstm_dropout=lstm_dropout,
                                                            flag_dilated_cov=flag_dilated_cov)

                crs_test_maes.append(test_val_maes)

                val_split_org = val_split_org + 1
                val_skip = val_split_org - 1
                if change_start == True:
                    strt = strt - 1

            # print(asd)
            """ print(crs_train_ls)
            print(crs_val_ls)
            print(crs_train_maes)
            print(crs_val_maes) """

        crs_train_ls = np.mean(np.asarray(crs_train_ls), axis=0)
        crs_val_ls = np.mean(np.asarray(crs_val_ls), axis=0)
        crs_train_maes = np.mean(np.asarray(crs_train_maes), axis=0)
        crs_val_maes = np.mean(np.asarray(crs_val_maes), axis=0)
        if val_skip > 0:
            crs_test_mae = np.mean(np.asarray(crs_test_maes), axis=0)

        # print(crs_val_ls)

        writer = SummaryWriter()

        for i in range(crs_train_ls.shape[0]):
            writer.add_scalar('cross_val/train', crs_train_ls[i], i + 1)

        for i in range(crs_val_ls.shape[0]):
            writer.add_scalar('cross_val/val', crs_val_ls[i], (i + 1) * get_train_mae)

        if out_bnk == 'right':
            crs_val_maes = crs_val_maes[:, 1]
            if val_skip > 0:
                crs_test_mae = crs_test_mae[:, 1]
            crs_train_maes = crs_train_maes[:, 1]

        elif out_bnk == 'left':
            crs_val_maes = crs_val_maes[:, 0]
            if val_skip > 0:
                crs_test_mae = crs_test_mae[:, 0]
            crs_train_maes = crs_train_maes[:, 0]

        elif out_bnk == 'both':
            crs_val_maes = (crs_val_maes[:, 0] + crs_val_maes[:, 1]) / 2
            if val_skip > 0:
                crs_test_mae = (crs_test_mae[:, 0] + crs_test_mae[:, 1]) / 2
            crs_train_maes = (crs_train_maes[:, 0] + crs_train_maes[:, 1]) / 2

        """ print(crs_train_ls)
        print(crs_train_ls[-1])
        print(crs_val_ls)
        print(crs_val_ls[-1])

        print(crs_train_maes)
        print(crs_train_maes[-1])
        print(crs_val_maes)
        print(crs_val_maes[-1]) """

        for i in range(crs_val_maes.shape[0]):
            writer.add_scalar('cross_val/Val_Reach_MAEs', crs_val_maes[i], i + 1)

        counter = 1
        for i in range(crs_train_maes.shape[0]):
            writer.add_scalar('cross_val/Train_MAEs', crs_train_maes[i], counter)
            counter += get_train_mae

        if val_skip <= 0:
            crs_test_mae = -1

        crs_hparam_logs = {'cross_val/crs_train_loss': crs_train_ls[-1], 'cross_val/crs_val_loss': crs_val_ls[-1],
                           'cross_val/crs_train_MAE': crs_train_maes[-1], 'cross_val/crs_val_MAE': crs_val_maes[-1],
                           'cross_val/crs_test_MAE': crs_test_mae}

        writer.add_hparams(hparam_def, crs_hparam_logs)
        writer.close()

        trial.report(crs_val_maes[-1], ((j + 1) * num_epochs))

        if trial.should_prune():
            raise optuna.TrialPruned()

    return crs_val_maes[-1]


# study = optuna.create_study(study_name='batch_norm',storage='sqlite:///data\\sqdb\\lin_both_fls_man.db',load_if_exists=True,direction='minimize',sampler=RandomSampler(),
#        pruner=HyperbandPruner(min_resource=1, max_resource=int(super_epochs*num_epochs), reduction_factor=3))

study = optuna.create_study(study_name='batch_norm', storage='sqlite:///data\\sqdb\\lin_imgs_both_fls_man_2021.db',
                            load_if_exists=True, direction='minimize', pruner=NopPruner())

study.optimize(objtv, n_trials=1)
# study.optimize(objtv)
pass