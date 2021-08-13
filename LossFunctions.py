import torch.nn.functional as F
import torch

def calculate_loss(pred_left, pred_right, pred_binl, pred_binr, out_flatten, bin_out_flatten_left,
                   bin_out_flatten_right, flag_use_lines, flag_bin_out, loss_func, right_loss_weight):
    # print(out_flatten[:,0:1].shape)

    if loss_func == 'l1_loss':
        if flag_bin_out:
            loss = F.l1_loss(pred_left, out_flatten[:, 0:1], reduction='mean') + \
                   F.l1_loss(pred_right, out_flatten[:, 1:2], reduction='mean') + \
                   F.binary_cross_entropy(pred_binl, bin_out_flatten_left, reduction='mean') + \
                   F.binary_cross_entropy(pred_binr, bin_out_flatten_right, reduction='mean')
        else:
            loss = ((1 - right_loss_weight) * F.l1_loss(pred_left, out_flatten[:, 0:1], reduction='mean')) + \
                   (right_loss_weight * F.l1_loss(pred_right, out_flatten[:, 1:2], reduction='mean'))
    elif loss_func == 'mse_loss':
        if flag_bin_out:
            loss = F.mse_loss(pred_left, out_flatten[:, 0:1], reduction='mean') + \
                   F.mse_loss(pred_right, out_flatten[:, 1:2], reduction='mean') + \
                   F.binary_cross_entropy(pred_binl, bin_out_flatten_left, reduction='mean') + \
                   F.binary_cross_entropy(pred_binr, bin_out_flatten_right, reduction='mean')
        else:
            loss = ((1 - right_loss_weight) * F.mse_loss(pred_left, out_flatten[:, 0:1], reduction='mean')) + \
                   (right_loss_weight * F.mse_loss(pred_right, out_flatten[:, 1:2], reduction='mean'))
    elif loss_func == 'huber_loss':
        if flag_bin_out:
            loss = F.smooth_l1_loss(pred_left, out_flatten[:, 0:1], reduction='mean') + \
                   F.smooth_l1_loss(pred_right, out_flatten[:, 1:2], reduction='mean') + \
                   F.binary_cross_entropy(pred_binl, bin_out_flatten_left, reduction='mean') + \
                   F.binary_cross_entropy(pred_binr, bin_out_flatten_right, reduction='mean')
        else:
            loss = ((1 - right_loss_weight) * F.smooth_l1_loss(pred_left, out_flatten[:, 0:1], reduction='mean')) + \
                   (right_loss_weight * F.smooth_l1_loss(pred_right, out_flatten[:, 1:2], reduction='mean'))
    elif loss_func == 'log_cosh':
        def log_cosh(pred, ground_t):
            return torch.mean(torch.log(torch.cosh((pred - ground_t) + 1e-12)))

        if flag_bin_out:
            loss = log_cosh(pred_left, out_flatten[:, 0:1]) + \
                   log_cosh(pred_right, out_flatten[:, 1:2]) + \
                   F.binary_cross_entropy(pred_binl, bin_out_flatten_left, reduction='mean') + \
                   F.binary_cross_entropy(pred_binr, bin_out_flatten_right, reduction='mean')
        else:
            loss = ((1 - right_loss_weight) * log_cosh(pred_left, out_flatten[:, 0:1])) + \
                   (right_loss_weight * log_cosh(pred_right, out_flatten[:, 1:2]))

    """ else :
        if loss_func == 'l1_loss' :
            loss = F.l1_loss(pred, out_flatten,reduction='mean')
        elif loss_func == 'mse_loss' :
            loss = F.mse_loss(pred, out_flatten,reduction='mean')
        elif loss_func == 'huber_loss' :
            loss = F.smooth_l1_loss(pred, out_flatten,reduction='mean') """
    # print(asd)
    return loss


