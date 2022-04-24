'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import random
import numpy as np
import prettytable as pt
import matplotlib.pyplot as plt

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar_local(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def print_banner(s):
    print("=*="*20 + f" {s} " + "=*="*20)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_weight_group_info(WEIGHT_GROUP_TABLE = None, weight_group = None):
    tb = pt.PrettyTable()
    if 'oral' in  weight_group.get_keys():
        tb.field_names = ["weight pool name", "size/1e4"]
        for key in weight_group.get_keys():
            w = weight_group.get(key)
            tb.add_row([key, w.shape[0] / 10000])
    else:
        tb.field_names = ["weight pool name", "size/1e4", "min/mean/max", "initial_fn", "initial_kwargs"]
        for key in weight_group.get_keys():
            w = weight_group.get(key)
            tb.add_row([key, w.shape[0] / 10000, f'{w.min().item():.4f}/{w.mean().item():.4f}/{w.max().item():.4f}',\
                WEIGHT_GROUP_TABLE[key][1].__name__, WEIGHT_GROUP_TABLE[key][2]])
    print(tb)

def print_task_model_instances_info(task_model_instances = None):
    print('Task_Model_instances are:')
    tb = pt.PrettyTable()
    for key in task_model_instances:
        if 'pmnist' in key:
            tb.field_names = ["Task Model", "cur_test_acc", "batch_size", "#train_batch", "optimizer", "ori_lr", "T_max", "train_idx", "#model params", "perm_key", "rand_seed_for_x", "rand_seed_for_y"]
        else:
            tb.field_names = ["Task Model", "cur_test_acc", "batch_size", "#train_batch", "optimizer", "ori_lr", "T_max", "train_idx", "#model params", "perm_key", "wp start pos"]
    for key in task_model_instances:
        TM = task_model_instances[key]
        if 'pmnist' in key:
            tb.add_row([key, TM.test_acc, TM.batch_size, TM.train_batch_num, TM.optimizer_name, TM.origin_lr, TM.T_max, TM.train_idx, TM.param_num, TM.perm_key, TM.model.rand_seed_for_x, TM.model.rand_seed_for_y]) 
        else:
            # print([key, TM.test_acc, TM.batch_size, TM.train_batch_num, TM.optimizer_name, TM.origin_lr, TM.T_max, TM.train_idx, TM.param_num, TM.perm_key, TM.wp_start_pos])
            tb.add_row([key, TM.test_acc, TM.batch_size, TM.train_batch_num, TM.optimizer_name, TM.origin_lr, TM.T_max, TM.train_idx, TM.param_num, TM.perm_key, TM.wp_start_pos]) 
    print(tb)

## TODO: Add time stamp to trial if needed
# import datetime
# now = datetime.datetime.now()
# print(now.year, now.month, now.day, now.hour, now.minute, now.second)
def get_trial_name(WEIGHT_GROUP_TABLE = None, task_model_instances = None, args = None):
    if args.save_path is not None:
        return args.save_path 
    info = ''
    if args.rand_seed_for_weight != 0:
        info = info + f'seed_{args.rand_seed_for_weight}/'
        if not os.path.isdir(f'./checkpoint/{info}'):
            os.mkdir(f'./checkpoint/{info}')
    info = info + f'perm_{args.rand_perm}_' if args.rand_perm != 0 else info
    info = info + 'dsp_' if args.different_start_pos is True else info
    info = info + 'pre_' if args.pretrained_wp is True else info
    if args.oral_init is True:
        info = info + 'oral_init_'
    else:
        for key in WEIGHT_GROUP_TABLE:
            info += '{}k_'.format(WEIGHT_GROUP_TABLE[key][0]//1000)
    print_pmnist = True
    for key in task_model_instances:
        if 'onlyG' not in key and 'GAN' not in key and 'pmnist' not in key:
            info += '{}_{}_'.format(key, task_model_instances[key].origin_lr)
        elif 'pmnist' in key and print_pmnist is True: 
            info = info + f'pmnist_{args.pmnist_num}_'
            info = info + 'same_arch' if args.same_arch is True else info
            print_pmnist = False
        elif 'pmnist' in key:
            continue
        else: ## print the number of recovered images
            info += '{}_{}_'.format(key, task_model_instances[key].num_pair*task_model_instances[key].batch_size)
    return info[:-1]

def print_bar_train_info(task_model_instances):
    info = ''
    pmnist_loss = 0
    pmnist_num = 0
    for key in task_model_instances:
        cur_batch_idx_in_epoch = task_model_instances[key].train_idx%task_model_instances[key].test_interval+1
        train_loss = task_model_instances[key].train_loss_for_one_epoch / cur_batch_idx_in_epoch 
        if 'pmnist' in key:
            pmnist_loss += train_loss
            pmnist_num += 1
        else:
            # info += '%s Loss: %.3f <- %d |' % (key, train_loss, task_model_instances[key].train_idx)
            info += '%s Loss: %.3f |' % (key, train_loss)
    if pmnist_num > 0:
        info += 'pmnist Avg. Loss: %.3f |' % (1.0 * pmnist_loss / pmnist_num)
    return info

def plot_wave(waveform,path):
    plt.figure()
    plt.plot(waveform.t().cpu().numpy())
    plt.savefig(path)
    plt.close()