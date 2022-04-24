'''
from TaskModel import *
from models.resnet import *
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from typing import OrderedDict
sns.set()
# SMALL_SIZE = 12
# MEDIUM_SIZE = 16
# BIGGER_SIZE = 16
# TITLE_SIZE = 20

# plt.rc('axes', titlesize=BIGGER_SIZE, labelsize = BIGGER_SIZE+2)     # fontsize of the axes title
# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels\
# plt.rc('xlabel', fontsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ylabel', fontsize=MEDIUM_SIZE)    # fontsize of the tick labels\
# plt.rc('legend', fontsize=MEDIUM_SIZE-2)    # legend fontsize
# plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title\

def get_params(params_origin, param_size, offset, rand_perm = None,  a = None, b = None):
    try: 
        params = params_origin if rand_perm is not None else params_origin#.clone() 
    except RuntimeError: 
        print('RuntimeError:The weight_group is too large (e.g. greater than 25 million).') 
        exit()
    new_params = []
    repete_count = (param_size + offset) // len(params)
    if(repete_count == 0):
        return params[offset:param_size + offset], param_size + offset
    
    new_params.append(params[offset:])
    repete_count -= 1
    for i in range(repete_count):
        if a == None and b == None:
            new_params.append(params)
        else:
            new_params.append(params*a[i]+b[i]) # 
    offset = param_size + offset - (repete_count+1) * len(params)  
    new_params.append(params[:offset])
    return torch.cat(new_params), offset
    
def copy_param_val(model, params, copy_buffer = False, rand_perm = None, wp_start_pos = None, **kwargs):
    offset = wp_start_pos.copy() if 'oral' not in params else dict({k:0 for k in params})
    params_permed = {key: params[key][rand_perm[key]] for key in rand_perm} if rand_perm is not None else params
    for name, module in model.named_modules():
        if not hasattr(module, 'weight') and (not hasattr(module, 'bias')): # or type(module.bias) is bool) and not hasattr(module, 'weight_ih_l0'):
            continue
        weight_key = 'oral' if 'oral' in params.get_keys() else 'normal'
        bias_key = 'oral' if 'oral' in params.get_keys() else 'normal'
        if type(module) in [torch.nn.modules.batchnorm.BatchNorm2d,torch.nn.modules.batchnorm.BatchNorm1d] and 'oral' not in params.get_keys():
            weight_key = 'ones'
            bias_key = 'zeros'
        if hasattr(module, 'weight_ih_l0') and type(module.bias) is bool: ## For RNN
            param_val, offset[weight_key] = get_params(params_permed[weight_key], np.prod(module.weight_ih_l0.shape), offset[weight_key], rand_perm, **kwargs)
            module.weight_ih_l0.data = param_val.reshape(module.weight_ih_l0.shape)
            param_val, offset[weight_key] = get_params(params_permed[weight_key], np.prod(module.weight_hh_l0.shape), offset[weight_key], rand_perm, **kwargs)
            module.weight_hh_l0.data = param_val.reshape(module.weight_hh_l0.shape)
            param_val, offset[weight_key] = get_params(params_permed[weight_key], np.prod(module.bias_ih_l0.shape), offset[weight_key], rand_perm, **kwargs)
            module.bias_ih_l0.data = param_val.reshape(module.bias_ih_l0.shape)
            param_val, offset[weight_key] = get_params(params_permed[weight_key], np.prod(module.bias_hh_l0.shape), offset[weight_key], rand_perm, **kwargs)
            module.bias_hh_l0.data = param_val.reshape(module.bias_hh_l0.shape)
        else:
            param_val, offset[weight_key] = get_params(params_permed[weight_key], np.prod(module.weight.shape), offset[weight_key], rand_perm, **kwargs)
            module.weight.data = param_val.reshape(module.weight.shape)
            if hasattr(module, 'bias') and module.bias is not None: #  and type(module.bias) is not bool:
                param_val, offset[bias_key] = get_params(params_permed[bias_key], np.prod(module.bias.shape), offset[bias_key], rand_perm, **kwargs)
                module.bias.data = param_val.reshape(module.bias.shape)

def get_paths():
    root = '/home/myang_20210409/yyf/model_overloading/result/'
    dsp_pth = root + 'dsp/seed_100_dsp_15000k_4k_4k_cifar10-ResNet18_0.1_cifar10-VGG_0.1_mnist-LeNet_0.01_dermamnist-LeNet_0.001_imdb-TextCNN_0.1_speechcommand-M5_0.1_cifar-onlyG_512/weight_group.pth'
    non_dsp_pth = root + 'dsp/seed_100_1500k_4k_4k_cifar10-ResNet18_0.01_cifar10-VGG_0.01_mnist-LeNet_0.01_dermamnist-LeNet_0.001_imdb-TextCNN_0.1_cifar-onlyG_512/weight_group.pth' 
    oral_resnet_path_list = [root + 'resnet-cifar-with-different-seed/seed_{}_15000k_4k_4k_cifar10-ResNet18_0.1/weight_group.pth'.format(i) for i in range(10)]  
    return [dsp_pth, non_dsp_pth] + oral_resnet_path_list

WEIGHT_GROUP_TABLE = OrderedDict({
    "normal":(2500_0000, nn.init.uniform_, {"a":-0.1, "b":0.1}),
    "ones": (10000, nn.init.ones_, {}),
    "zeros": (10000, nn.init.zeros_, {}),
    # "embedding": (1000_0000, nn.init.normal_, {"mean":0.0, "std":1.0}), # For Embedding
    # "gamma": (1, nn.init.zeros_, {}), # For xresnet
    })
param_sum = dict({k:0 for k in WEIGHT_GROUP_TABLE})

def load_weight_groups():
    CPU = torch.device('cpu')
    path_list = get_paths()
    weight_groups = []
    for idx, p in enumerate(path_list):
        weight_group_np = torch.load(p, map_location=CPU)
        weight_groups.append(weight_group_np)
    return weight_groups

def load_models_from_weight_groups():
    weight_groups = load_weight_groups()
    # for wg in weight_groups:
    #     print(wg['normal'][:10])
    models = []
    for wg in weight_groups:
        m = Resnet18(num_classes = 10)
        copy_param_val(m,wg,wp_start_pos=param_sum)
        models.append(m)
    # for m in models:
    #     for name, param in m.named_parameters():
    #         print(param.view(-1)[:10])
    #         break
    return models

def _plot_histogram_layer(param_df, layer_name):
    mods = param_df['Model'].unique()
    fig, axs = plt.subplots(figsize = (20, 8), ncols = len(mods), nrows =  1, sharey = True)
    
    for idx in range(len(mods)):
        sns.histplot(data = param_df.loc[param_df['Model'] == mods[idx]], x='Weight', bins = 100, binrange = (-1.0, 1.0), ax = axs[idx])
        axs[idx].set_title(mods[idx], fontsize = 10)
        if(idx == 0):
            axs[idx].set_ylabel(layer_name)
    path = f"plots/resnet_{layer_name}.png"
    plt.savefig(path, dpi = 100)
    print(f"produce figure in @{path}")
    plt.close(fig)  

def plot_histogram_layer():
    models = load_models_from_weight_groups()
    for layer_idx, k in tqdm(enumerate(models[0].state_dict())):
        if("running" in k or "tracked" in k):
            continue
        param_df = []
        for idx in range(len(models)):
            param = models[idx].state_dict()[k].view(-1).numpy()
            for val in param: ## val is scalar
                if idx > 1:
                    y = f'oral_{idx}'
                elif idx == 1:
                    y = 'same pos'
                else:
                    y = 'different pos'
                param_df.append([val,y])
        param_df = pd.DataFrame(param_df, columns=['Weight', 'Model'])
        _plot_histogram_layer(param_df, f"{k}")


def plot_histogram_all_weight():
    weight_group = load_weight_groups()
    for key in weight_group[0]:
        weight_group_df = []
        for idx, wg in enumerate(weight_group):
            for val in wg[key]: ## val is scalar
                if idx > 1:
                    y = f'oral_{idx}'
                elif idx == 1:
                    y = 'same pos'
                else:
                    y = 'different pos'
                weight_group_df.append([val, y])
        weight_group_df = pd.DataFrame(weight_group_df, columns=['Weight', 'Model'])
        _plot_histogram_layer(weight_group_df, f"{key}")

def main():
    # plot_histogram_layer()
    plot_histogram_all_weight()

if __name__ == '__main__':
    main()
'''


## the plot code for the utaf results
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='ticks')
import fire
import csv
import numpy as np
from collections import OrderedDict
import matplotlib as mpl
import pandas as pd
import json

SMALL_SIZE = 12
MEDIUM_SIZE = 16

BIGGER_SIZE = 16
TITLE_SIZE = 20

plt.rc('axes', titlesize=BIGGER_SIZE, labelsize = BIGGER_SIZE+2)     # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels\
# plt.rc('xlabel', fontsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ylabel', fontsize=MEDIUM_SIZE)    # fontsize of the tick labels\
plt.rc('legend', fontsize=MEDIUM_SIZE-2)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

PREFIX = 'exp_data'

## used to plot the limit suite 
def prepare_fig_d1_data():
    path = f'{PREFIX}/D1.csv'
    reader = csv.reader(open(path, 'r'))
    next(reader)
    df = []
    ratio = []
    for idx, line in enumerate(reader):
        df.append([idx, float(line[1]), 'carrier'])
        df.append([idx, float(line[2]), 'secret'])
        ratio.append(float(line[0]))
    df = pd.DataFrame(df, columns = ['ratio', "ACC", "mode"])
    
    return df, np.array(ratio)

def prepare_fig_d2_data():
    path = f'{PREFIX}/D2.csv'
    reader = csv.reader(open(path, 'r'))
    next(reader) # skip the header
    df = []
    
    for idx,line in enumerate(reader):
        secret_arrs = json.loads(line[-1])
        df.append([idx, float(line[1]), 'carrier'])
        for val in secret_arrs:
            df.append([idx, float(val), 'secret'])
    df = pd.DataFrame(df, columns = ['N', 'ACC', 'mode'])
    return df

def plot_limit_suite():
    fig, axs = plt.subplots(figsize = (8, 4), ncols = 2, nrows =  1)
    d1_df, ratio = prepare_fig_d1_data()
    d2_df = prepare_fig_d2_data()
    
    sns.lineplot(x = 'ratio', y = 'ACC', hue = 'mode', style='mode', hue_order = ['carrier', 'secret'], data = d1_df, palette=sns.color_palette()[2:4], ax = axs[0], markers=True, dashes=False, ci = "sd", linewidth = 2.5, markersize = 10)
    sns.lineplot(x = 'N', y = 'ACC', hue = 'mode', style='mode', hue_order = ['carrier', 'secret'], data = d2_df, palette=sns.color_palette()[2:4], ax = axs[1], markers=True, dashes=False, ci = "sd", linewidth = 2.5, markersize = 10)

    ## the size ratio curve
    ax0_twin = axs[0].twinx()
    ax0_twin.plot([i for i in range(len(ratio))], ratio, marker='D', c = sns.color_palette()[0], linewidth = 1.5, ls="--")
    ax0_twin.set_ylim(0.0, 1.0)
    ax0_twin.set_ylabel('Size Ratio')

    
    ## carrier normal 
    axs[0].axhline(y=95.5, c=sns.color_palette()[2], ls="--", lw=2)
    ## secret normal 
    axs[0].axhline(y=98.45, c=sns.color_palette()[3], ls="--", lw=2)
    
    axs[0].set_ylim(50, 100)
    axs[1].set_ylim(80, 100)
    axs[0].set_xticks([1, 3, 5, 7, 9, 11])
    axs[0].set_xticklabels(['0.9', '0.7', '0.5', '0.3', '0.1', '0.01'])

    axs[1].set_xticks([0,1,2,3,4,5])
    axs[1].set_xticklabels(['10', '20', '50', '100', '200', '400'])

    axs[1].set_ylabel('')
    axs[0].set_ylabel('ACC (%)')
    axs[1].set_xlabel("Number of Models ($N$)")
    axs[0].set_xlabel("Reduction of ParamPool ($\gamma$)")
    handles, labels = axs[0].get_legend_handles_labels()
    # twin_handles, _ = ax0_twin.get_legend_handles_labels()
    axs[0].legend(handles=handles, labels=['Carrier', 'Secret'], title = '')
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles=handles, labels=['Carrier', 'Secret'], title = '')
    axs[0].set_title('(a)', fontsize=BIGGER_SIZE)
    axs[1].set_title('(b)', fontsize=BIGGER_SIZE)
    plt.tight_layout()
    fig.savefig('limit_suite_paper.pdf')





if __name__ == '__main__':
    fire.Fire()


