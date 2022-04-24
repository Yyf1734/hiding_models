import os
from parser import get_parser
## get parsers and set the CUDA VISIBLE
args = get_parser()
from collections import OrderedDict
from typing import Dict, Text
from itertools import cycle
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from train_one_batch import * 
import os
import torchvision.utils as vutils
# ## for imagenette 
# from fastai.basics import *
# from fastai.vision.models.xresnet import *

from dataset import *
from utils import *
from TaskModel import TaskModel, WeightGroup
from models.fcn import FCN, LR_FCN
from models.lenet import LeNet
from models.textcnn import TextCNN
from models.resnet import *
from models.m5 import M5
from models.vgg import *
from models.dcgan import *
from models.mobilenetv1 import mobilenetv1_ssd
from models.ssd import MultiboxLoss
from models.TSModels.tsmodel import TSClassifier
from models.TSModels.loss import BCELoss
from models.seq2seq import Seq2Seq
from models.nnfunc import *
'''
task_model_entry = (dataloader_func, model, model_kwargs, train_one_batch_fn, \
                          optimizer, optim_kwargs, 
                          scheduler, sch_kwargs, loss_fn) # the best acc of full model
'''
TASK_MODEL_ENTRY = OrderedDict({
    "cifar10-ResNet18": (get_cifar10_dataloader, Resnet18, {"num_classes": 10}, train_one_batch_for_clf, \
                            torch.optim.SGD, {"lr":0.1, "momentum":0.9, "weight_decay":5e-4}, \
                            torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()),  ## 95.65% <- SGD with lr 0.1, 11164362 + 4800
    "cifar10-VGG": (get_cifar10_dataloader, vgg16, {"num_classes": 10}, train_one_batch_for_clf, \
                            torch.optim.SGD, {"lr":0.1, "momentum":0.9, "weight_decay":5e-4}, \
                             torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()),   ## 94.05% <- SGD with lr 0.1
    # "gtsrb-VGG": (get_gtsrb_dataloader, vgg16, {"num_classes": 43}, train_one_batch_for_clf, \
    #                         torch.optim.SGD, {"lr":0.01, "momentum":0.9, "weight_decay":5e-4}, \
    #                         torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()), ## 
    "mnist-LeNet": (get_mnist_dataloader_32, LeNet, {}, train_one_batch_for_clf, \
                            torch.optim.SGD, {"lr":0.01*args.hidden_ratio, "momentum":0.9, "weight_decay":5e-4}, \
                            torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()), ## 99.4% <- SGD with lr 1e-2
    "dermamnist-LeNet": (get_dermamnist_dataloader, LeNet, {}, train_one_batch_for_clf, \
                            torch.optim.SGD, {"lr":0.001, "momentum":0.9, "weight_decay":5e-4}, \
                            torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()), ## 73.07 <- SGD with lr 0.001
    "imdb-TextCNN": (get_imdb_dataloader, TextCNN, {}, train_one_batch_for_clf, \
                            torch.optim.SGD, {"lr":0.01, "momentum":0.9, "weight_decay":5e-4}, \
                            torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()),  ## 90.04% <- SGD with lr 0.01
    "speechcommand-M5": (get_speechcommand_dataloader, M5, {}, train_one_batch_for_clf, \
                            torch.optim.SGD, {"lr":0.1, "momentum":0.9, "weight_decay":5e-4}, \
                             torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()), ## 96.92% <- SGD with lr 0.1
    "cifar-onlyG": (get_cifar10_dataloader_gan,generator,{}, train_one_batch_onlyG,
 	                 torch.optim.Adam,{"lr":2e-4,"betas":(0.5,0.999)},
 	                 torch.optim.lr_scheduler.CosineAnnealingLR, {}, Image_Pair_Loss),
    # "Speech-onlyG":(get_speechcommand_dataloader, voiceG,{},train_one_batch_onlyG,
    #                 torch.optim.Adam,{"lr":2e-4,"betas":(0.5,0.999)},
    #                 torch.optim.lr_scheduler.CosineAnnealingLR, {}, Image_Pair_Loss),
    "Speech-onlyG_shuffle":(get_speechcommand_dataloader, voiceG,{},train_one_batch_onlyG_shuffle,
                    torch.optim.Adam,{"lr":2e-4,"betas":(0.5,0.999)},
                    torch.optim.lr_scheduler.CosineAnnealingLR, {}, Image_Pair_Loss),
	# "cifar-GAN": (get_cifar10_dataloader_gan,generator,{}, train_one_batch_gan,
    #            torch.optim.Adam,{"lr":2e-4,"betas":(0.5,0.999)},
    #            torch.optim.lr_scheduler.CosineAnnealingLR, {},nn.BCELoss()), # -40
    # "bufferG": (get_speechcommand_dataloader,generator_without_bn,{}, train_one_batch_bufferG,
    #            torch.optim.Adam,{"lr":2e-4,"betas":(0.5,0.999)},
    #            torch.optim.lr_scheduler.CosineAnnealingLR, {},nn.CrossEntropyLoss()),
    # "pmnist-fcn": (get_mnist_dataloader_28, FCN, {}, train_one_batch_for_clf,
    #                         torch.optim.SGD, {"lr":0.01, "momentum":0.9, "weight_decay":5e-4}, \
    #                         torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()),  ## 98.5% <- SGD with lr 0.01
    "warfit-fcn": (get_warfit_dataloader, LR_FCN, {}, train_one_batch_for_lr,
                            torch.optim.SGD, {"lr":0.001, "momentum":0.9}, \
                            torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.MSELoss()), # loss=8.7 with lr 0.001
    # "imagenette-xse_resnext50": (get_imagenette_dataloader, xse_resnext50, {'n_out':10, 'act_cls':Mish, 'sa':1, 'sym':0, 'pool':MaxPool}, train_one_batch_for_clf,
    #                         ranger, {"lr":8e-3, "wd":1e-2, 'mom':0.95, 'sqr_mom':0.99, 'eps':1e-6, 'beta':0}, \
    #                         None, {}, LabelSmoothingCrossEntropy()), # 92.15%,  25547802 + 34176
    # "VOC-mobilenetv1_ssd":(get_VOC_dataloader, mobilenetv1_ssd, {"num_classes": 21},train_one_batch_ssd,\
    #                         torch.optim.SGD,  {"lr":0.002, "momentum":0.9, "weight_decay":5e-4}, \
    #                         torch.optim.lr_scheduler.CosineAnnealingLR, {}, MultiboxLoss()), # loss=3.8, {'normal': 9447108, 'ones': 10944, 'zeros': 10944},
    "Uwave-rnn_ts":(partial(get_ts_dataloader, dataset_name='UWaveGestureLibrary_3dim'), TSClassifier, {"model_name":'gru', "dataset_name":'UWaveGestureLibrary_3dim'}, train_one_batch_for_clf,\
                            torch.optim.RMSprop, {"lr":0.002, "momentum":0.99, "weight_decay":0.99},\
                            torch.optim.lr_scheduler.CosineAnnealingLR, {}, BCELoss(class_num=8)), # 82% ± 5% with lr 0.002
    # "seq2seq-rnn":(get_eng_fra_dataloader, Seq2Seq, {}, train_one_batch_for_seq2seq,\
    #                         torch.optim.SGD, {"lr":0.01},\
    #                         torch.optim.lr_scheduler.CosineAnnealingLR, {}, torch.nn.NLLLoss()), # 4.096 {3476221}
    # "cifar10-VGG": (get_cifar10_dataloader, vgg16_wo_bn, {"num_classes": 10}, train_one_batch_for_clf, \
    #                         torch.optim.SGD, {"lr":0.0002*args.hidden_ratio, "momentum":0.9, "weight_decay":5e-4}, \
    #                          torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()), 
    # "cifar10-VGG": (get_cifar10_dataloader, vgg11_wo_bn, {"num_classes": 10}, train_one_batch_for_clf, \
    #                         torch.optim.SGD, {"lr":0.0002, "momentum":0.9, "weight_decay":5e-4}, \
    #                          torch.optim.lr_scheduler.CosineAnnealingLR, {}, nn.CrossEntropyLoss()), 
    })

## weight_group_table = (param_num, initial_func, initial_func_kwargs)
## xse_resnext50: {'normal': 25547802, 'ones': 34176, 'zeros': 34176}
WEIGHT_GROUP_TABLE = OrderedDict({
    "normal":[int(11164362*args.hidden_ratio), nn.init.uniform_, {"a":-0.1, "b":0.1}],
    "ones": [4800, nn.init.ones_, {}],
    "zeros": [4800, nn.init.zeros_, {}],
    # "embedding": [1000_0000, nn.init.normal_, {"mean":0.0, "std":1.0}], # For Embedding
    "gamma": [1, nn.init.zeros_, {}], # For xresnet
    })

def get_param_group(device = None, oral_init = False, pretrained_init = False):
    # device = torch.device('cpu')
    weight_group = WeightGroup()
    if oral_init is True:
        weight_pool = []
        if len(TASK_MODEL_ENTRY) > 1:
            print('ERROR: There are more than 1 task-model instance in oral_init mode !')
            exit()
        for key in TASK_MODEL_ENTRY:
            _, model, model_kwargs, _, _, _, _, _, _ = TASK_MODEL_ENTRY[key]
            model_oral = model(**model_kwargs)
            for name, param in model_oral.named_parameters():
                lens = np.prod(param.shape)
                print(f'{name} min is {param.min().item()} max is {param.max().item()} mean is {param.mean().item()}')
                weight_pool.append(param.view(-1).data)
        weight_pool = torch.cat(weight_pool).to(device)
        weight_group.set(key = 'oral', v = weight_pool)
    elif pretrained_init is True:
        weight_group_ = torch.load(f'pretrained_wp/weight_group.pth')
        weight_group = WeightGroup()
        for key in weight_group_:
            weight_group.set(key, weight_group_[key].to(device))
    else:
        for key in WEIGHT_GROUP_TABLE:
            param_num, initial_func, param_kwargs = WEIGHT_GROUP_TABLE[key][:4]
            weight_pool = Variable(torch.zeros((param_num,))) # , requires_grad = True)
            initial_func(weight_pool, **param_kwargs)
            weight_pool = weight_pool.to(device)
            weight_group.set(key = key, v = weight_pool)
    print_weight_group_info(WEIGHT_GROUP_TABLE, weight_group)
    return weight_group

def get_task_model_instances(args = None) -> Dict[str, TaskModel]:
    print('Building task_model_instances ...')
    task_model_instances = OrderedDict()
    param_sum = dict({k:0 for k in WEIGHT_GROUP_TABLE})
    for i, key in enumerate(TASK_MODEL_ENTRY):
        dataloader_func, model, model_kwargs, train_one_batch_fn, optimizer_fn, optim_kwargs, scheduler_fn, _, loss_fn = TASK_MODEL_ENTRY[key]
        if 'pmnist' in key:
            param_num_max = 0
            hidden_dim = [100, 100]
            for pi in tqdm(range(args.pmnist_num)):
                if args.same_arch is False:
                    hidden_dim = [args.pmnist_num + 2*pi, 50 + args.pmnist_num - pi]
                    # hidden_dim = [100 + args.pmnist_num + 2*pi, 50 + args.pmnist_num - pi]
                param_num_cur = hidden_dim[0] * 28*28 + hidden_dim[0]*hidden_dim[1]+hidden_dim[1]*10
                param_num_max = max(param_num_max, param_num_cur)
                key_cur = key + f'_{28*28}_{hidden_dim}_{10}'
                rand_seed_for_x = pi if args.shuffle_mnist is True else 0
                rand_seed_for_y = pi if args.change_label is True else 0
                ## set the seed to shuffle xs or ys
                model_kwargs = {"hidden_dims":hidden_dim,"rand_seed_for_x": rand_seed_for_x, "rand_seed_for_y": rand_seed_for_y}
                task_model_instances[key_cur] = TaskModel(key_cur, dataloader_func, model, model_kwargs, train_one_batch_fn, optimizer_fn, optim_kwargs, scheduler_fn, \
                                                loss_fn, torch.device(f'cuda:{args.which_cuda}'), args.max_iter, WEIGHT_GROUP_TABLE, (pi+1)*args.rand_perm*10, \
                                                param_sum.copy(), args.test_interval, args.keep_training)
                for k in task_model_instances[key_cur].param_num:
                    param_sum[k] = (param_sum[k] + task_model_instances[key_cur].param_num[k]) % WEIGHT_GROUP_TABLE[k][0] if args.different_start_pos is True else 0
                # print(task_model_instances[key_cur].model.model)
                for x, y in task_model_instances[key_cur].train_loader:
                    if(len(x.shape) > 2):
                        x = x.view(-1, np.prod(x.shape[1:]))
                    if args.shuffle_mnist and task_model_instances[key_cur].model.rand_seed_for_x > 0:
                        setup_seed(task_model_instances[key_cur].model.rand_seed_for_x)
                        perm = torch.randperm(28*28)
                        x = x[:,perm]
                    x = x.view(-1, 1,28,28)
                    subdir = f'./checkpoint/{get_trial_name(WEIGHT_GROUP_TABLE, task_model_instances, args)}'
                    if not os.path.isdir(f'{subdir}'):
                        os.mkdir(f'{subdir}')
                    if not os.path.isdir(f'{subdir}/pmnist'):
                        os.mkdir(f'{subdir}/pmnist')
                    y = task_model_instances[key_cur].model.change_label(y)
                    vutils.save_image(x,f'{subdir}/pmnist/{pi}.png',normalize=True)
                    break
                # print(f"{key_cur} has been built.")
            WEIGHT_GROUP_TABLE['normal'][0] = param_num_max
        else:
            task_model_instances[key] = TaskModel(key, dataloader_func, model, model_kwargs, train_one_batch_fn, optimizer_fn, optim_kwargs, scheduler_fn, \
                                                loss_fn, torch.device(f'cuda:{args.which_cuda}'), args.max_iter, WEIGHT_GROUP_TABLE, (i+1)*args.rand_perm, \
                                                param_sum.copy(), args.test_interval, args.keep_training)
            for k in task_model_instances[key].param_num:
                param_sum[k] = (param_sum[k] + task_model_instances[key].param_num[k]) % WEIGHT_GROUP_TABLE[k][0] if args.different_start_pos is True else 0
            task_model_instances[key].further_init(pair_size = args.pair_size) ## For GAN
            print(f"{key} has been built.")
    print('Task_model_instances have been built.')
    return task_model_instances
