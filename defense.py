import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import prune
import tqdm

from models.nnfunc import copy_param_val, model_to_weight_group
from TaskModel import WeightGroup
from utils import progress_bar_local, print_bar_train_info
from train_one_batch import get_buffer_param, test_clf

## For adding noise
def add_noise_to_weight_pool(weight_group = None, mean = 0, std = 0.001, keys=['normal']):
    print('Add Normal Noise with mean {} and std {} to weight_pool {}.'.format(mean, std, keys))
    difference_norm = 0
    for key in keys:
        weight_group_clone = weight_group[key].clone()
        lens = weight_group[key].shape[0]
        noise = np.random.normal(mean, std, lens)
        weight_group[key] = (weight_group[key]+ noise).to(dtype=torch.float)
        difference_norm += torch.norm(weight_group[key] - weight_group_clone)
    print('After adding noise to weight pool, the diffetence_norm is {}'.format(difference_norm))
    return weight_group

def add_noise_to_carrier(weight_group = None, mean = 0, std = 0.001, keys=['normal'], carrier_param_num = None):
    print('Add Normal Noise with mean {} and std {} to the whole carrier model.'.format(mean, std))
    difference_norm = 0
    for key in keys:
        weight_group_clone = []
        lens = weight_group[key].shape[0]
        carrier_lens = carrier_param_num[key]
        repete_num = carrier_lens // lens
        for i in range(repete_num):
            weight_group_clone.append(weight_group[key].clone())
        # weight_group_clone.append(weight_group[key][:carrier_param_num[key] % lens].clone())
        weight_group_clone = torch.cat(weight_group_clone,dim=0)
        print(f'the weight pool is expanded to {weight_group_clone.shape[0]}')
        noise = np.random.normal(mean, std, carrier_lens-carrier_param_num[key] % lens)
        weight_group[key] = (weight_group_clone+noise).to(dtype=torch.float)
        difference_norm += torch.norm(weight_group[key] - weight_group_clone)
    print('After adding noise to carrier, the diffetence_norm is {}'.format(difference_norm))
    return weight_group

## For quantization
def simple_quantization_to_float16(weight_group = None, keys=['normal']):
    print('Quantize weight_pool {} to float16.'.format(keys))
    difference_norm = 0
    for key in keys:
        weight_group_clone = weight_group[key].clone()
        weight_group[key] = (weight_group[key]).to(dtype=torch.float16).to(dtype=torch.float32)
        difference_norm += torch.norm(weight_group[key] - weight_group_clone)
    print('After quantization, the diffetence_norm is {}'.format(difference_norm))
    return weight_group

## For pruning
def prune_tensors(tensors, prune_ratio = None):
    weights=np.abs(tensors.cpu().numpy())
    weightshape=weights.shape
    rankedweights=weights.reshape(weights.size).argsort()
    
    num = weights.size
    prune_num = int(np.round(num*prune_ratio))
    count=0
    masks = np.zeros_like(rankedweights)
    for n, rankedweight in enumerate(rankedweights):
        if rankedweight > prune_num:
            masks[n]=1
        else: count+=1
    print("total weights:", num)
    print("weights pruned:",count)
    masks=masks.reshape(weightshape)
    weights=masks*weights
    return torch.from_numpy(weights).to(dtype=torch.float32), masks

def prune_weight_pool(weight_group = None, keys=['normal'], prune_ratio = 0.2):
    print('Pruning the whole weight_pool {}.'.format(keys))
    difference_norm = 0
    for key in keys:
        weight_group_clone = weight_group[key].clone()
        weight_group[key], _ = prune_tensors(weight_group[key], prune_ratio = prune_ratio)
        difference_norm += torch.norm(weight_group[key] - weight_group_clone)
    print('After pruning, the diffetence_norm is {}'.format(difference_norm))
    return weight_group

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):
    num_zeros = 0
    num_elements = 0
    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
    sparsity = num_zeros / num_elements
    return num_zeros, num_elements, sparsity

def measure_global_sparsity(model, weight=True, bias=False,
                            conv2d_use_mask=False, linear_use_mask=False):
    num_zeros = 0
    num_elements = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements
        elif isinstance(module, torch.nn.Linear):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements
    sparsity = num_zeros / num_elements
    return num_zeros, num_elements, sparsity

def remove_parameters(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
    return model

def prune_carrier(task_model = None, weight_group_ = None, prune_ratio = 0.2, filter_pruning = False, recover_for_pruning = False):
    print('Pruning the carrier model {}'.format(task_model.name))
    ## weight pool to carrier
    model = task_model.model.to(task_model.device)
    weight_group = WeightGroup()
    for key in weight_group_:
        weight_group.set(key, weight_group_[key])
    copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    for name, module in model.named_modules():
        if filter_pruning is True and (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)): ## WP
            print(f'prune the module of {name}')
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)
        if filter_pruning is False and (isinstance(module, nn.Conv2d)): ## FP
            prune.ln_structured(module, name='weight', amount=prune_ratio, dim=0, n=1) # Ln-norm
    remove_parameters(model)
    num_zeros, num_elements, sparsity = measure_global_sparsity(model)
    print('num_zeros %d \t num_elements %d \t sparsity %.3f' % (num_zeros, num_elements, sparsity))

    ## carrier to weight pool
    print('After purning, the weight pool is constrcuted from the carrier.')
    if recover_for_pruning:
        weight_group = wp_recover_from_redundancy(model, weight_group_)
    else:
        weight_group = model_to_weight_group(model,{key: weight_group_[key].shape[0] for key in weight_group_})

    for key in weight_group:
        print(f'{key}.len is {weight_group[key].shape[0]}')
    
    
    difference_norm = 0
    for key in weight_group:
        lens = min(weight_group[key].shape[0], weight_group_[key].shape[0])
        difference_norm += torch.norm(weight_group[key][:lens] - weight_group_[key][:lens])
    print('After {} pruning, the diffetence_norm is {}'.format('filter' if filter_pruning else 'weight', difference_norm))
    return weight_group

def wp_recover_from_redundancy(model = None, weight_group_ = None):
    ## TODO：wp_start_pos
    ## Only for simple model, e.g. resnet
    weight_group = model_to_weight_group(model,{key: weight_group_[key].shape[0] for key in weight_group_}, clip = False)

    print('Origin wp from pth')
    ori_wp_num = {key: weight_group_[key].shape[0] for key in weight_group_}
    print(ori_wp_num)
    print('The model param after pruning')
    model_wp_num = {key: weight_group[key].shape[0] for key in weight_group}
    print(model_wp_num)

    for key in weight_group_:
        zero_num = torch.sum(weight_group_[key] == 0).item()
        print(f'The origin wp {key} has {zero_num} zeros.')
    prune_zero_num = 0
    for key in weight_group:
        zero_num = torch.sum(weight_group[key][:ori_wp_num[key]] == 0).item() 
        print(f'The wp from pruned model param {key} has {zero_num} zeros.')
        prune_zero_num += zero_num 
    
    print('*************** Let\'s recover it from nonzeros. *******************')
    for key in weight_group:
        padding_zeros = torch.zeros(ori_wp_num[key] - model_wp_num[key] % ori_wp_num[key])
        weight_group[key] = torch.cat([weight_group[key], padding_zeros], dim=0)
        weight_group[key] = weight_group[key].view(-1, ori_wp_num[key]) 
        nonzero_nums = torch.sum(weight_group[key] != 0, dim=0) 

        temp = torch.nan_to_num(torch.div(torch.sum(weight_group[key],dim=0), nonzero_nums))
        # diff = torch.norm(torch.sum(weight_group[key],dim=0)-torch.mul(temp, nonzero_nums))
        # print(f'local diff is {diff}')
        weight_group[key] =  temp 
    
    recover_zero_num = 0
    for key in weight_group:
        zero_num = torch.sum(weight_group[key] == 0).item() 
        print(f'The wp {key} from pruned model after recovering has {zero_num} zeros.')
        recover_zero_num += zero_num
    print(f'*************** Recover {prune_zero_num - recover_zero_num} nonzeros. *******************')
    return weight_group

## For fine tuning
## About thr lr
# 190th 0.00049882
# 191th 0.00039426
# 192th 0.00030195
# 193th 0.00022190
# 194th 0.00015413
# 195th 0.00009866
# 196th 0.00005551
# 197th 0.00002467
# 198th 0.00000617
# 199th 0.00000000
## TODO: reset the lr and initialize the scheduler
def finetuning(task_model = None, weight_group_ = None, ft_index = -1, ft_epochs = None):
    ## For resnet18
    ## layer name: [conv1.weight, bn1.weight, bn_bias, layer1, layer2, layer3, layer4, linear.weight, linear_bias]
    ## start_layer_index: 1, 4, 16, 31, 46, 61
    start_layer_index_list = [1, 4, 16, 31, 46, 61]
    sli = start_layer_index_list[ft_index]
    ## weight pool to carrier
    model = task_model.model.to(task_model.device)
    weight_group = WeightGroup()
    for key in weight_group_:
        weight_group.set(key, weight_group_[key])
    # copy_param_val(model, params = weight_group, rand_perm=task_model.rand_perm, wp_start_pos = task_model.wp_start_pos)
    get_buffer_param(task_model, weight_group, use_test_dataloader=False)
    ft_layer_name = []
    idx = 0
    for name, param in model.named_parameters():
        idx += 1
        if idx < sli:
            param.requires_grad = False
        else:
            param.requires_grad = True
            ft_layer_name.append(name)
    
    print('Finetuning the carrier model {} from {} to {}'.format(task_model.name, ft_layer_name[0], ft_layer_name[-1]))
    model.train()
    max_iter = ft_epochs*task_model.test_interval

    parameters = filter(lambda p: p.requires_grad, task_model.model.parameters())
    task_model.optimizer = torch.optim.SGD(parameters, lr=0.0001, momentum=0.9, weight_decay=5e-4)
    task_model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(task_model.optimizer,T_max = ft_epochs+1)
    for i in range(max_iter):
        task_model.train_idx += 1
        try:
            inputs, targets = next(task_model.train_dataloader_iter)
        except:
            task_model.train_dataloader_iter = iter(task_model.train_loader)
            inputs, targets = next(task_model.train_dataloader_iter)
        task_model.train_one_batch(task_model, inputs, targets)
        progress_bar_local(i, max_iter, print_bar_train_info({task_model.name: task_model}))
        if (i + 1) % 391 == 0:
            print('test model after fine tuning')
            test_clf(-1, task_model)
            task_model.scheduler_step()

    ## carrier to weight pool
    weight_group = model_to_weight_group(model,{key: weight_group_[key].shape[0] for key in weight_group_})

    difference_norm = 0
    for key in weight_group:
        lens = min(weight_group[key].shape[0], weight_group_[key].shape[0])
        difference_norm += torch.norm(weight_group[key].to(torch.device('cpu'))[:lens] - weight_group_[key][:lens])
    print('After fine tuning from {} to {}, the diffetence_norm is {}'.format(task_model.name, ft_layer_name[0], difference_norm))
    return weight_group

def defense_to_weight_pool(task_model_instances = None, weight_group_ = None, args = None):
    ## weight_group_: saved weight group
    defense_method = args.defense_method
    if defense_method is None:
        return weight_group_
    print(f'____________________Test with defense method {defense_method}____________________')
    if defense_method == 'quant':
        weight_group_ = simple_quantization_to_float16(weight_group_)
    elif defense_method == 'noise':
        weight_group_ = add_noise_to_weight_pool(weight_group_)
        # weight_group_ = add_noise_to_carrier(weight_group_, carrier_param_num = task_model_instances["cifar10-ResNet18"].param_num)
    elif defense_method == 'prune':
        weight_group_ = prune_carrier(task_model_instances['cifar10-ResNet18'], weight_group_, args.prune_ratio, args.filter_pruning, args.recover_for_pruning)
        # weight_group_ = prune_weight_pool(weight_group_, args.prune_ratio)
    elif defense_method == 'finetune':
        weight_group_ =  finetuning(task_model_instances['cifar10-ResNet18'], weight_group_, args.ft_index, args.ft_epochs)
    else:
        print(f'{args.defense_method} does not exist, only support to one of (quant, prune, noise, finetune)')
        exit()
    return weight_group_