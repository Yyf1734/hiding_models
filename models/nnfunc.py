from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## util for param_group
def is_bn_weight(name):
    return (name.find('bn') != -1 and name.find('weight') != -1) or name.find('shortcut.1.weight') != -1

def is_bn_bias(name):
    return (name.find('bn') != -1 and name.find('bias') != -1) or name.find('shortcut.1.bias') != -1

def param_name_to_key(name):
    if is_bn_weight(name):
        return 'ones'
    elif is_bn_bias(name):
        return 'zeros'
    # elif 'embedding' in name:
    #     return 'embedding'
    else: ## for normal params
        return 'normal'

def get_params(params_origin, param_size, offset, rand_perm = None,  a = None, b = None):
    ## TODO: clone() is necessary for n-copy except n == 1 
    ## Rand_perm automatically clone the params; Clone() is necessary for multi-reuse of weight pool.
    ## When weight_group is too large (e.g. greater than 25 million), one GPU is not enough for operation of clone(). 
    try: 
        params = params_origin if rand_perm is not None else params_origin.clone() 
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
    if params is None:
        return
    offset = wp_start_pos.copy() if 'oral' not in params.get_keys() else dict({k:0 for k in params.get_keys()})
    params_permed = {key: params.get(key)[rand_perm[key]] for key in rand_perm} if rand_perm is not None else params.to_dict()
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
    if copy_buffer:
        buffers = params.get_g_buffers()
        # print(f'buffers in weight group is {len(buffers)}')
        offset = 0
        for name, buffer in model.named_buffers():
            buffer_size = np.prod(buffer.shape)
            if not 'running' in name:
                continue
            # print(name)
            # print(buffer.shape)
            buffer_val = buffers[offset:buffer_size + offset]
            buffer.data = buffer_val.reshape(buffer.shape).data
            offset =  buffer_size + offset

## TODO: group, the length of params_delta
def _copy_param_delta(net_param, params, offset):
    params_delta = torch.zeros_like(params)
    net_param = net_param.view(-1)
    param_size = net_param.shape[0]
    repete_count = (param_size + offset) // len(params)
    if(repete_count == 0):
        params_delta[offset:param_size + offset] = net_param - params[offset:param_size+offset]
        offset += param_size
        return params_delta, offset
    cur = 0
    params_delta[offset:] = net_param[cur:cur+len(params) - offset]-params[offset:]
    repete_count -= 1
    cur += len(params) - offset
    for i in range(repete_count):
        params_delta += net_param[cur:cur+len(params)] - params
        cur += len(params)
    offset = param_size + offset - (repete_count + 1) * len(params)
    params_delta[:offset] += net_param[cur:] - params[:offset]
    return params_delta, offset

def copy_param_delta(model, params, rand_perm = None, rand_perm_reverse = None,  wp_start_pos = None, **kwargs):
    if params is None:
        return None
    params_delta = dict({k: torch.zeros_like(params.get(k)) for k in params.get_keys()})
    params_permed = {key: params.get(key)[rand_perm[key]] for key in rand_perm} if rand_perm is not None else params.to_dict()
    offset = wp_start_pos.copy() if 'oral' not in params.get_keys() else dict({k: 0 for k in params.get_keys()})
    for name, module in model.named_modules():
        if not hasattr(module, 'weight') and (not hasattr(module, 'bias')): # or type(module.bias) is bool) and not hasattr(module, 'weight_ih_l0'):
            continue
        weight_key = 'oral' if 'oral' in params.get_keys() else 'normal'
        bias_key = 'oral' if 'oral' in params.get_keys() else 'normal'
        if type(module) in [torch.nn.modules.batchnorm.BatchNorm2d,torch.nn.modules.batchnorm.BatchNorm1d] and 'oral' not in params.get_keys():
            weight_key = 'ones'
            bias_key = 'zeros'
        if hasattr(module, 'weight_ih_l0') and type(module.bias) is bool: ## For RNN
            local_param_delta, offset[weight_key] = _copy_param_delta(module.weight_ih_l0.data, params_permed[weight_key], offset[weight_key])
            params_delta[weight_key][:len(local_param_delta)] += local_param_delta
            local_param_delta, offset[weight_key] = _copy_param_delta(module.weight_hh_l0.data, params_permed[weight_key], offset[weight_key])
            params_delta[weight_key][:len(local_param_delta)] += local_param_delta
            local_param_delta, offset[bias_key] = _copy_param_delta(module.bias_ih_l0.data, params_permed[weight_key], offset[weight_key])
            params_delta[bias_key][:len(local_param_delta)] += local_param_delta
            local_param_delta, offset[bias_key] = _copy_param_delta(module.bias_hh_l0.data, params_permed[weight_key], offset[weight_key])
            params_delta[bias_key][:len(local_param_delta)] += local_param_delta
        else:
            local_param_delta, offset[weight_key] = _copy_param_delta(module.weight.data, params_permed[weight_key], offset[weight_key])
            params_delta[weight_key][:len(local_param_delta)] += local_param_delta
            if hasattr(module, 'bias') and module.bias is not None: # and type(module.bias) is not bool:
                local_param_delta, offset[bias_key] = _copy_param_delta(module.bias.data, params_permed[bias_key], offset[bias_key])
                params_delta[bias_key][:len(local_param_delta)] += local_param_delta
        ## TODO: TRICK -> For xse_resnext50
        if hasattr(module, 'gamma') and module.gamma is not None:
            params.set('gamma', module.gamma)
    ## For buffers
    buffers = []
    for name, buffer in model.named_buffers():
        if 'running' in name:
            buffers.append(buffer.data.detach().view(-1))
    if len(buffers) > 1:
        buffers = torch.cat(buffers, dim=0)
        params.set_buffers(buffers)
        b = params.get_buffers()
    ## reverse the rand_perm
    if rand_perm_reverse is not None:
        for key in rand_perm_reverse:
            params_delta[key][:len(rand_perm_reverse[key])] = params_delta[key][rand_perm_reverse[key]]
    return params_delta

def accumulate_param_delta(model_delta, local_model_delta, alpha_group = None):
    # TODO: alpha = lr?
    # if alpha_group is None:
    #     alpha_group = dict()
    #     for key in local_model_delta:
    #         alpha_group[key] = torch.ones(1)
    for key in local_model_delta:
        if not (key in model_delta):
            model_delta[key] = local_model_delta[key] 
        else: 
            model_delta[key] += local_model_delta[key] 
    return model_delta

def upadte_weight_group(weight_group, param_delta):
    for key in param_delta: ## len(param_delta) may less than len(weight_group)

        weight_group.add(key, param_delta[key])
    return weight_group

def save_models(model = None, acc = None, idx = None, subdir = None, device = None, key = None, save = False):
    if not save:
        return
    state = {
        'net': model.cpu().state_dict(), ## if net is still in the GPU, weight_group will be saved at the same time.
        'acc': acc,
        'idx': idx, 
    }
    torch.save(state, f'{subdir}/{key}.pth')
    model#.to(device)

from TaskModel import WeightGroup
def model_to_weight_group(model = None, weight_group_num = None, wp_start_pos = 0, return_dict = True, clip = True):
    weight_group= dict({'normal':[],'ones':[], 'zeros':[]})
    for name, module in model.named_modules():
        if not hasattr(module, 'weight') and (not hasattr(module, 'bias')):
            continue
        weight_key = 'normal'
        bias_key =  'normal'
        if type(module) in [torch.nn.modules.batchnorm.BatchNorm2d,torch.nn.modules.batchnorm.BatchNorm1d]:
            weight_key = 'ones'
            bias_key = 'zeros'
        if hasattr(module, 'weight_ih_l0') and type(module.bias) is bool: ## For RNN
            weight_group[weight_key].append(module.weight_ih_l0.data.view(-1).clone())
            weight_group[weight_key].append(module.weight_hh_l0.data.view(-1).clone())
            weight_group[bias_key].append(module.bias_ih_l0.data.view(-1).clone())
            weight_group[bias_key].append(module.bias_hh_l0.data.view(-1).clone())
        else:
            weight_group[weight_key].append(module.weight.data.view(-1).clone())
            if hasattr(module, 'bias') and module.bias is not None:
                weight_group[bias_key].append(module.bias.data.view(-1).clone())
    ## cat and clip
    ## TODO: randperm; 
    #  if the model is carrier, there is no need to randperm
    for key in weight_group:
        if clip:
            weight_group[key] = torch.cat(weight_group[key],dim=0)[wp_start_pos : wp_start_pos+weight_group_num[key]]
        else:
            weight_group[key] = torch.cat(weight_group[key],dim=0) 
    if not return_dict:
        ## return as WeightGroup
        WP = WeightGroup()
        for key in weight_group:
            WP.set(key, weight_group[key])
    else:
        ## return as dict
        WP = weight_group
    return WP