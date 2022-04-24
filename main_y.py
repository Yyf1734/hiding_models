from initial import *
import os
from train_one_batch import * 
from utils import *
from TaskModel import  WeightGroup
from models.nnfunc import *
from defense import *

def train(idx = None, max_iter = None, accum_mode = None, weight_group = None, selected_task_model_keys = None, **kwargs):
    param_delta = dict()
    for key in selected_task_model_keys:
        keep_training_idx = 0
        while task_model_instances[key].keep_training > keep_training_idx: ## For PMNIST
            keep_training_idx += 1
            task_model_instances[key].train_idx += 1
            if keep_training_idx % task_model_instances[key].test_interval == 0:
                print('scheduler_step For PMNIST')
                task_model_instances[key].scheduler_step() 
            if task_model_instances[key].train_idx % task_model_instances[key].test_interval == 0:
                task_model_instances[key].time_to_test = True
            task_model_instances[key].model.train()
            try:
                inputs, targets = next(task_model_instances[key].train_dataloader_iter)
            except:
                task_model_instances[key].train_dataloader_iter = iter(task_model_instances[key].train_loader)
                inputs, targets = next(task_model_instances[key].train_dataloader_iter)
            if 'pmnist' in key:
                targets = task_model_instances[key].model.change_label(targets)
            # print(f'{idx} start')
            local_param_delta = task_model_instances[key].train_one_batch(task_model_instances[key], inputs, targets, weight_group, **kwargs)
            # print(f'{idx} finish')
            param_delta = accumulate_param_delta(param_delta, local_param_delta)
            if accum_mode == 'in_order':
                weight_group = upadte_weight_group(weight_group, param_delta)
                param_delta = dict()
            progress_bar_local(idx, max_iter, print_bar_train_info(task_model_instances))
    if not accum_mode == 'in_order':
        param_delta = accumulate_param_delta(param_delta, local_param_delta)
        weight_group = upadte_weight_group(weight_group, param_delta)

def test(idx = None, weight_group = None, best_acc_sum_for_all_task = None, save = False, test_trainset_acc = False, subdir = None):
    ## idx == -1: Directly testing the acc of task_model_intances from the saved weight_group.pth
    cur_acc_sum = 0
    cur_acc_list = {k:0 for k in task_model_instances}
    for key in task_model_instances:
        # if task_model_instances[key].time_to_test is False: ## Asynchronous test: Not train across all the batches yet
        if idx >= 0 and (idx + 1) % task_model_instances[key].test_interval  != 0: ## Synchronize test
            return best_acc_sum_for_all_task
        if key == "Speech-onlyG_shuffle":
            if idx == args.max_iter - 1:
                task_model_instances[key].save_GAN_voice(subdir)
            else:
                task_model_instances[key].save_GAN_voice(subdir, save_one = True)
            cur_acc_list[key] = test_gan(task_model_instances[key], subdir)
        elif key == "cifar-onlyG":
            task_model_instances[key].save_GAN_image(subdir)
            cur_acc_list[key] = test_gan(task_model_instances[key],subdir)
        elif key == 'bufferG':
            M_test = M5()
            weight_group.set_g_buffers(2*task_model_instances[key].model(task_model_instances[key].fix_noise_to_buffers).view(-1))
            copy_param_val(M_test, weight_group, copy_buffer = True, wp_start_pos = task_model_instances[key].wp_start_pos)
            cur_acc_list[key] = test_clf(idx, task_model_instances[key], weight_group, subdir, M_test=M_test)
        elif 'war' in key:
            cur_acc_list[key] = test_lr(idx, task_model_instances[key], weight_group, subdir)
        elif 'ssd' in key:
            cur_acc_list[key] = test_ssd(idx, task_model_instances[key], weight_group, subdir)
        elif 'seq' in key:
            cur_acc_list[key] = test_seq2seq(idx, task_model_instances[key], weight_group, subdir)
        elif 'wave' in key:
            cur_acc_list[key] = test_clf(idx, task_model_instances[key], weight_group, subdir) / 100
        else: 
            ## test onlyG for resnet if cifar-onlyG in task_model_instances
            task_model_onlyG = task_model_instances['cifar-onlyG'] if 'cifar-onlyG' in task_model_instances else None
            cur_acc_list[key] = test_clf(idx, task_model_instances[key], weight_group, subdir, task_model_onlyG=task_model_onlyG)
        if idx > 0:
            task_model_instances[key].test_acc_list.append(cur_acc_list[key])
            np.save(f'{subdir}/{task_model_instances[key].name}_test_acc.npy',np.array(task_model_instances[key].test_acc_list))
            cur_acc_sum += cur_acc_list[key]
        else:
            save_models(model = task_model_instances[key].model, acc = cur_acc_list[key], \
                            idx = idx, subdir = subdir, device = task_model_instances[key].device, key = key, save = save)
    if idx > 0:
        print(f'cur_acc_sum is {cur_acc_list}, best_acc_sum is {best_acc_sum_for_all_task}')

    if cur_acc_sum > best_acc_sum_for_all_task and idx >= 0:
        with torch.no_grad():
            for key in task_model_instances:
                task_model_instances[key].test_acc = cur_acc_list[key]
                copy_param_val(task_model_instances[key].model, params = weight_group, rand_perm = task_model_instances[key].rand_perm, wp_start_pos = task_model_instances[key].wp_start_pos)
                # for name, param in task_model_instances[key].model.named_parameters():
                #     print(param.view(-1)[:10])
                #     print(task_model_instances[key].wp_start_pos)
                #     break
                save_models(model = task_model_instances[key].model, acc = cur_acc_list[key], \
                            idx = idx, subdir = subdir, device = task_model_instances[key].device, key = key, save = save)
                best_acc_sum_for_all_task = cur_acc_sum
            weight_group.save(path = f'{subdir}/weight_group.pth')
        print(f'Update cur_best_acc_sum to {best_acc_sum_for_all_task}: {[task_model_instances[key].test_acc for key in task_model_instances]}')
    return best_acc_sum_for_all_task


def main(args = None):
    global task_model_instances
    device = torch.device(f'cuda:{args.which_cuda}')
    task_model_instances = get_task_model_instances(args)
    print_task_model_instances_info(task_model_instances)
    best_acc_sum_for_all_task = -1000
    ## The path to save loss list and the weight pool with largest acc sum
    subdir = f'./checkpoint/{get_trial_name(WEIGHT_GROUP_TABLE, task_model_instances, args)}' \
            if args.test_path is None else args.test_path
    print(f'save pth is: {subdir}')
    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    if args.test is True:
        weight_group_ = torch.load(f'{subdir}/weight_group.pth', map_location=torch.device('cpu'))
        ## defense
        weight_group_ = defense_to_weight_pool(task_model_instances, weight_group_, args)
        weight_group = WeightGroup()
        for key in weight_group_:
            weight_group.set(key, weight_group_[key])
        get_buffer_param(task_model_instances, weight_group, use_test_dataloader=False)
        test(-1, weight_group, best_acc_sum_for_all_task, args.save_model,subdir=subdir)
        return
    else:
        setup_seed(args.rand_seed_for_weight)
        weight_group = get_param_group(device, args.oral_init, args.pretrained_wp)
        if args.pretrained_wp is True: ## get the buffers from train_loader
            test(-1, weight_group, best_acc_sum_for_all_task, args.save_model,subdir=subdir)
        task_model_all_keys = [k for k in task_model_instances]
        task_model_num = len(task_model_instances)
        selected_task_model_num = int(task_model_num * args.sample_ratio) 
        max_iter = int(args.max_iter / args.sample_ratio)
        for idx in range(max_iter):
            if args.train_mode == 'random':
                rand_selected_idx = torch.randperm(task_model_num)[:selected_task_model_num]
                selected_task_model_keys = [task_model_all_keys[s] for s in rand_selected_idx]
            else:
                selected_task_model_keys = task_model_all_keys 
            train(idx, max_iter, args.accum_mode, weight_group, selected_task_model_keys)
            if 'pmnist-fcn' not in TASK_MODEL_ENTRY:
                best_acc_sum_for_all_task = test(idx, weight_group, best_acc_sum_for_all_task, args.save_model, subdir=subdir)
            elif (idx + 1) % args.test_interval == 0:
                for pmnist_ in task_model_instances:
                    task_model_instances[pmnist_].scheduler_step()
                    task_model_instances[pmnist_].test_epoch += 1
                    task_model_instances[pmnist_].train_loss_list.append(task_model_instances[pmnist_].train_loss_for_one_epoch / task_model_instances[pmnist_].test_interval)
                    task_model_instances[pmnist_].train_loss_for_one_epoch = 0
        if 'pmnist-fcn' in TASK_MODEL_ENTRY:
            best_acc_sum_for_all_task = test(args.test_interval-1, weight_group, best_acc_sum_for_all_task, args.save_model, subdir=subdir)
        else:
            print_task_model_instances_info(task_model_instances)
        print_weight_group_info(WEIGHT_GROUP_TABLE, weight_group)

if __name__ == '__main__':
    print(args)
    main(args)
    print(args)