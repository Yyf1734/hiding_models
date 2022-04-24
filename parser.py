import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(description="model_overloading_main")
    parser.add_argument('--which_cuda', default=0, type=int, help='the index of cuda to use')
    parser.add_argument('--max_iter', default=78200, type=int, help='the training epoch max iter')
    parser.add_argument('--oral_init', action='store_true', help='whether use oral initialization')
    parser.add_argument('--save_model', action='store_true', help='whether to save each model')
    parser.add_argument('--test', action='store_true', help='whether to test the saved weight_group')
    parser.add_argument('--pair_size',default=80,type=int,help='the number of batches used to map in GAN')
    parser.add_argument('--accum_mode', default='in_order', type=str, help='How to accuamulate the local grads')
    parser.add_argument('--train_mode', default='all', type=str, help='whether randomly sample the task_model for one epoch, all/random')
    parser.add_argument('--sample_ratio', default=1, type=float, help='the ratio of task models sampled in one epoch')
    parser.add_argument('--rand_perm', default=0, type=int, help='whether rand perm the weight pool for each task (0 or others)')
    parser.add_argument('--rand_seed_for_weight', default=100, type=int, help='seed to initialize weight group')
    parser.add_argument('--pmnist_num', default=10, type=int, help='the number of pmnist tasks')
    parser.add_argument('--shuffle_mnist', action='store_true', help='whether to shuffle the mnist data')
    parser.add_argument('--change_label', action='store_true', help='whether to change label in pmnist')
    parser.add_argument('--same_arch', action='store_true', help='whether to test in same arch')
    parser.add_argument('--keep_training', default=1, type=int, help='the iters for one task instance continuously trainging')
    parser.add_argument('--different_start_pos', action='store_true', help='whether to fetch weight from different position in weight pool')
    parser.add_argument('--pretrained_wp', action='store_true', help='whether to fetch weight from a pretrained weight pool')
    parser.add_argument('--save_path', default=None, type=str, help='the weight pool checkpoint save path')
    parser.add_argument('--test_interval', default=391, type=int, help='the test interval to test each model; for schedule.step')
    parser.add_argument('--test_path', default=None, type=str, help='the checkpoint path to test')
    parser.add_argument('--hidden_ratio', default=1, type=float, help='the ratio of params provided to the carrier model')
    parser.add_argument('--model_path', default=None, type=str, help='the model checkpoint save path')
    ## defense
    parser.add_argument('--defense_method', default=None, type=str, help='the defense methods include: quant, prune, noise, finetune')
    parser.add_argument('--noise_std', default=0.01, type=float, help='the std of noise')
    parser.add_argument('--prune_ratio', default=0, type=float, help='the ratio of params pruning')
    parser.add_argument('--filter_pruning', action='store_true', help='prune the filter or weight (i.e. FP or WP)')
    parser.add_argument('--recover_for_pruning', action='store_true', help='whether to recover nonzeros for pruning)') 
    parser.add_argument('--ft_index', default=-1, type=int, help='finetuning the carrier from the ft_index to the last layer')
    parser.add_argument('--ft_epochs', default=10, type=int, help='finetuning epochs')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.which_cuda}' ## Before import torch
    args.which_cuda=0
    return args