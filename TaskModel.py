import os
from torch.nn.functional import normalize
import torchvision.utils as vutils
import numpy as np
import torch
from utils import progress_bar_local, print_banner, setup_seed, plot_wave
from models.dcgan import discriminator,generator
from models.nnfunc import param_name_to_key
from torchvision import transforms as transforms
from dataset import PairImage, PairVoice
import torchaudio
import matplotlib.pyplot as plt

class WeightGroup(object):
    ''' Privacy variable  __weight_group'''
    __weight_group = dict()
    
    @classmethod
    def get(self, key):
        return self.__weight_group[key]
    @classmethod
    def to_dict(self):
        return {key:self.__weight_group[key] for key in self.__weight_group}
    @classmethod
    def get_keys(self):
        return self.__weight_group.keys()
    @classmethod
    def set(self, key, v):
        self.__weight_group[key] = v
    @classmethod
    def add(self, key, v):
        self.__weight_group[key] = self.__weight_group[key] + v
    @classmethod 
    def save(self, path):
        # for key in self.get_keys():
        torch.save(self.__weight_group, path)

    ## TODO: For buffers, what if more than two models have buffers?
    __buffers = None # Ground_truth
    __g_buffers = None # Generated 
    @classmethod
    def set_buffers(self, v):
        self.__buffers = v
    @classmethod
    def get_buffers(self):
        return self.__buffers
    @classmethod
    def set_g_buffers(self, v):
        self.__g_buffers = v
    def get_g_buffers(self):
        return self.__g_buffers

def check_model(task_name, model):
    sum = 0
    ssum = 0
    for name, param in model.named_parameters():
        # print(name, np.prod(param.shape))
        ssum += np.prod(param.shape)
    # if "ts" in task_name:
    #     print('Models from ts task.')
    #     return 
    for name, m in model.named_modules():
        if not hasattr(m, 'weight') and (not hasattr(m, 'bias')): # or type(m.bias) is bool) and not hasattr(m, 'weight_ih_l0'):
            # print(type(m), ' has no weight or bias.')
            continue
        elif type(m) in [torch.nn.modules.batchnorm.BatchNorm2d,torch.nn.modules.batchnorm.BatchNorm1d]:
            weight_len = np.prod(m.weight.shape) if hasattr(m, 'weight') and m.weight is not None else 0
            bias_len = np.prod(m.bias.shape) if hasattr(m, 'bias') and m.bias is not None else 0
        elif hasattr(m, 'weight_ih_l0') and type(m.bias) is bool:
            weight_len = np.prod(m.weight_ih_l0.shape) + np.prod(m.weight_hh_l0.shape)
            bias_len = np.prod(m.bias_ih_l0.shape) + np.prod(m.bias_hh_l0.shape)
        else:
            weight_len = np.prod(m.weight.shape) if hasattr(m, 'weight') and m.weight is not None else 0
            bias_len = np.prod(m.bias.shape) if (hasattr(m, 'bias') and m.bias is not None and type(m.bias) is not bool) else 0
        # print('hasattr', name, type(m), weight_len + bias_len)
        sum += (weight_len + bias_len)
    if sum != ssum:
        print(f'check {task_name}')
        print(f'#parameters from named_parameters is {ssum}')
        print(f'#parameters from named_modules is {sum}')
        print('***************The #parameters are different from named_parameters and named_modules.****************')
        exit()

def get_model_param_dict(task_name, model):
    param_num = dict()
    if "_ts" in task_name:
        param_num['normal'] = 0
        for name, param in model.named_parameters():
            param_num['normal'] += np.prod(param.shape)
        return param_num
    for name, module in model.named_modules():
        if not hasattr(module, 'weight') and (not hasattr(module, 'bias')):
            continue
        if hasattr(module, 'weight_ih_l0') and type(module.bias) is bool:
            num = np.prod(module.weight_ih_l0.shape)+np.prod(module.weight_hh_l0.shape)+np.prod(module.bias_ih_l0.shape)+np.prod(module.bias_hh_l0.shape)
        else:
            num = np.prod(module.weight.shape)
        weight_key = 'ones' if type(module) in [torch.nn.modules.batchnorm.BatchNorm2d,torch.nn.modules.batchnorm.BatchNorm1d] else 'normal'
        # print('dict: ',weight_key, num)
        param_num[weight_key] = num if weight_key not in param_num else param_num[weight_key] + num
        if hasattr(module, 'bias') and module.bias is not None and type(module.bias) is not bool:
            bias_key = 'zeros' if type(module) in [torch.nn.modules.batchnorm.BatchNorm2d,torch.nn.modules.batchnorm.BatchNorm1d] else 'normal'
            num = np.prod(module.bias.shape)
            param_num[bias_key] = num if bias_key not in param_num else param_num[bias_key] + num
    return param_num

class TaskModel():
    def __init__(self,key = None, dataloader_func = None,
                model = None, model_kwargs = None, train_one_batch_fn = None, optimizer_fn = None, optim_kwargs = None, 
                scheduler_fn = None, loss_fn = None, device = None, max_iter = None, weight_table = None, perm_key = None, 
                wp_start_pos = None, test_interval = None, keep_training = None, **kwargs):
        self.name = key
        self.device = device
        train_loader, test_loader = dataloader_func()
        self.train_loader = train_loader
        self.test_loader = test_loader
        ## Not instance from the beginning, should be initialized when the model has been called
        self.model = model(**model_kwargs)
        
        check_model(self.name, self.model)
        self.loss_fn = loss_fn
        ## To calculate the average train loss during training
        self.train_idx = 0
        self.train_loss_for_one_epoch = 0
        ## Use to print the info of test phase
        self.test_epoch = 0
        self.test_acc = 0
        for x, _ in self.train_loader:
            self.batch_size = x.shape[0]
            break
        self.train_batch_num = len(self.train_loader)
        self.test_batch_num = len(self.test_loader) if self.test_loader is not None else None
        self.train_dataloader_iter= iter(self.train_loader)
        self.train_one_batch = train_one_batch_fn
        self.keep_training = keep_training

        ## Use to plot the loss & acc curve
        self.train_loss_list = [] ## Error: Each element correponds to train loss for one batch rather than one epoch
        self.test_loss_list = [] 
        self.test_acc_list = []
        
        self.wp_start_pos = wp_start_pos
        ##  randperm
        self.param_num = get_model_param_dict(self.name, self.model)
        self.perm_key = perm_key ## 0 or others
        self.rand_perm = None if perm_key is 0 else dict()
        self.rand_perm_reverse = None if perm_key is 0 else dict()
        if perm_key != 0:
            ## TODO: time consuming
            setup_seed(self.perm_key)
            for key in self.param_num:
                perm_lens = min(self.param_num[key], weight_table[key][0])
                ## randperm first model_param_num 
                self.rand_perm[key] = torch.randperm(perm_lens)
                ## reverse rand_perm
                self.rand_perm_reverse[key] = self.rand_perm[key]
                for i in range(perm_lens):
                    self.rand_perm_reverse[key][self.rand_perm[key][i]] = i

        ## optimizer & scheduler
        self.max_iter = max_iter
        self.time_to_test = False
        self.test_interval = test_interval # self.train_batch_num
        self.T_max = self.max_iter / self.test_interval if scheduler_fn is not None else None
        self.optimizer_name = optimizer_fn.__name__
        if self.name != "seq2seq-rnn":
            parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = optimizer_fn(parameters, **optim_kwargs)
            self.scheduler = scheduler_fn(self.optimizer, T_max = self.T_max) if scheduler_fn is not None else None
        ## For seq2seq
        else:
            self.optimizer_encoder = optimizer_fn(self.model.encoder.parameters(), **optim_kwargs)
            self.scheduler_encoder = scheduler_fn(self.optimizer_encoder, T_max = self.T_max) if scheduler_fn is not None else None
            self.optimizer_decoder = optimizer_fn(self.model.decoder.parameters(), **optim_kwargs)
            self.scheduler_decoder = scheduler_fn(self.optimizer_decoder, T_max = self.T_max) if scheduler_fn is not None else None
            from dataset import prepareData
            input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
            self.input_lang = input_lang
            self.output_lang = output_lang
        try:
            self.origin_lr = optim_kwargs['lr']# self.optimizer.state_dict()['param_groups'][0]['lr']
        except:
            exit()

        # For buffers
        self.noise_k_to_buffers = 100 ## fixed
        self.fix_noise_to_buffers = None

        ## For GAN
        self.D = None
        self.optimizerD = None
        self.num_pair = 0
        self.fix_noise = None
        self.g_image_pair = None
        self.image_pair = None
        self.fake_image = None
        self.pair_size = 1
        self.noise_k_to_image = 100 ## fixed
        self.GAN_image_pair_loss_list = []
        self.g_image_list = []

        self.idx = 0 # training iter index

    def scheduler_step(self):
        if self.name == "seq2seq-rnn":
            self.scheduler_encoder.step()
            self.scheduler_decoder.step()
        elif self.scheduler is not None:
            self.scheduler.step()

    ## For GAN
    def further_init(self, pair_size = 0, verbose = True):
        if ('GAN' in self.name or 'onlyG' in self.name) and 'onlyG_shuffle' not in self.name:
            self.pair_size = pair_size
            self.num_pair = self.pair_size*self.batch_size
            fix_noise_list = []
            image_pair_list = []
            setup_seed(self.noise_k_to_image)
            for i,(data,_) in enumerate(self.train_loader):
                noise = torch.randn(128,100,1) if 'Speech' in self.name else torch.randn(128,100,1,1)
                fix_noise_list.append(noise)
                image_pair_list.append(data)
                if i+1>=pair_size:
                    break
            self.fix_noise = torch.stack(fix_noise_list)#.to(self.device)
            self.image_pair = torch.stack(image_pair_list)#.to(self.device)
            if verbose:
                print(f'fix noise:{self.fix_noise.shape}')
                print(f'image pair:{self.image_pair.shape}')
            if 'Speech' not in self.name:
                vutils.save_image(self.image_pair[0][0:64],\
                f'real_image_pair_{self.num_pair}.png',normalize=True)
            else:
                plot_wave(self.image_pair[0],f'real_voice_{self.name}_{self.num_pair}.png')
            if 'GAN' in self.name:
                self.D = discriminator()#.to(self.device)
                self.optimizerD = torch.optim.Adam(self.D.parameters(),lr=2e-4,betas=(0.5,0.999))
            if verbose:
                print("GAN related generation complete.")

        elif 'buffer' in self.name:
            setup_seed(self.noise_k_to_buffers)
            self.fix_noise_to_buffers = torch.randn(4,100,1,1)#,device=self.device)
            print('buffer noise set down')
            self.test_interval = 391

        # shuffle multiple-batch
        elif 'onlyG_shuffle' in self.name:
            dataset = PairImage(self,pair_size)
            if 'Speech' not in self.name:
                dataset = PairImage(self,pair_size) #TODO: PairVoice类生成
            else:
                dataset = PairVoice(self,pair_size)
            print(f'fix noise:{self.fix_noise.shape}')
            print(f'image pair:{self.image_pair.shape}')
            print(f'pair:{self.image_pair.shape}')
            print("GAN related generation complete.")
            # self.epoch_interval = epoch_interval
            self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers = 0)
            self.train_dataloader_iter = iter(self.train_loader)
            if 'Speech' not in self.name:
                vutils.save_image(self.image_pair[0:self.batch_size],\
                    f'real_image_pair_{self.name}_{self.num_pair}.png',normalize=True)
            else:
                plot_wave(self.image_pair[0],f'real_voice_{self.name}_{self.num_pair}.png')
        
    def get_fake_data(self, batch_idx = None):
        cifar_gan = self.model(self.fix_noise[batch_idx])
        return cifar_gan

    #TODO: save generated 1 voice waveform plot
    def save_GAN_voice(self, subdir=None, save_one=False):
        if not os.path.isdir(f'{subdir}/g_pair_voice'):
            os.mkdir(f'{subdir}/g_pair_voice')
        if not os.path.isdir(f'{subdir}/g_pair_voice/wav'):
            os.mkdir(f'{subdir}/g_pair_voice/wav')
        if not os.path.isdir(f'{subdir}/g_pair_voice/png'):
            os.mkdir(f'{subdir}/g_pair_voice/png')
        waveform = self.g_image_pair[0].detach()
        plot_wave(waveform,f'{subdir}/g_pair_voice/{self.test_epoch}.png')
        self.test_epoch += 1 #self.epoch_interval
        model = self.model
        model.eval()
        idx = 1
        for i,(noise,voice) in enumerate(self.train_loader):
            noise = noise.to(self.device)
            pair_voice = voice.to(self.device)
            generated_voice = model(noise)
            for j in range(self.batch_size):
                torchaudio.save(f'{subdir}/g_pair_voice/wav/{idx}.wav',generated_voice[j].cpu(),16000)
                torchaudio.save(f'{subdir}/g_pair_voice/wav/{idx}_gt.wav',pair_voice[j].cpu(),16000)
                plot_wave(generated_voice[j].detach(),f'{subdir}/g_pair_voice/png/{idx}.png')
                plot_wave(pair_voice[j].detach(),f'{subdir}/g_pair_voice/png/{idx}_gt.png')
                idx += 1
                if save_one:
                    return

    def save_GAN_image(self, subdir = ''):
        if not 'onlyG' in self.name:
            if not os.path.isdir(f'{subdir}/g_fake_img'):
                os.mkdir(f'{subdir}/g_fake_img')
            vutils.save_image(self.fake_image[0:64],f'{subdir}/g_fake_img/{self.test_epoch}_{self.num_pair}_{self.name}.png',normalize=True)
        if not os.path.isdir(f'{subdir}/g_pair_img'):
            os.mkdir(f'{subdir}/g_pair_img')
        ## TODO: torch.normalize is not 0.5
        vutils.save_image(self.g_image_pair[0:64],f'{subdir}/g_pair_img/{self.test_epoch}_{self.num_pair}_{self.name}.png',normalize=True)
        np.save(f'{subdir}/g_pair_img/{self.num_pair}.npy',np.array(self.GAN_image_pair_loss_list))
        self.test_epoch += 1

if __name__ == '__main__':
    pass