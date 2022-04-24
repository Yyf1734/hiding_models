import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import OrderedDict
import random

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class FCN(nn.Module):
    def __init__(self, input_dim = 28*28, hidden_dims = [100, 100, 100], class_num = 10, bias = True, rand_seed_for_x = 0, rand_seed_for_y = 0):
        super(FCN, self).__init__()
        fc_layers = OrderedDict()
        hidden_dims = [input_dim] + hidden_dims
        for i, (dh_1, dh) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            fc_layers[f'Linear{i}'] = nn.Linear(dh_1, dh, bias)
            # fc_layers[f'bn{i}'] = nn.BatchNorm2d(dh)
            fc_layers[f'ReLU{i}']=nn.ReLU(inplace = True)
        fc_layers['classifier']=nn.Linear(hidden_dims[-1], class_num, bias)
        self.model = nn.Sequential(fc_layers)
        self.layer_num = len(hidden_dims) + 1
        self.rand_seed_for_x = rand_seed_for_x
        self.rand_seed_for_y = rand_seed_for_y
        self.class_num = class_num

    def forward(self, x):
        if(len(x.shape) > 2):
            x = x.view(-1, np.prod(x.shape[1:]))
        if self.rand_seed_for_x > 0:
            setup_seed(self.rand_seed_for_x)
            perm = torch.randperm(28*28)
            x = x[:,perm]
        x = self.model(x)
        return x
    
    def change_label(self, targets):
        if self.rand_seed_for_y > 0:
            setup_seed(self.rand_seed_for_y)
            perm = torch.randperm(self.class_num)
            for i in range(len(targets)):
                targets[i] = perm[targets[i]]
        return targets

class LR_FCN(nn.Module):
    def __init__(self, layers:list = [31, 15]):
        super(LR_FCN, self).__init__()
        assert len(layers) >= 1
        self.layer_num = len(layers)
        
        hidden_layers = []
        for i in range(self.layer_num):
            if i + 1 < self.layer_num:
                hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
                hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(layers[self.layer_num - 1], 1)

    def forward(self, x):
        output = x
        for i in range(len(self.hidden_layers)):
            output = self.hidden_layers[i](output)

        return self.output_layer(output)

if __name__ == '__main__':
    x = torch.randn((3, 28, 28))
    fcn = FCN(28*28, [128,400], 10)
    print(fcn.model)
    print(fcn(x))
    lr = LR_FCN()
    print(lr)