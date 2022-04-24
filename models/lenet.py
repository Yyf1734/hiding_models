from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .nnfunc import copy_param_val
class LeNet(nn.Module):
    def __init__(self, feat_dim = 400, out_dim = 10, **kwargs):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, bias = False)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias = False)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(feat_dim, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, out_dim)
        # conv = nn.Conv2d(6, 16, kernel_size=5,bias = True, stride = self.first_m.stride, padding = self.first_m.padding)
        # conv = conv.to(self.device)
        # conv.weight = Parameter(true_weight)
        # conv.bias = self.first_m.bias
        # x_2 = conv(x_1)

        
    def forward(self, x, params=None, **kwargs):
        if params is not None:
            copy_param_val(self, params, **kwargs)
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x