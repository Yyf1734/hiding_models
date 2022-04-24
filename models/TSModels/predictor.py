import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math

class MLPPredictor(torch.nn.Module):
    def __init__(self, config):
        super(MLPPredictor, self).__init__()

        self.class_num = config.class_num
        
        self.hidden_size = config.hidden_size
        self.wo = torch.nn.Parameter(torch.zeros(self.hidden_size, self.class_num))
        self.bo = torch.nn.Parameter(torch.zeros(self.class_num))
        
        torch.nn.init.xavier_uniform_(self.wo)
        
    def init_params(self):
        pass
    
    def forward(self, h):
        o = torch.einsum('bk,kj->bj', (h, self.wo)) + self.bo
        
        return o

class RNNPredictor(torch.nn.Module):
    def __init__(self, config):
        super(RNNPredictor, self).__init__()

        self.class_num = config.class_num
        
        self.hidden_size = config.hidden_size
        self.weight = torch.nn.Parameter(torch.zeros(self.hidden_size, self.class_num))
        self.bias = torch.nn.Parameter(torch.zeros(self.class_num))
        
        torch.nn.init.xavier_uniform_(self.weight)
        
    def init_params(self):
        pass
    
    def forward(self, h):
        h = h[:,-1,:].squeeze(dim=1)
        o = torch.einsum('bk,kj->bj', (h, self.weight)) + self.bias
        
        return o
    
class AttentionPredictor(torch.nn.Module):
    def __init__(self, config):
        super(AttentionPredictor, self).__init__()
        
        self.hidden_size = config.hidden_size
        self.class_num = config.class_num
        
        self.wo = torch.nn.Parameter(torch.zeros(self.hidden_size, self.class_num))
        self.bo = torch.nn.Parameter(torch.zeros(self.class_num))
        
        torch.nn.init.xavier_uniform_(self.wo)
        
    def init_params(self):
        pass
        
    def forward(self, h):
        s = h[:,-1,:].squeeze(dim=1)
        att = torch.einsum('btk,bk->bt', (h, s)) 
        # att /= math.sqrt(self.hidden_size)
        att = F.softmax(att, dim=-1)
        
        h = torch.einsum('btk,bt->bk', (h, att))
        o = torch.einsum('bk,kj->bj', (h, self.wo)) + self.bo
        
        return o

