from numpy import mod
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math

from torch import Tensor
from torch.nn import Module,MultiheadAttention,Dropout,Linear,ModuleList,LayerNorm
from torch.nn.init import xavier_uniform_

def get_rnn_weight_bias(rnn):
    weight = []
    bias = []
    for name, param in rnn.named_parameters():
        if 'weight_' in name:
            weight.append(param.data.view(-1))
        if 'bias_' in name:
            bias.append(param.data.view(-1))
    return torch.cat(weight), torch.cat(bias)

class MLPEncoder(torch.nn.Module):
    def __init__(self, config):
        super(MLPEncoder, self).__init__()
        
        self.max_len = config.max_len
        self.input_size = config.input_size
        self.out_size = config.hidden_size
        
        self.hidden_size = config.mlp_config.hidden_size
        self.activate_type = config.mlp_config.activate_type
        
        
        if self.activate_type == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        if self.activate_type == 'tanh':
            self.act = torch.nn.Tanh()
        if self.activate_type == 'relu':
            self.act = torch.nn.ReLU()
        
        self.hidden1 = torch.nn.Linear(self.max_len * self.input_size, self.hidden_size, bias=True)
        self.hidden2 = torch.nn.Linear(self.hidden_size, self.out_size, bias=True)
        
    def init_params(self):
        pass
        
    def forward(self, x):
        B = x.size(0)
        x_ = x.view(B, -1)
        h = self.hidden1(x_)
        
        if self.activate_type is not None:
            h = self.act(h)
        
        h = self.hidden2(h)
        
        return h

class RNNEncoder(torch.nn.Module):
    def __init__(self, config):
        super(RNNEncoder, self).__init__()
        
        self.hidden_size = config.hidden_size
        self.input_size = config.input_size
        
        self.rnn_type = config.rnn_config.rnn_type
        self.rnn_layer = config.rnn_config.rnn_layer
            
        if self.rnn_type == 'gru':
            self.rnn = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.rnn_layer, bias=True, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.rnn_layer, bias=True, batch_first=True)
        else:
            self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.rnn_layer, bias=True, batch_first=True)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if type(m) in [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):
        # x = x.unsqueeze(-1)
        h, _ = self.rnn(x, None)
        # self.weight, self.bias = get_rnn_weight_bias(self.rnn)
        return h

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def init_params(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(torch.nn.Module):
    def __init__(self, num_inputs, num_channels, config):
        super(TemporalConvNet, self).__init__()
        
        kernel_size = config.kernel_size
        dropout = config.dropout
        stride = config.stride
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if type(m) in [TemporalBlock]:
                m.init_params()
        
    def forward(self, x):
        return self.network(x)
    
class TCNEncoder(torch.nn.Module):
    def __init__(self, config):
        super(TCNEncoder, self).__init__()
        
        self.input_size = config.input_size
        
        self.hidden_sizes = config.tcn_config.hidden_sizes
        self.cnn = TemporalConvNet(num_inputs=self.input_size, num_channels=self.hidden_sizes, config=config.tcn_config)
        
        self.init_params()
        
        
    def init_params(self):
        self.cnn.init_params()
        
    def forward(self, x):
        #x_ = x.unsqueeze(1)
        
        x_ = x.transpose(2, 1)
        
        h = self.cnn(x_)
        
        h = h.transpose(2, 1)
        return h
    
class CNNEncoder(torch.nn.Module):
    def __init__(self, config):
        super(CNNEncoder, self).__init__()
        
        self.strides = config.cnn_config.strides
        self.kernel_sizes = config.cnn_config.kernel_sizes
        self.channels = config.cnn_config.channels
        
        # print(self.strides)
        # print(self.kernel_sizes)
        
        self.input_size = config.input_size
        
        self.cnn_layer = len(self.strides)
        
        last_channel = self.input_size
        for i in range(self.cnn_layer):
            cnn_name = 'cnn_%d' % (i)
            setattr(self, cnn_name, nn.Conv1d(last_channel, self.channels[i], self.kernel_sizes[i],
                                           stride=self.strides[i], padding=0, dilation=1))
            last_channel = self.channels[i]
            
        #self.init_params()
            
    def init_params(self):
        for i in range(self.cnn_layer):
            cnn_name = 'cnn_%d' % (i)
            cnn = getattr(self, cnn_name)
            torch.nn.init.xavier_uniform(cnn.weight)
    
    def forward(self, x):
        #h = x.unsqueeze(1)
        h = x.transpose(2, 1)
        # print(h.shape)
        
        for i in range(self.cnn_layer):
            cnn_name = 'cnn_%d' % (i)
            cnn = getattr(self, cnn_name)
            h = cnn(h)
            # print(h.shape)
        
        
        h = h.squeeze(dim=1)
        return h
            
    
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
    
class TransformerFirstEncoderLayer(Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerFirstEncoderLayer, self).__init__()
        self.linear1 = Linear(1, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(1)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerFirstEncoderLayer, self).__setstate__(state)

    def forward(self, src):
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class TransformerEncoder(torch.nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        
        self.hidden_size = config.hidden_size
        
        self.dropout = config.tfm_config.dropout
        self.nhead = config.tfm_config.nhead
        self.dim_ff = config.tfm_config.dim_ff
        self.activation = config.tfm_config.activation
        self.tfm_layer = config.tfm_config.tfm_layer
        
        for i in range(self.tfm_layer):
            tfm_name = 'transformer_%d' % (i)
            if i > 0:
                setattr(self, tfm_name, torch.nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.nhead, dim_feedforward=self.dim_ff, dropout=self.dropout, activation=self.activation))
            else:
                setattr(self, tfm_name, TransformerFirstEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.dim_ff, dropout=self.dropout, activation=self.activation))
    
    def init_params(self):
        pass
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        
        h = x.permute(1, 0, 2)
        
        for i in range(self.tfm_layer):
            tfm = getattr(self, 'transformer_%d'%(i))
            h = tfm(h)
        
        h = h.permute(1, 0, 2)
        return h