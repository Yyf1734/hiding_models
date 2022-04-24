import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math
from typing import OrderedDict

from .predictor import RNNPredictor, AttentionPredictor, MLPPredictor
from .encoder import RNNEncoder, TCNEncoder, MLPEncoder, TransformerEncoder, CNNEncoder
from .config import *


class TSClassifier(torch.nn.Module):
    def __init__(self, model_name = None, dataset_name = None):
        super(TSClassifier, self).__init__()
        config = get_config(model_name = model_name)
        config = get_dataset_config(dataset_name=dataset_name, config=config)
        self.encoder_type = config.encoder_type
        self.predictor_type = config.predictor_type
        
        self.encoder = None
        if self.encoder_type == 'rnn':
            self.encoder = RNNEncoder(config)
        if self.encoder_type == 'tcn':
            self.encoder = TCNEncoder(config)
        if self.encoder_type == 'mlp':
            self.encoder = MLPEncoder(config)
        if self.encoder_type == 'tfm':
            self.encoder = TransformerEncoder(config)
        if self.encoder_type == 'cnn':
            self.encoder = CNNEncoder(config)
            
        self.fc = None
        if self.predictor_type == 'rnn':
            self.fc = RNNPredictor(config)
        if self.predictor_type == 'attention':
            self.fc = AttentionPredictor(config)
        if self.predictor_type == 'mlp':
            self.fc = MLPPredictor(config)
            
    def init_params(self):
        self.encoder.init_params()
        self.fc.init_params()
        
    def forward(self, x):
        h = self.encoder(x)
        o = self.fc(h)
        return o

