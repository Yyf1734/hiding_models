from collections import OrderedDict
from .nnfunc import copy_param_val


'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes = 10, add_bn = True):
        super(VGG, self).__init__()
        self.add_bn = add_bn
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, params=None, **kwargs):
        if params is not None:
            copy_param_val(self, params, **kwargs)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [(f'{i}_MaxPool2d',nn.MaxPool2d(kernel_size=2, stride=2))]
            elif self.add_bn is True:
                layers += [(f'{i}_conv2d', nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
                        (f'{i}_bn', nn.BatchNorm2d(x)),
                        (f'{i}_relu', nn.ReLU(inplace=True))]
                in_channels = x
            else:
                layers += [(f'{i}_conv2d', nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
                        (f'{i}_relu', nn.ReLU(inplace=True))]
                in_channels = x 
        layers += [('AvgPool2d', nn.AvgPool2d(kernel_size=1, stride=1))]
        return nn.Sequential(OrderedDict(layers))



def vgg11(num_classes=None):
    return VGG('VGG11', num_classes)

def vgg11_wo_bn(num_classes=None):
    return VGG('VGG11', num_classes, add_bn=False)

def vgg13(num_classes=None):
    return VGG('VGG13', num_classes)
    
def vgg16(num_classes=None):
    return VGG('VGG16', num_classes)

def vgg16_wo_bn(num_classes=None):
    return VGG('VGG16', num_classes, add_bn=False)

def vgg19(num_classes=None):
    return VGG('VGG19', num_classes)

def resnet18(num_classes=None):
    print('vgg')

class VGG_pytorch(nn.Module):
    def __init__(self, vgg_name, num_classes = 43, dropout = 0.5):
        super(VGG_pytorch, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            # nn.Dropout(p=dropout),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        # self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, params=None, **kwargs):
        if params is not None:
            copy_param_val(self, params, **kwargs)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [(f'{i}_MaxPool2d',nn.MaxPool2d(kernel_size=2, stride=2))]
            else:
                layers += [(f'{i}_conv2d', nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
                        (f'{i}_bn', nn.BatchNorm2d(x)),
                        (f'{i}_relu', nn.ReLU(inplace=True))]
                in_channels = x
        # layers += [('AvgPool2d', nn.AvgPool2d(kernel_size=1, stride=1))]
        layers += [('AvgPool2d', nn.AdaptiveAvgPool2d((7, 7)))]
        return nn.Sequential(OrderedDict(layers))

def vgg11_pytorch(num_classes=None):
    return VGG_pytorch('VGG11', num_classes)

def vgg13_pytorch(num_classes=None):
    return VGG_pytorch('VGG13', num_classes)
    
def vgg16_pytorch(num_classes=None):
    return VGG_pytorch('VGG16', num_classes)

def vgg19_pytorch(num_classes=None):
    return VGG_pytorch('VGG19', num_classes)