import torch
import torch.nn as nn
from typing import OrderedDict

class generator(nn.Module):
    def __init__(self,nc=3, nz=100, ngf=64):
        super().__init__()
        self.main = nn.Sequential(OrderedDict([
                # input is Z, going into a convolution
                ('conv1',nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False)),
                ('bn1',nn.BatchNorm2d(ngf * 8)),
                ('relu1',nn.ReLU(True)),
                # state size. (ngf*8) x 4 x 4
                ('conv2',nn.ConvTranspose2d(ngf * 8,ngf * 4, 4, 2, 1, bias=False)),
                ('bn2',nn.BatchNorm2d(ngf * 4)),
                ('relu2',nn.ReLU(True)),
                # state size. (ngf*4) x 8 x 8
                ('conv3',nn.ConvTranspose2d(ngf * 4,ngf * 2, 4, 2, 1, bias=False)),
                ('bn3',nn.BatchNorm2d(ngf * 2)),
                ('relu3',nn.ReLU(True)),
                # state size. (ngf*2) x 16 x 16
                ('conv4',nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)),
                ('bn4',nn.BatchNorm2d(ngf)),
                ('relu4',nn.ReLU(True)),
                ('conv5',nn.ConvTranspose2d( ngf, nc, kernel_size=1, stride=1, padding=0, bias=False)),
                ('tanh5',nn.Tanh())
                ]))
        
    def forward(self,x,params = None,**kwargs):
        if params is not None:
            print('dcgan_wrong')
            # copy_param_val(self, params, **kwargs)
        x = self.main(x)
        return x

class discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.main(x)
        x = x.view(-1,1).squeeze(1)
        return x

class generator_without_bn(nn.Module):
    def __init__(self,nc=3, nz=100, ngf=64):
        super().__init__()
        self.main = nn.Sequential(OrderedDict([
                # input is Z, going into a convolution
                ('conv1',nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False)),
                # ('bn1',nn.BatchNorm2d(ngf * 8)),
                ('relu1',nn.ReLU(True)),
                # state size. (ngf*8) x 4 x 4
                ('conv2',nn.ConvTranspose2d(ngf * 8,ngf * 4, 4, 2, 1, bias=False)),
                # ('bn2',nn.BatchNorm2d(ngf * 4)),
                ('relu2',nn.ReLU(True)),
                # state size. (ngf*4) x 8 x 8
                ('conv3',nn.ConvTranspose2d(ngf * 4,ngf * 2, 4, 2, 1, bias=False)),
                # ('bn3',nn.BatchNorm2d(ngf * 2)),
                ('relu3',nn.ReLU(True)),
                # state size. (ngf*2) x 16 x 16
                ('conv4',nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)),
                # ('bn4',nn.BatchNorm2d(ngf)),
                ('relu4',nn.ReLU(True)),
                ('conv5',nn.ConvTranspose2d( ngf, nc, kernel_size=1, stride=1, padding=0, bias=False)),
                ('tanh5',nn.Tanh())
                ]))
        
    def forward(self,x,params = None,**kwargs):
        if params is not None:
            print('dcgan_wrong')
            # copy_param_val(self, params, **kwargs)
        x = self.main(x)
        return x

class voiceG(nn.Module):
    def __init__(self,nz=100,nc=1):
        super().__init__()
        self.main = nn.Sequential(OrderedDict([
            # input (100,1)
            ('conv1',nn.ConvTranspose1d(nz, 64, 3, 1 , 0 , bias=False)),
            ('bn1',nn.BatchNorm1d(64)),
            ('relu1',nn.ReLU(True)),
            # (64,3)

            ('conv2',nn.ConvTranspose1d(64, 64, 27, 3 , 0,bias=False, dilation=2)),
            ('bn2',nn.BatchNorm1d(64)),
            ('relu2',nn.ReLU(True)),
            # (64,59)

            ('conv3',nn.ConvTranspose1d(64, 32, 38, 3 , 0,bias=False, dilation=2)),
            ('bn3',nn.BatchNorm1d(32)),
            ('relu3',nn.ReLU(True)),
            # (32,249)

            ('conv4',nn.ConvTranspose1d(32, 32, 52, 3 , 2,bias=False, dilation=5)),
            ('bn4',nn.BatchNorm1d(32)),
            ('relu4',nn.ReLU(True)),
            # (32,996)
 
            ('conv5',nn.ConvTranspose1d(32, nc, 80, 16 , 0,bias=False)),
            ('tanh',nn.Tanh())
            # (1,16000)
            ])
        )

    def forward(self,x, **kwargs):
        x = self.main(x)
        return x