'''
Created on Apr 10, 2020

@author: Quang TRAN
'''

import torch
import torch.nn as nn
import os
import sys

from torch.nn import functional as F

import utils.utils as utils


class GeneratorWithCondition_NoNoise_V7(nn.Module):
    '''
    A generator without noise z
    '''

    def __init__(self, opt):
        super(GeneratorWithCondition_NoNoise_V7, self).__init__()
        self.nfg = 64  # the size of feature map
        self.c = opt.channels  # output channel
        self.model_path = opt.default_model_path
        self.path = opt.path  # output folders
        filter_size = 4
        stride_size = 2
        
        self.down_sample_blocks = nn.Sequential(
            nn.Conv2d(self.c * 2, self.nfg * 2, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(self.nfg * 2, self.nfg * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(self.nfg * 2, self.nfg * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            nn.BatchNorm2d(self.nfg * 4),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(self.nfg * 4, self.nfg * 8, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            nn.BatchNorm2d(self.nfg * 8),
            nn.LeakyReLU(0.02, inplace=True)
            )
        
        self.up_sample_block = nn.Sequential(
            nn.ConvTranspose2d(self.nfg * 8, self.nfg * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            nn.BatchNorm2d(self.nfg * 4),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(self.nfg * 4, self.nfg * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(self.nfg * 2, self.nfg, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            nn.BatchNorm2d(self.nfg),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(self.nfg, self.c, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.Tanh()
            )
    
    def forward(self, tensor0, tensor2):
        '''
        Forward method which do padding on input if the input frames are not in square size or size is not multiples of 32
        :param tensor0:
        :param tensor2:
        '''
        h0 = int(list(tensor0.size())[2])
        w0 = int(list(tensor0.size())[3])
        h2 = int(list(tensor2.size())[2])
        w2 = int(list(tensor2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if (h0 % 32 != 0 or (h0 - w0) < 0):
            pad_h = 32 - (h0 % 32) if (h0 - w0) >= 0 else 32 - (h0 % 32) + (w0 - h0)
            tensor0 = F.pad(tensor0, (0, 0, 0, pad_h))
            tensor2 = F.pad(tensor2, (0, 0, 0, pad_h))
            h_padded = True

        if (w0 % 32 != 0 or (h0 - w0) > 0):
            pad_w = 32 - (w0 % 32) if (h0 - w0) <= 0 else 32 - (h0 % 32) + (h0 - w0)
            tensor0 = F.pad(tensor0, (0, pad_w, 0, 0))
            tensor2 = F.pad(tensor2, (0, pad_w, 0, 0))
            w_padded = True
        
        out = torch.cat((tensor0, tensor2), 1)  # @UndefinedVariable
        
        out_down = self.down_sample_blocks(out)
        out_up = self.up_sample_block(out_down)
        
        if h_padded:
            out_up = out_up[:, :, 0:h0, :]
        if w_padded:
            out_up = out_up[:, :, :, 0:w0]
          
        return out_up