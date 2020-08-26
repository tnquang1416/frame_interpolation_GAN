'''
Created on Apr 10, 2020

@author: lab1323pc
'''

import torch
import torch.nn as nn
import os


class DisCriminatorWithCondition_V2(nn.Module):

    def __init__(self, opt):
        super(DisCriminatorWithCondition_V2, self).__init__()
        self.nfg = 64  # the size of feature map
        self.c = opt.channels  # output channel
        self.model_path = opt.default_model_path
        self.path = opt.path  # output folders
        
        self.conv_blocks = nn.Sequential(
            # input is c * 64 * 64
            nn.Conv2d(self.c, self.nfg, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # state: nfg * 32 * 32
            nn.Conv2d(self.nfg, self.nfg * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(self.nfg * 2, self.nfg * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(self.nfg * 4, self.nfg * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(self.nfg * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
            )
    
    def forward(self, data):
        return self.conv_blocks(data)
    
    def save(self):
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(self, self.model_path + "/final_gennet_dis_" + self.path + ".pt");
        return;
