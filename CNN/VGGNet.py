#!/usr/bin/env python  
# -*- coding: utf-8 -*-

"""
@author: TenYun  
@contact: qq282699766@gmail.com  
@time: 2018/5/24 14:10
VGGNet 使用了更小的滤波器，使用了更深的结构
AlexNet8层网络，VGGNet有16~19层网络
使用了3X3滤波器和2X2大池化层
层叠很小的滤波器的感受野和一个大的滤波器的感受野是相同的，还能减少参数，同时有更深的网络结构
只是对网络层的堆叠，没有太多创新
"""
import torch
from torch import nn
from torch.autograd import Variable


class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, padding=1),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096*4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 16 最后一层不要ReLU和Dropout
            nn.Linear(4096, num_classes),
        )
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x