#!/usr/bin/env python  
# -*- coding: utf-8 -*-

"""
@author: TenYun  
@contact: qq282699766@gmail.com  
@time: 2018/5/24 10:12 
"""

import torch
from torch import nn
from torch.autograd import Variable

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()  # b, 3, 32, 32
        layer1 = nn.Sequential()
        """
        Conv2d常用的参数有5个：
        @in_channels : 输入数据体的深度
        @out_channels : 输出数据体的深度
        @kernel_size : 滤波器（卷积核）大小，可以使用一个数字表示高和宽相同的卷积核，也可以使用不同的数字，如kernel_size=(3,2)
        @stride : 滑动的步长
        @padding: 四周进行填充的像素个数，0表示不填充
        @bias : 是否使用偏置
        @group
        @dilation 
        
        输出尺寸计算：
        W: 输入数据大小，F：感受野尺寸，S表示步长，P表示边界填充0的数量
        输出= (W-F+2P) / S + 1
        例如：输入7x7, 滤波器3x3, 步长1，填充0，那么
        （7-3+2X0） / 1 + 1 = 5, 即输出空间大小是5x5
        """
        layer1.add_module('conv1', nn.Conv2d(3, 32, 3, 1, padding=1))
        # (32 - 3 + 2X1) / 1 + 1 = 32
        # b, 32, 32, 32
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b, 32,w
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(32, 64, 3, 1, padding=1))
        # (32 - 3 + 2X1) / 1 + 1 = 64
        # b, 64, 16, 16
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))  # b, 64, 8, 8
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(64, 128, 3, 1, padding=1))
        # (64 - 3 + 2X1) / 1 + 1 = 128
        # b, 128, 8, 8
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))  # b, 128, 4, 4
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(2048, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc2', nn.Linear(512, 64))
        layer4.add_module('fc relu2', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(64, 10))
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        fc_input = conv3.view(conv3.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out

model = SimpleCNN()
# print(model)
# 提取需要的层
# new_model = nn.Sequential(*list(model.children())[:2])
# print(new_model)
# 提出所有的卷积层
# conv_model = SimpleCNN()
# for layer in model.named_modules():
#     if isinstance(layer[1], nn.Conv2d):
#         print(layer[0])
#         print('*'*10)
#         print(layer[1])
#         conv_model.add_module(name=layer[0].split('.')[1], module=layer[1])
# print(conv_model)

for param in model.named_parameters():
    print(param[0])

from torch.nn import init
# 权重是一个Variable，所以取出其data属性，进行处理即可
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        init.normal(m.weight.data)
        init.xavier_normal(m.weight.data)
        init.kaiming_normal(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m. nn.Linear):
        m.weight.data.normal_()