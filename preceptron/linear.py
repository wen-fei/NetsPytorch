#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:TenYun
@time: 2019/3/1
@contact: tenyun.zhang.cs@gmail.com
"""
from perceptron import Perceptron
# 定义激活函数
f = lambda x: x


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        """初始化"""
        Perceptron.__init__(self, input_num, f)

