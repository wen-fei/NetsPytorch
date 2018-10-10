#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:TenYun
@time: 2018/10/09
@contact: tenyun.zhang.cs@gmail.com
"""
import os
import numpy as np


def get_data(opt):
    """
    :param opt: 配置选项，Config对象
    :return word2ix: dict,每个字对应的序号，形如'月'->100
    :return ix2word: dict,每个序号对应的字，形如'100'->'月'
    :return data: numpy数组，每一行是一首诗对应的字的下标
    """
    # opt = config.Config()
    if os.path.exists(opt.pickle_path):
        data = np.load(opt.pickle_path)
        data, word2ix, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()
        return data, word2ix, ix2word

    # 如果没有处理好的二进制文件，则处理原始的json文件
