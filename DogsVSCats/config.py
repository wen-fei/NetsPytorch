#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:TenYun
@time: 2018/10/09
@contact: tenyun.zhang.cs@gmail.com
"""
import warnings
import torch as t


# 配置文件
class DefaultConfig(object):
    env = 'default'         # visdom环境
    model = 'ResNet34'       # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = '/home/tenyun/Documents/data/DogsVSCats/train/'   # 训练数据存放路径
    test_data_root = '/home/tenyun/Documents/data/DogsVSCats/test1/'     # 测试数据存放路径
    # load_model_path = 'checkpoints/alexnet_1009_19:08:40.pth'   # 加载预训练模型的路径，为None代表不加载
    load_model_path = None

    batch_size = 16         # bacth size
    use_gpu = True          # use GPU or not
    num_workers = 4         # how many workers for loading data
    print_freq = 20         # print info every N batch

    debug_file = '/tmp/debug'   # if os.path.exists(debug_file) : enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = 0.001                # init learning rate
    lr_decay = 0.95         # when val_loss increase, lr = lr * lr_decay
    # weight_decay = 1e-4     # 损失函数
    weight_decay = 0

    def parse(self, kwargs):
        """
        根据字典kwargs更新config参数
        :param kwargs:
        :return:
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        # opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')
        # 打印配置信息
        print('user config: ')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()