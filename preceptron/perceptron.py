#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:TenYun
@time: 2018/10/09
@contact: tenyun.zhang.cs@gmail.com
"""
import numpy as np


class Perceptron(object):

    def __init__(self, input_num, activator):
        """
        初始化感知机，设置输入参数的个数，以及激活函数
        :param input_num:
        :param activator:
        """
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        """
        打印学习到的权重、偏置项
        :return:
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        """
        输入向量，输出感知机的计算结果
        :param input_vec:
        :return:
        """
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        m = np.array(input_vec) * np.array(self.weights)
        res = m.sum()
        return self.activator(res) + self.bias

    def train(self, input_vec, labels, iteration, rate):
        """
        输入训练数据
        :param input_vec:
        :param labels: 与每个向量对应的label
        :param iteration:  训练轮数
        :param rate: 学习率
        """
        for i in range(iteration):
            self._one_iterator(input_vec, labels, rate)

    def _one_iterator(self, input_vecs, labels, rate):
        """
        一次迭代，把所有数组计算一遍
        """
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            # 计算输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        """
        按照感知机规则更新权重
        """
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        self.weights = list(map(
            lambda t: t[0] + rate * delta * t[0],
            zip(input_vec, self.weights)
        ))
        # 更新bias
        self.bias += rate * delta
