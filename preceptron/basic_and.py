#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:TenYun
@time: 2019/3/1
@contact: tenyun.zhang.cs@gmail.com
"""
from perceptron import Perceptron


def f(x):
    """激活函数"""
    return 1 if x > 0 else 0


def get_training_dataset():
    """基于and真值表构建训练数据"""
    input_vecc = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vecc, labels


def train_and_preceptron():
    """使用and真值表训练感知机"""
    p = Perceptron(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 20, 0.1)
    return p


if __name__ == '__main__':
    and_preceptron = train_and_preceptron()
    print(and_preceptron)
    # 测试
    print('1 and 1 = %d' % and_preceptron.predict([1, 1]))
    print('0 and 0 = %d' % and_preceptron.predict([0, 0]))
    print('1 and 0 = %d' % and_preceptron.predict([1, 0]))
    print('0 and 1 = %d' % and_preceptron.predict([0, 1]))
