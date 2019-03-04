#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:TenYun
@time: 2019/3/4
@contact: tenyun.zhang.cs@gmail.com
"""
from functools import reduce


def gradient_check(network, sample_feature, sample_label):
    """
    gradient check
    :param network:  the network object
    :param sample_feature: items features
    :param sample_label: items lables
    :return:
    """
    # calc the network error
    network_error = lambda vec1, vec2: 0.5 * reduce(
        lambda a, b: a + b,
        list(map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                 zip(vec1, vec2)
                 ))
    )
    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)
    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()

        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon  # 刚才加过1次，这里需要减2倍
        error2 = network_error(network.predict(sample_feature), sample_label)

        # 计算期望梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)
        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))

