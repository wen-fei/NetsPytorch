#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:TenYun
@time: 2019/3/4
@contact: tenyun.zhang.cs@gmail.com
"""
import random


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        """init the conn, the init weight is a vary small random float"""
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        """calc the gradient """
        return self.gradient

    def get_gradient(self):
        """ get the current gradient """
        return self.gradient

    def update_weight(self, rate):
        """ update the weight by gradient descent """
        self.calc_gradient()
        self.weight + rate * self.gradient

    def __str__(self):
        """ print the conn info """
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)


