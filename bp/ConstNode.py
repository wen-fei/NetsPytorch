#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:TenYun
@time: 2019/3/4
@contact: tenyun.zhang.cs@gmail.com
"""
from functools import reduce


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        """ construct the node object """
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        """ append one conn to downstream """
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        """ according to (4) calc the delta when the node belong to hidden layer """
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream.delta * conn.weight,
            self.downstream, 0.0
        )
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        """ print the node info """
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str
