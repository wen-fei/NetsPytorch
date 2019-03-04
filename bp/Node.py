#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:TenYun
@time: 2019/3/4
@contact: tenyun.zhang.cs@gmail.com
"""
from functools import reduce


def sigmoid(output):
    pass


class Node(object):
    """ 节点类， 负责记录和维护节点自身信息以及与整个节点相关的上下游连接，实现输出值和误差项的计算"""

    def __init__(self, layer_index, nodex_index):
        """
        构造节点对象
        :param layer_index: 节点所属的层编号
        :param nodex_index: 节点的编号
        """
        self.layer_index = layer_index
        self.node_index = nodex_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        """设置节点的输出值，如果节点属于输入层会用到此函数"""
        self.output = output

    def append_downstream_connection(self, conn):
        """添加一个到下游节点的连接"""
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        """ append a con to upstream """

    def calc_output(self):
        """ according to (1) get the node output"""
        output = reduce(
            lambda ret, conn: ret + conn.upstream_node.output * conn.weight,
            self.upstream, 0.0
        )
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        """according to (4) get the delta when the nodes belong to hidden layer """
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0
        )

    def calc_output_layer_delta(self, label):
        """according to (3) get the delta when the nodes belong to output layer """
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        """print the node info """
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + "\n\t" + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream: ' + downstream_str + '\n\tupstream: ' + upstream_str
