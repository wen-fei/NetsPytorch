#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:TenYun
@time: 2019/3/4
@contact: tenyun.zhang.cs@gmail.com
"""
from bp.ConstNode import ConstNode
from bp.Node import Node


class Layer(object):
    """ init one layer """
    def __init__(self, layer_index, node_count):
        """
        :param layer_index: layer index
        :param node_count:  node numbers of a layer
        """
        self.layer_index = layer_index
        self.nodes= []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        """ set the output of a layer"""
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        """ calc the layer output vector """
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        """print the layer info"""
        for node in self.nodes:
            print(node)

