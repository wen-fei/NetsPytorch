#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:TenYun
@time: 2019/3/4
@contact: tenyun.zhang.cs@gmail.com
"""
from bp.Connection import Connection
from bp.Connections import Connections
from bp.Layer import Layer


class Network(object):
    def __init__(self, layers):
        """ init a full-connected network"""
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer+1].nodes[:-1]]

    def train(self, labels, data_set, rate, iteration):
        """
        train this network
        :param labels:  array, the train_data labels
        :param data_set: two-dimensional array, the train_date feature
        """
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        """ train the network using one item """
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        """ calc every node delta"""
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        """ update the every conn weight """
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.upstream_weight(rate)

    def calc_gradient(self):
        """calc every conn gradient"""
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        """get the gradient on a conn when calc one item in the network"""
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        """get the predict value """
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))

    def dump(self):
        """ print the network info"""
        for layer in self.layers:
            layer.dump()
