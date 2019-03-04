#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:TenYun
@time: 2019/3/4
@contact: tenyun.zhang.cs@gmail.com
"""
import struct


class Loader(object):
    """ data loader class """
    def __init__(self, path, count):
        """
        init data loader
        :param path: data path
        :param count: items number
        """
        self.path = path
        self.count = count

    def get_file_content(self):
        """ read file data """
        with open(self.path, 'rb') as f:
            content = f.read()
        return content

    def to_int(self, byte):
        return struct.unpack('B', byte)[0]
