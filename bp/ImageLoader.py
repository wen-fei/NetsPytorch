#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:TenYun
@time: 2019/3/4
@contact: tenyun.zhang.cs@gmail.com
"""
from bp.Loader import Loader


class ImageLoader(Loader):
    """ image data loader"""
    def get_picture(self, content, index):
        """ get image from file"""
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(self.to_int(content[start + i * 28  + j-1 : start + i * 28 + j]))
        return picture

    def get_one_sample(self, picture):
        """ transform the image to input date """
        samples = []
        for i in range(28):
            for j in range(28):
                samples.append(picture[i][j])
        return samples

    def load(self):
        """ load data and get all items vec"""
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(self.get_one_sample(self.get_picture(content, index)))
        return data_set
