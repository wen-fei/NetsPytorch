#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:TenYun
@time: 2018/10/09
@contact: tenyun.zhang.cs@gmail.com
"""
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


# 数据加载
class DogCat(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        获取所有图片地址，并根据训练、验证、测试划分数据
        :param root: 路径
        :param transforms: 转换
        :param train: 训练
        :param test: 测试
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: data/test1.jpg
        # train: data/cat.100004.jps
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # 划分训练集、验证集，比例训练：验证= 7:3
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num)]

        # 数据转换操作，测试验证和训练的数据转换有所区别
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # 测试集和验证集
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor,
                    normalize
                ])
            # 训练集
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),  # 随机水平翻转
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        返回一张图片的数据
        如果是测试集，没有图片id，如果1000.jpg返回1000
        :param index:
        :return: data, label
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        """
        返回数据集中图片个数
        :return:
        """
        return len(self.imgs)
