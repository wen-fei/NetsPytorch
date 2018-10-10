#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:TenYun
@time: 2018/10/09
@contact: tenyun.zhang.cs@gmail.com
"""

import visdom
import time
import numpy as np


class Visualizer(object):
    """
    封装了visdom的基本操作
    """
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 保存('loss', 23) 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        :param env: 环境名称
        :param kwargs: 参数
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次绘制多个
        :param d: dict (name, value)
        """
        for k, v in d.items():
            self.plot(k, v)

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        # visdom接收numpy或者tensor数据，所以需要进转换
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_img', t.Tensor(3, 64, 64))
        self.img('input_img', t.Tensor(100, 164, 64))
        self.img('input_img', t.Tensor(100, 3, 64, 64), nrows=10)
        !!! don't
        --self.img('input_img', t.Tensor(100, 64, 64), nrows=10)--
        !!!
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        :param info: 日志信息
        :param win: 日志panel
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info
        ))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        self.function 等价于 self.vis.function
        自定义的plot、log、plot_many等除外
        """
        return getattr(self.vis, name)
