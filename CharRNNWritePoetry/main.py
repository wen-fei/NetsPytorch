#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:TenYun
@time: 2018/10/09
@contact: tenyun.zhang.cs@gmail.com
"""
import sys

from config import Config
from dataLoader import get_data
import torch as t
import torch.nn as nn
from torch.autograd import Variable
from model import PoetryModel
from torchnet import meter
from tqdm import tqdm
from utils import Visualizer

opt = Config()


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)

    vis = Visualizer(env=opt.env)

    # 获取数据
    data, word2ix, ix2word = get_data(opt)
    data = t.from_numpy(data)
    dataloader = t.utils.data.DataLoader(
        data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=1
    )

    # 模型定义
    model = PoetryModel(len(word2ix), 128, 256)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))

    if opt.use_gpu:
        model.cuda()
    loss_meter = meter.AverageValueMeter()

    for epoch in range(opt.epoch):
        loss_meter.reset()
        for ii, data_ in tqdm(enumerate(dataloader)):
            # 训练
            data_ = data_.long().transpose(1, 0).contiguous()
            if opt.use_gpu:
                data_ = data_.cuda()
                optimizer.zero_grad()

                # 输入和目标错开
                input_, target = Variable(data_[:-1, :]), Variable(data_[1:, :])
                # target = target.cuda()
                output, _ = model(input_)
                loss = criterion(output, target.view(-1))
                loss.backward()
                optimizer.step()

                loss_meter.add(loss.data[0])

                # 可视化
                if (ii + 1) % opt.plot_every == 0:

                    # if os.path.exists(opt.debug_file):
                    #     ipdb.set_trace()
                    vis.plot('loss', loss_meter.value()[0])

                    # 诗歌原文
                    poetrys = [[ix2word[_word] for _word in data_[:, _iii]] for _iii in range(data_.size(1))][:16]
                    vis.text('<br>'.join([''.join(poetry) for poetry in poetrys]), win='origin_poem')

                    gen_poetries = []
                    # 分别以这几个字作为诗歌的第一个字，生成8首诗
                    for word in list('春江花月夜凉如水'):
                        gen_poetry = ''.join(generate(model, word, ix2word, word2ix))
                        gen_poetries.append(gen_poetry)
                    vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]), win='gen_poem')

        t.save(model.state_dict(), '%s_%s.path' % (opt.model_prefix, epoch))


def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """ 给定几个词，生成一首完整的诗歌 """

    results = list(start_words)  # 初始值，床前明月光
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    if opt.use_gpu:
        input = input.cuda()
    hidden = None

    # 用以控制生成诗歌的意境和长短（五言还是七言、四言等）
    if prefix_words:
        for word in prefix_words:
            input = Variable(input)
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        input = Variable(input)
        output, hidden = model(input, hidden)
        if i < start_word_len:
            # ‘窗前明月光’五个字依次作为输入，计算隐藏单元
            w = start_words[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            # 用预测的词作为新的输入，计算隐藏单元和预测新的输出
            top_index = output.data[0].topk(1)[1][0]
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results


def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    生成藏头诗
    :param start_words: 藏头诗字
    """
    results = []
    start_words_len = len(start_words)
    input = (t.Tensor([word2ix['<START>']]).view(1, 1).long())
    if opt.use_gpu:
        input = input.cuda()
    hidden = None

    index = 0  # 用来指示已经生成了多少句藏头诗
    pre_word = '<START>'  # 上一个词
    if prefix_words:
        for word in prefix_words:
            input = Variable(input)
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        input = Variable(input)
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0]
        w = ix2word[top_index]

        if pre_word in {u'。', u'！', '<START>'}:
            # 如果遇到句号、感叹号等，把藏头诗的词作为下一个句的输入
            if index == start_words_len:
                # 如果生成的诗歌已经包含全部藏头诗，则结束
                break
            else:
                # 把藏头诗的词作为输入，预测下一个词
                w = start_words[index]
                index += 1
                input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            # 把上一次预测的词作为输入，继续预测下一个词
            input = input.data.new([word2ix[w]]).view(1, 1)
        results.append(w)
        pre_word = w
    return results


def gen(**kwargs):
    """
    提供命令借口，用以生成相应的诗
    """
    for k, v in kwargs.items():
        setattr(opt, k, v)

    # 加载数据和模型
    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), 128, 256)
    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    if opt.use_gpu:
        model.cuda()
    # python2和python3 字符串兼容
    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            prefix_words = opt.prefix_words if opt.prefix_words else None
        else:
            start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode(
                'utf8') if opt.prefix_words else None
    else:
        start_words = opt.start_words.decode('utf8')
        prefix_words = opt.prefix_words.decode('utf8') if opt.prefix_words else None
    # 编码问题，半角改成全角，古诗中都是全角符号
    start_words = start_words.replace(',', u'，')\
        .replace('.', u'。')\
        .replace('?', u'？')

    # 判断是藏头诗还是普通诗
    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
    print(''.join(result))


if __name__ == '__main__':
    import fire
    fire.Fire()
    # train()