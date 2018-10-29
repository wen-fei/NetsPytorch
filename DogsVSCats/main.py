#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:TenYun
@time: 2018/10/09
@contact: tenyun.zhang.cs@gmail.com
"""
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchnet import meter
import models
from config import DefaultConfig
from utils.visualize import Visualizer
from data.dataset import DogCat
from tqdm import tqdm
opt = DefaultConfig()


def train(**kwargs):
    """ 训练 """

    # 根据命令行参数更新配置
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # step1 ： 模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2 : 数据
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # setp3 : 目标函数和优化器
    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=opt.weight_decay)

    # step4 : 统计指标，平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # step5 : 训练
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):

            # 训练模型参数
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # 更新统计指标以及可视化
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data, target.data)

            if (ii + 1) % opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])

                # 如果需要的话，进入debug模式
                # if os.path.exists(opt.debug_file):
                #     import ipdb;
                #     ipdb.set_trace()
        model.save()

        # 计算验证集上的指标以及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch : {epoch}, lr : {lr}, loss : {loss}, train_cm : {train_cm}, val_cm : {val_cm}".format(
            epoch  = epoch,
            loss   = loss_meter.value()[0],
            val_cm = str(val_cm.value()),
            train_cm = str(confusion_matrix.value()),
            lr = lr

        ))

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    """ 计算模型在验证集上的准确率等信息，用以辅助训练 """
    # 把模型设为验证模式
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in tqdm(enumerate(dataloader)):
        input, label = data
        val_input = Variable(input, volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    # 把模型改为训练模式
    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def test(**kwargs):
    """ test """
    opt.parse(kwargs)

    # 模型
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # 数据
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )
    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = Variable(data, volatile=True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        probability = softmax(score)[:, 1].data.tolist()
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
        results += batch_results
    write_csv(results, opt.result_file)
    return results


def help():
    """
    打印帮助信息
    python file.py help
    """

    print("""
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example:
        python {0} train --env='env0701' --lr=0.01
        python {0} test --dataset='path/to/dataset/root'
        python {0} help
    avaiable args:
    """.format(__file__))
    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


if __name__ == '__main__':
    # import fire
    # fire.Fire()
    train()