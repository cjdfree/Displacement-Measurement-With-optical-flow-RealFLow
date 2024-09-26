from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from raft import RAFT
import evaluate
import datasets
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# 最大光流大小、训练状态打印频率和验证频率
# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100  # 训练状态打印频率
VAL_FREQ = 5000


# 序列损失函数：计算光流预测与真实光流之间的损失
def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


# 计算模型的参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 创建优化器和学习率调度器
def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


# 记录训练过程中的指标和状态
class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    # 打印训练状态
    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        # 如果没有writer，则先创建一个
        if self.writer is None:
            self.writer = SummaryWriter()

        # 写入writer，并清空running_loss
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    # 向日志记录器推送训练指标
    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    # 将结果写入tensorboard中
    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    # nn.DataParallel 用于并行化训练
    # Parameter Count 模型参数
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    # 加载模型
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    # 数据集和优化器
    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    # 初始化训练步数、梯度缩放器、日志记录器等
    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    # 测试频率，是否添加噪声
    VAL_FREQ = 5000
    add_noise = True

    # 是否保持训练
    should_keep_training = True
    while should_keep_training:

        # 循环设置训练
        for i_batch, data_blob in enumerate(train_loader):
            # 优化器梯度清零
            optimizer.zero_grad()

            # 把数据放到GPU上面
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            # 添加噪声之后对图片进行的操作
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            # 前向传播：预测光流，设置了里面的迭代次数，前面默认值是12
            flow_predictions = model(image1, image2, iters=args.iters)            

            # 计算损失和指标
            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)

            # 反向传播和参数更新
            scaler.scale(loss).backward()  # scale：梯度缩放器，反向传播
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # 对梯度进行裁剪，防止梯度爆炸
            
            scaler.step(optimizer)  # 优化器参数更新
            scheduler.step()  # 更新学习率
            scaler.update()  # 更新梯度缩放器的缩放因子

            # 记录日志
            logger.push(metrics)

            # 验证validate，设置了验证频率VAL_FREQ，前面是每5000 step验证一次
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                # 每5000 step 保存一次模型参数
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}

                # 对于不同的数据集，选择不同的验证函数
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                # 写入日志当中
                logger.write_dict(results)

                # 将模型设置回训练模式，并在 FlyingChairs 以外的数据集上冻结 BatchNorm 层
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            # 训练步数达到预设的步数，就退出训练
            if total_steps > args.num_steps:
                should_keep_training = False
                break

    # 退出训练的循环后，关闭日志记录器，保存最终的模型参数，并返回保存的模型路径。
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")

    # 使用哪一个训练的数据集
    parser.add_argument('--stage', help="determines which dataset to use for training")

    # 恢复checkpoints，使用哪一个pth来训练
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    # 是否使用小模型。如果设置了该标志，模型将使用较小的参数规模
    parser.add_argument('--small', action='store_true', help='use small model')

    parser.add_argument('--validation', type=str, nargs='+')

    # 训练的学习率，总步数，batch_size，图像的尺寸，GPU数，是否使用混合精度训练
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # 迭代次数，默认为12，这是光流估计模型的迭代次数
    parser.add_argument('--iters', type=int, default=12)

    # 权重衰减，默认值为0.00005
    parser.add_argument('--wdecay', type=float, default=.00005)

    # Adam优化器的epsilon值，默认为1e-8
    parser.add_argument('--epsilon', type=float, default=1e-8)

    # clip: 梯度裁剪的阈值，默认为1.0
    # dropout: Dropout的概率，默认为0.0
    # gamma: 指数加权的参数，默认为0.8
    # add_noise: 是否向输入图像添加噪声
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    # 设置了随机种子，以确保实验的可重复性
    torch.manual_seed(1234)
    np.random.seed(1234)

    # 检查是否存在checkpoints文件夹，如果不存在则创建该文件夹
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # 训练入口
    train(args)
