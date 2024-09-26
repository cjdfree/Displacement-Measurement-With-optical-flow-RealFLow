
# 此代码是在train.py的基础上修改得到的,逻辑一样,只是为了保存好原来的代码,所以重新写了一份
from __future__ import print_function, division
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from RAFT.core.raft import RAFT
import evaluate
import train_datasets
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

# 源码，后面对这些参数进行了整合
# # 最大光流大小、训练状态打印频率和验证频率
# # exclude extremly large displacements
# MAX_FLOW = 400
# SUM_FREQ = 10  # 训练状态打印的频率
# VAL_FREQ = 100


# 序列损失函数：计算光流预测与真实光流之间的损失
def sequence_loss(flow_preds, flow_gt, valid, max_flow, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    # 获取光流预测序列的长度
    n_predictions = len(flow_preds)
    # 初始化损失为0.0
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # 计算真实光流的模长
    valid = (valid >= 0.5) & (mag < max_flow)  # 通过阈值来排除无效像素和超出最大光流值的像素

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)  # 计算当前帧的权重，帧数越多，权重越大
        i_loss = (flow_preds[i] - flow_gt).abs()  # 计算当前帧和真实光流之间的绝对误差
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()  # 将当前帧的误差 * 权重并累加到flow_loss中

    # EPE：光流评价指标，也就是所有的光流预测值与真实值的平方和开平方
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]  # 仅保留有效像素处的端点误差

    # 'epe': 平均端点误差
    # '1px': 误差小于1像素的比例
    # '3px': 误差小于3像素的比例
    # '5px': 误差小于5像素的比例
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
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    # 使用一周期学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100, pct_start=0.05,
                                              cycle_momentum=False, anneal_strategy='linear')
    # 返回优化器和学习率调度器
    return optimizer, scheduler


# 记录训练过程中的指标和状态
class Logger:
    def __init__(self, model, scheduler, train_summary_frequency):
        self.model = model
        self.scheduler = scheduler
        self.train_summary_frequency = train_summary_frequency
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    # 打印训练状态
    def _print_training_status(self):
        # 显示多少次训练结果的平均值

        # 源码
        # metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        # training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        # metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        for key, value in self.running_loss.items():
            self.running_loss[key] = self.running_loss[key] / self.train_summary_frequency
        print(f'training_status: total_step: {self.total_steps}, last_learning_rate: {self.scheduler.get_last_lr()[0]}')
        print(f"metrics_status: EPE: {self.running_loss['epe']:.4f}, 1px: {self.running_loss['1px']:.6f}, 3px: {self.running_loss['3px']:.6f}, 5px: {self.running_loss['5px']:.6f}")

        # 如果没有writer，则先创建一个
        if self.writer is None:
            self.writer = SummaryWriter()

        # 每次写入writer，并清空running_loss
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / self.train_summary_frequency, self.total_steps)
            self.running_loss[k] = 0.0

    # 向日志记录器推送训练指标
    def push(self, metrics):

        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            # 用metrics写到对象的running_loss属性中
            self.running_loss[key] += metrics[key]

        if self.total_steps % self.train_summary_frequency == self.train_summary_frequency - 1:
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


def train(args, generated_dataset_path, checkpoints_path_each_round,
          train_result_path_each_round, validation_results_path_each_round, flag, data_type, scene,
          max_flow, train_summary_frequency, validation_frequency):
    # nn.DataParallel 用于并行化训练
    # Parameter Count 模型参数
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    # 加载模型：暂时不加载参数，直接从头训练
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    # 把模型放到cuda上面，同时开始训练
    model.cuda()
    model.train()

    # 如果不是在chairs上训练的，那就冻结一部分参数
    if args.stage != 'chairs':
        model.module.freeze_bn()

    # 加载数据集，主要实现逻辑在dataset.py里面
    train_loader = train_datasets.fetch_dataloader(args, generated_dataset_path, flag, data_type, scene)

    # 优化器
    optimizer, scheduler = fetch_optimizer(args, model)

    # 初始化训练步数、梯度缩放器、日志记录器等
    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, train_summary_frequency)

    # 测试频率，是否添加噪声，修改了源码，整合了
    # VAL_FREQ = 100
    # add_noise = True

    # 是否保持训练
    should_keep_training = True
    while should_keep_training:

        # i_batch: 迭代次数
        # data_blob: 数据，循环里面是一个batch_size的数据
        for i_batch, data_blob in tqdm(enumerate(train_loader)):
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
            # loss: 所有损失的总和
            # metrics: {'epe': x.xx, '1px': x.xx, '3px', x.xx, '5px': x.xx}
            loss, metrics = sequence_loss(flow_predictions, flow, valid, max_flow, args.gamma)

            # 自己保存metrics到文件
            file_path = os.path.join(train_result_path_each_round, f'{total_steps + 1}_metrics.json')

            # 将数据写入 JSON 文件
            with open(file_path, "w") as json_file:
                json.dump(metrics, json_file)
                print(f"JSON 文件已保存至{file_path}")

            # 反向传播和参数更新
            scaler.scale(loss).backward()  # scale：梯度缩放器，反向传播
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # 对梯度进行裁剪，防止梯度爆炸

            scaler.step(optimizer)  # 优化器参数更新
            scheduler.step()  # 更新学习率
            scaler.update()  # 更新梯度缩放器的缩放因子

            # 释放未使用的CUDA内存
            torch.cuda.empty_cache()

            # 记录日志
            logger.push(metrics)

            # 验证validate，设置了验证频率VAL_FREQ，前面是每5 step验证一次
            if total_steps % validation_frequency == validation_frequency - 1:

                PATH = os.path.join(checkpoints_path_each_round, f'{total_steps + 1}_{args.name}.pth')
                torch.save(model.state_dict(), PATH)

                results = {}

                # 对于不同的数据集，选择不同的验证函数
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        # FlyingChairs(test)
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        # sintel(train)
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        # KITTI-2015 (train)
                        results.update(evaluate.validate_kitti(model.module))
                    elif val_dataset == 'steel_ruler':
                        # 自己的测试函数：train & test
                        results.update(evaluate.validate_steel_ruler(model.module, flag, data_type, scene, generated_dataset_path, generated_dataset_path))
                    elif val_dataset == 'five_floors_frameworks':
                        # 自己的测试函数：train & test
                        results.update(evaluate.validate_five_floors_frameworks(model.module, flag, data_type, scene, generated_dataset_path))

                # 写入日志当中
                logger.write_dict(results)

                # 自己保存metrics到文件
                file_path = os.path.join(validation_results_path_each_round, f'{total_steps + 1}_metrics.json')

                # 将数据写入 JSON 文件
                with open(file_path, "w") as json_file:
                    json.dump(metrics, json_file)
                    print(f"JSON 文件已保存至{file_path}")

                # 将模型设置回训练模式，并在 FlyingChairs 以外的数据集上冻结 BatchNorm 层
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()

            total_steps += 1

            # 训练步数达到预设的步数，就退出训练
            if total_steps > args.num_steps:
                should_keep_training = False
                break

    # 退出训练的循环后，关闭日志记录器，保存最终的模型参数，并返回保存的模型路径
    logger.close()

    PATH = os.path.join(checkpoints_path_each_round, f'{args.name}.pth')
    torch.save(model.state_dict(), PATH)

    # 释放不用的内存
    torch.cuda.empty_cache()

    return PATH



