import argparse
import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import shutil
import imageio
import pandas as pd
from scipy.signal import resample

from torchvision.utils import save_image

from RAFT.core.utils import flow_viz
from RAFT.core.utils.frame_utils import writeFlow
from RAFT.core.utils.utils import InputPadder
import render_datasets

import traceback

import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from RAFT.core.raft import RAFT
from function_decorator import print_function_name_decorator
from render_image_RealFLow import render_local
from finetune import *


class Experiment:

    @print_function_name_decorator
    def __init__(self, total_scenes, model_list, flag, round_num, total_steps,
                 max_flow, train_summary_frequency, validation_frequency):

        self.total_scenes = total_scenes  # 所有数据集的场景
        self.model_list = model_list  # 使用的模型
        self.flag = flag  # 帧不更新
        self.round_sum = round_num  # 总测试轮数
        self.total_steps = total_steps  # 迭代的步数
        self.max_flow = max_flow  # 最大光流
        self.train_summary_frequency = train_summary_frequency  # 训练状态打印频率
        self.validation_frequency = validation_frequency  # 验证频率

        self.file_types = ['generated_dataset', 'checkpoints', 'train_results',
                           'validation_results', 'train_metrics_img', 'validation_metrics_img',
                           'pre_trained_test', 'original_pre_trained_test']

        # 生成数据集和模型训练的配置参数: {'generated_dataset': args, 'finetune': args, 'pre_trained_test': args}
        self.config = {}

        # {'steel_ruler': {'Industrial_camera_0%': path, 'Industrial_camera_5%': path},
        #  'five_floors_frameworks': {'0-4_0%': path, '01-02_5%': path}}
        # 按照上面的格式来存路径
        self.original_dataset_dict = {}  # 原始数据集的字典，里面存放路径
        self.generated_dataset = {}  # 所有生成数据集汇总：存放都是路径
        self.checkpoints = {}  # 所有生成模型汇总: {}
        self.train_results = {}  # 所有训练结果的metrics的json文件路径
        self.validation_results = {}  # 所有测试结果的metrics的json文件路径
        self.metrics_dict = {}  # 读取出来的metrics的汇总字典
        self.train_metrics_img = {}  # 画图train_metrics的路径
        self.validation_metrics_img = {}  # 画图validation_metrics的路径
        self.pre_trained_test = {}  # 存放预训练模型再次预测的结果
        self.original_pre_trained_test = {}  # 存放预训练模型未经RealFlow训练来预测的结果

    @print_function_name_decorator
    def init_experiment(self):
        for model_name in self.model_list:
            self.make_dir(model_name)
            for file_type in self.file_types:
                # 创建属性的字典
                experiment_attribute = getattr(self, file_type)
                experiment_attribute[model_name] = {}
                for data_type, scenes in self.total_scenes.items():
                    experiment_attribute[model_name][data_type] = {}
                    for scene in scenes:
                        experiment_attribute[model_name][data_type][scene] = os.path.join(model_name, file_type, data_type, scene)
                        # 创建文件夹
                        self.make_dir(os.path.join(model_name, file_type))
                        self.make_dir(os.path.join(model_name, file_type, data_type))
                        self.make_dir(os.path.join(model_name, file_type, data_type, scene))

        for data_type, scenes in self.total_scenes.items():
            self.original_dataset_dict[data_type] = {}
            for scene in scenes:
                # 初始化字典，其中original_dataset_dict比较特殊，不同模型之间用的是一样的，所以没有model_name这个key
                self.original_dataset_dict[data_type][scene] = os.path.join(f'original_dataset', data_type, scene)

    @print_function_name_decorator
    def round_experiment(self):
        # 迭代循环的生成数据集，训练模型，以及模型重新在原本的任务上做直接预测
        for flag in self.flag:
            for model_name in self.model_list:
                for data_type, scenes in self.total_scenes.items():
                    for scene in scenes:
                        # 一个模型只需要raft-things预测一次，另外单独存放文件夹
                        # self.original_pre_trained_predict(flag, model_name, data_type, scene)

                        for round_num in tqdm(range(self.round_sum)):
                            # 渲染数据集
                            torch.cuda.empty_cache()
                            self.generate_dataset(round_num + 1, flag, model_name, data_type, scene,
                                                  self.original_dataset_dict[data_type][scene],
                                                  self.generated_dataset[model_name][data_type][scene],
                                                  self.checkpoints[model_name][data_type][scene])

                            # 训练模型
                            torch.cuda.empty_cache()
                            self.finetune_model(round_num + 1, flag, model_name, data_type, scene,
                                                self.generated_dataset[model_name][data_type][scene],
                                                self.checkpoints[model_name][data_type][scene],
                                                self.train_results[model_name][data_type][scene],
                                                self.validation_results[model_name][data_type][scene])

                            # 模型直接预测
                            torch.cuda.empty_cache()
                            self.finetune_pre_trained_predict(round_num + 1, flag, model_name, data_type, scene)

    @print_function_name_decorator
    def control(self):
        try:
            self.init_experiment()
            # self.round_experiment()
            # self.result_visualization()
            self.compare_original_and_finetune()
        except Exception as e:
            print("RealFlowOpticalFlow Failure!!!")
            print(e)
            traceback.print_exc()

    @print_function_name_decorator
    def make_dir(self, path):
        if not os.path.exists(path):
            # 如果文件路径不存在，则创建它
            os.makedirs(path)
            print(f"文件路径 {path} 不存在，已成功创建")
        else:
            print(f"文件路径 {path} 已存在")

    @print_function_name_decorator
    def generate_dataset(self, round_num, flag, model_name, data_type, scene, original_dataset_path,
                              generated_dataset_path, models_path):

        print(f'round: {round_num}, model: {model_name}, data_type: {data_type}, scene: {scene}'
              f'original_dataset: {original_dataset_path}, generated_dataset: {generated_dataset_path}, '
              f'checkpoints_path: {models_path}')

        # 创建数据集，并添加到整个列表中
        generated_dataset_obj = GeneratedDataset(generated_dataset_path)

        # 创建生成数据集模型的配置文件
        # 第一轮使用raft-things生成数据集，后面使用训练好的模型进行生成
        if round_num == 1:
            checkpoint_path_each_round = f'../RAFT/models/{model_name}.pth'
        else:
            checkpoint_path_each_round = os.path.join(models_path, f'round{round_num-1}', f'{model_name}_{data_type}_{scene}.pth')

        generated_dataset_path_each_round = os.path.join(generated_dataset_obj.path, f'round{round_num}')
        generated_dataset_obj.make_dir_for_dataset(generated_dataset_path_each_round)
        self.config['generated_dataset'] = self.generated_dataset_config(checkpoint_path_each_round,
                                                                         generated_dataset_path_each_round)

        # load RAFT model: pre-trained model
        model = torch.nn.DataParallel(RAFT(self.config['generated_dataset']))
        model.load_state_dict(torch.load(self.config['generated_dataset'].model))
        model.cuda()
        model.eval()

        # choose your dataset here
        dataset = render_datasets.Structure(original_dataset_path)

        with torch.no_grad():
            render_local(model, dataset, data_type, scene, flag,
                         self.config['generated_dataset'].save_path,
                         float(self.config['generated_dataset'].alpha),
                         str(self.config['generated_dataset'].splatting),
                         iters=int(self.config['generated_dataset'].iter))

        # 计算完之后也尝试释放内存
        torch.cuda.empty_cache()

    @print_function_name_decorator
    def generated_dataset_config(self, checkpoint_path, generated_dataset_path):
        parser = argparse.ArgumentParser()
        # RAFT parameteqqrs
        parser.add_argument('--model', help="restore checkpoint", default=checkpoint_path)
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=True)
        parser.add_argument('--save_location', help="save the results in local or oss", default='local')
        parser.add_argument('--save_path', help=" local path to save the result",
                            default=generated_dataset_path)
        parser.add_argument('--iter', help=" kitti 24, sintel 32", default=32)
        parser.add_argument('--alpha', default=0.75)
        parser.add_argument('--splatting', help=" max or softmax", default='softmax')
        parser.add_argument('--gpus', type=int, nargs='+', default=[0])

        args = parser.parse_args()

        return args

    @print_function_name_decorator
    def finetune_model(self, round_num, flag, model_name, data_type, scene, generated_dataset_path, models_path, train_results_path, validation_results_path):

        # 计算完之前也尝试释放内存
        torch.cuda.empty_cache()

        train_result_path_each_round = os.path.join(train_results_path, f'round{round_num}')
        validation_results_path_each_round = os.path.join(validation_results_path, f'round{round_num}')
        checkpoints_path_each_round = os.path.join(models_path, f'round{round_num}')
        self.make_dir(train_result_path_each_round)
        self.make_dir(validation_results_path_each_round)
        self.make_dir(checkpoints_path_each_round)

        # 第一轮训练的模型是无参数的模型，从头开始训练
        # 后面轮数训练的模型是前一轮训练完成的模型
        if round_num == 1:
            # 创建训练模型的配置文件
            checkpoint_path_each_round = os.path.join(f'../RAFT/models/{model_name}.pth')
        else:
            # 创建训练模型的配置文件
            checkpoint_path_each_round = os.path.join(models_path, f'round{round_num-1}', f'{model_name}_{data_type}_{scene}.pth')

        self.config['finetune'] = self.finetune_config(checkpoint_path_each_round, model_name, data_type, scene)

        # 设置了随机种子，以确保实验的可重复性
        torch.manual_seed(1234)
        np.random.seed(1234)

        # 训练入口
        generated_dataset_path = os.path.join(generated_dataset_path, f'round{round_num}')
        train(self.config['finetune'], generated_dataset_path, checkpoints_path_each_round,
              train_result_path_each_round, validation_results_path_each_round, flag, data_type, scene,
              self.max_flow, self.train_summary_frequency, self.validation_frequency)

        # 计算完之后也尝试释放内存
        torch.cuda.empty_cache()

    @print_function_name_decorator
    def finetune_config(self, checkpoint_path, model_name, data_type, scene):
        '''
            --name: 试验名字
            --stage: 用来训练的数据集
            --small： 是否用小模型，如果设置了该标志，模型将使用较小的参数规模，官方给出的案例没有用这个，那我们也先不用
            --validation： 测试的数据集
            --lr：学习率
            --num_steps：训练步数，暂定30步
            --batch_size： 训练数据批次量，设置为2
            --image_size： 图片尺寸，改成我们自己的，但是这一个参数后面是给了crop_size，用来做数据增强，要保持高和宽都是8的倍数
            --gpus：多少个GPU
            --mixed_precision：是否使用复合精度，提高运算速度
            --iters： 这是光流估计模型的迭代次数，默认为12
            --wdecay： 权重衰减，默认值为0.00005
            --epsilon： Adam优化器的epsilon值，默认为1e-8
            --clip: 梯度裁剪的阈值，默认为1.0
            --dropout: Dropout的概率，默认为0.0
            --gamma: 指数加权的参数，默认为0.8
            --add_noise: 是否向输入图像添加噪声
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', default=f'{model_name}_{data_type}_{scene}', help="name your experiment")
        parser.add_argument('--stage', help="determines which dataset to use for training", default=f'{data_type}')
        parser.add_argument('--validation', type=str, nargs='+', default=f'{data_type}')
        parser.add_argument('--restore_ckpt', help="restore checkpoint", default=checkpoint_path)

        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--lr', type=float, default=0.00002)
        parser.add_argument('--gpus', type=int, nargs='+', default=[0])
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=True)
        parser.add_argument('--iters', type=int, default=12)
        parser.add_argument('--wdecay', type=float, default=.00005)
        parser.add_argument('--epsilon', type=float, default=1e-8)
        parser.add_argument('--clip', type=float, default=1.0)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
        parser.add_argument('--add_noise', action='store_true')

        if data_type == 'steel_ruler':
            parser.add_argument('--image_size', type=int, nargs='+', default=[344, 1272])
            parser.add_argument('--num_steps', type=int, default=self.total_steps['steel_ruler'])
            parser.add_argument('--batch_size', type=int, default=2)
        elif data_type == 'five_floors_frameworks':
            parser.add_argument('--image_size', type=int, nargs='+', default=[1912, 592])
            parser.add_argument('--num_steps', type=int, default=self.total_steps['five_floors_frameworks'])
            parser.add_argument('--batch_size', type=int, default=2)

        args = parser.parse_args()

        return args

    @print_function_name_decorator
    def result_visualization(self):

        # train_result和validate_result可以复用
        splits = ['train', 'validation']
        for split in splits:
            self.load_metrics(split)  # 加载数据
            self.metrics_each_round(split)  # 每一轮分开画
            self.metrics_whole_round(split)  # 所有画在一起

    @print_function_name_decorator
    def load_metrics(self, split):

        results_dict = getattr(self, split + '_results')

        for model_name in self.model_list:

            self.metrics_dict[model_name] = {}

            for data_type, scenes in self.total_scenes.items():

                self.metrics_dict[model_name][data_type] = {}
                for scene in scenes:
                    self.metrics_dict[model_name][data_type][scene] = {}
                    round_paths = os.listdir(results_dict[model_name][data_type][scene])
                    # 对列表进行排序，按照数字的大小进行排序
                    round_paths = sorted(round_paths, key=lambda x: int(x[5:]))

                    for round_path in round_paths:

                        self.metrics_dict[model_name][data_type][scene][round_path] = {}
                        each_round_path = os.path.join(results_dict[model_name][data_type][scene], round_path)
                        json_file_list = os.listdir(each_round_path)
                        # 需要对json文件进行排序再取出
                        json_file_list = sorted(json_file_list, key=lambda x: int(os.path.splitext(x)[0].split('_')[0]))

                        for step, json_file_path in enumerate(json_file_list):
                            each_json_file = os.path.join(each_round_path, json_file_path)

                            # 读取 JSON 文件
                            with open(each_json_file, 'r') as file:
                                data = json.load(file)

                                for metric_type in data.keys():
                                    if step == 0:
                                        self.metrics_dict[model_name][data_type][scene][round_path][metric_type] = []
                                    self.metrics_dict[model_name][data_type][scene][round_path][metric_type].append(
                                        data[metric_type])

    @print_function_name_decorator
    def metrics_each_round(self, split):

        results_dict = getattr(self, split + '_results')
        metrics_image_dict = getattr(self, split + '_metrics_img')

        for model_name in self.model_list:
            for data_type, scenes in self.total_scenes.items():
                for scene in scenes:
                    round_paths = os.listdir(results_dict[model_name][data_type][scene])
                    # 对列表进行排序，按照数字的大小进行排序
                    round_paths = sorted(round_paths, key=lambda x: int(x[5:]))

                    for index, round_path in enumerate(round_paths):
                        for metric_type in self.metrics_dict[model_name][data_type][scene][round_path].keys():
                            # 建个文件夹分开放
                            img_each_metrics_type_path = os.path.join(metrics_image_dict[model_name][data_type][scene], metric_type)
                            self.make_dir(img_each_metrics_type_path)

                            self.plot_line_chart_each_round(data_type, scene, round_path, metric_type, index,
                                                 self.metrics_dict[model_name][data_type][scene][round_path][metric_type],
                                                 img_each_metrics_type_path, split)

    @print_function_name_decorator
    def plot_line_chart_each_round(self, data_type, scene, round_path, metric_type, index, y, img_each_metrics_type_path, split):

        x = None
        if split == 'train':
            x = np.arange(len(y)) + 1  # x 轴数据
        elif split == 'validation':
            if data_type == 'steel_ruler':
                x = np.arange(10, self.total_steps['steel_ruler']+1, self.validation_frequency)  # x 轴数据
            elif data_type == 'five_floors_frameworks':
                x = np.arange(10, self.total_steps['five_floors_frameworks']+1, self.validation_frequency)  # x 轴数据

        # 绘制折线图
        plt.figure(figsize=(6, 6))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.plot(x, y, label=metric_type)  # 绘制第一组数据的折线
        # 添加标题和标签
        plt.title(f'{data_type}_{scene}_{round_path}')  # 添加标题
        plt.xlabel('iteration', fontsize=12, fontweight='bold')
        plt.ylabel(metric_type, fontsize=12, fontweight='bold')
        plt.xticks(range(0, len(y), 5), fontsize=10)
        if metric_type == 'epe':
            if data_type == 'steel_ruler':
                # np.arange生成的数组不包括终止值
                plt.xticks(np.arange(0, self.total_steps['steel_ruler']+1, self.total_steps['steel_ruler']//10), fontsize=10)
                plt.yticks(np.arange(0, 30, 5), fontsize=10)
            elif data_type == 'five_floors_frameworks':
                plt.xticks(np.arange(0, self.total_steps['five_floors_frameworks']+1, self.total_steps['five_floors_frameworks']//10), fontsize=10)
                plt.yticks(np.arange(0, 30, 5), fontsize=10)

        else:
            if data_type == 'steel_ruler':
                plt.xticks(np.arange(0, self.total_steps['steel_ruler']+1, self.total_steps['steel_ruler']//10), fontsize=10)
                plt.yticks(np.linspace(0, 1.2, 7), fontsize=10)
            elif data_type == 'five_floors_frameworks':
                plt.xticks(np.arange(0, self.total_steps['five_floors_frameworks']+1, self.total_steps['five_floors_frameworks']//10), fontsize=10)
                plt.yticks(np.linspace(0, 1.2, 7), fontsize=10)

        plt.savefig(os.path.join(img_each_metrics_type_path, f'{round_path}_{metric_type}.png'))
        plt.show()

    @print_function_name_decorator
    def metrics_whole_round(self, split):
        metric_types = ['epe', '1px', '3px', '5px']
        for model_name in self.model_list:
            for data_type, scenes in self.total_scenes.items():
                for scene in scenes:
                    for metric_type in metric_types:
                        self.plot_line_chart_whole_round(model_name, data_type, scene, metric_type, split)

    @print_function_name_decorator
    def plot_line_chart_whole_round(self, model_name, data_type, scene, metric_type, split):

        metrics_image_dict = getattr(self, split + '_metrics_img')

        plt.figure(figsize=(6, 6))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'  # 设置Times New Roman
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        colors = ['red', 'green', 'blue', 'purple', 'cyan']
        linestyles = ['-', '--', '-.', ':', '-']

        x, y = None, None
        rounds = [f'round{i+1}' for i in range(5)]
        for index, round_number in enumerate(rounds):
            y = self.metrics_dict[model_name][data_type][scene][round_number][metric_type]

            if split == 'train':
                x = np.arange(len(y)) + 1
            elif split == 'validation':
                if data_type == 'steel_ruler':
                    x = np.arange(10, self.total_steps['steel_ruler']+1, self.validation_frequency)
                elif data_type == 'five_floors_frameworks':
                    x = np.arange(10, self.total_steps['five_floors_frameworks']+1, self.validation_frequency)

            plt.plot(x, y, label=round_number, color=colors[(index % len(colors))], linestyle=linestyles[index // (len(rounds) // len(linestyles))], linewidth=1)

        # 添加标题和标签
        # plt.title(f'{model_name}-{data_type}-{scene}-{metric_type}', fontsize=16, fontweight='bold')  # 添加标题
        plt.xlabel('step', fontsize=16, fontweight='bold')
        plt.ylabel(metric_type, fontsize=16, fontweight='bold')
        # plt.legend(loc='upper left', ncol=3, fontsize=12)
        plt.legend(loc='upper left', ncol=2, fontsize=14)
        if metric_type == 'epe':
            if data_type == 'steel_ruler':
                # np.arange生成的数组不包括终止值
                plt.xticks(np.arange(0, self.total_steps['steel_ruler']+1, self.total_steps['steel_ruler']//10), fontsize=14)
                if model_name == 'raft-things' and scene == 'Industrial_camera_0%':
                    plt.yticks(np.linspace(0, 25, 6), fontsize=14)
                else:
                    plt.yticks(np.linspace(0, 5, 6), fontsize=14)

            elif data_type == 'five_floors_frameworks':
                plt.xticks(np.arange(0, self.total_steps['five_floors_frameworks']+1, self.total_steps['five_floors_frameworks']//10), fontsize=14)
                plt.yticks(np.linspace(0, 5, 6), fontsize=14)
        else:
            if data_type == 'steel_ruler':
                plt.xticks(np.arange(0, self.total_steps['steel_ruler']+1, self.total_steps['steel_ruler']//10), fontsize=14)
                plt.yticks(np.linspace(0, 1.2, 7), fontsize=14)
            elif data_type == 'five_floors_frameworks':
                plt.xticks(np.arange(0, self.total_steps['five_floors_frameworks']+1, self.total_steps['five_floors_frameworks']//10), fontsize=14)
                plt.yticks(np.linspace(0, 1.2, 7), fontsize=14)

        # 保存图片
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_image_dict[model_name][data_type][scene], f'{model_name}_{data_type}_{scene}_{metric_type}.png'))
        plt.show()

    @print_function_name_decorator
    def finetune_pre_trained_predict(self, round_num, flag, model_name, data_type, scene):

        # 现在每一轮都在预测，那就不用排序再取出round了
        # round_paths = os.listdir(self.checkpoints[model_name][data_type][scene])
        # # 对列表进行排序，按照数字的大小进行排序
        # round_paths = sorted(round_paths, key=lambda x: int(x[5:]))
        #
        # # checkpoints_path，直接取最后一round最后一个，每一轮都是取最大的那个round
        # checkpoint_path = os.path.join(self.checkpoints[model_name][data_type][scene], f'{round_paths[-1]}', f'{model_name}_{data_type}_{scene}.pth')
        # print(f'checkpoint_path: {checkpoint_path}')

        checkpoint_path = os.path.join(self.checkpoints[model_name][data_type][scene], f'round{round_num}',
                                       f'{model_name}_{data_type}_{scene}.pth')
        print(f'checkpoint_path: {checkpoint_path}')
        pre_trained_save_path = os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}')
        self.make_dir(pre_trained_save_path)

        # 配置预训练模型测试的变量
        self.config['pre_trained_test'] = self.pre_trained_test_config(checkpoint_path, pre_trained_save_path)

        # load RAFT model
        model = torch.nn.DataParallel(RAFT(self.config['pre_trained_test']))
        # 加载模型, strict=False表示可以不按照完全相同的模型架构来加载，一般用在迁移学习，但是我们没有对模型架构进行过调整，所以用哪个都行
        model.load_state_dict(torch.load(self.config['pre_trained_test'].model), strict=False)
        model.cuda()
        model.eval()

        # 源图片
        img_file_path = self.original_dataset_dict[data_type][scene]
        # 存放光流数据的文件
        Data_Box = []

        # 分别创建存放图片和光流数据的文件
        Data_Box_path = os.path.join(pre_trained_save_path, 'Data_Box')
        image_path = os.path.join(pre_trained_save_path, 'image')
        self.make_dir(Data_Box_path)
        self.make_dir(image_path)

        # 使用dataset加载后优化代码
        dataset_structure = render_datasets.Structure(img_file_path)
        with torch.no_grad():
            for val_id in tqdm(range(len(dataset_structure) - 1)):
                img1, img2, _, _ = dataset_structure[val_id]
                img1 = img1[None].cuda()
                img2 = img2[None].cuda()

                padder = InputPadder(img1.shape)
                img1, img2 = padder.pad(img1, img2)

                flow_low, flow_up = model(img1, img2, iters=self.config['pre_trained_test'].iters,
                                          test_mode=True)
                flow_data = flow_up[0].permute(1, 2, 0).cpu().numpy()
                flo = flow_viz.flow_to_image(flow_data)
                cv2.imwrite(os.path.join(pre_trained_save_path, 'image', f'flow_up_{val_id}.png'), flo)

                Data_Box.append(flow_data)
                print(f'Data_box_size: {len(Data_Box)}, type: {type(Data_Box)}')

                if (val_id + 1) % 200 == 0 and val_id > 0:
                    try:
                        npz_path = os.path.join(Data_Box_path, f'flow_up_data_chunk_{val_id // 200 + 1}.npz')
                        np.savez(npz_path, arr=Data_Box)
                        # 每次存完之后就清空数据
                        Data_Box.clear()
                        print("Section saved array using Numpy's savez method successfully.")
                        print("Data_Box cleared successfully.")
                    except Exception as e:
                        print("An error occurred:", e)

                # 释放GPU内存
                torch.cuda.empty_cache()

    @print_function_name_decorator
    def pre_trained_test_config(self, checkpoint_path, save_path):
        parser = argparse.ArgumentParser()
        # RAFT parameters
        parser.add_argument('--model', help="restore checkpoint", default=checkpoint_path)
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=True)
        parser.add_argument('--save_location', help="save the results in local or oss", default='local')
        parser.add_argument('--save_path', help=" local path to save the result", default=save_path)
        parser.add_argument('--iters', help=" kitti 24, sintel 32", default=32)
        parser.add_argument('--gpus', type=int, nargs='+', default=[0])

        args = parser.parse_args()

        return args

    @print_function_name_decorator
    def original_pre_trained_predict(self, flag, model_name, data_type, scene):

        checkpoint_path = f'../RAFT/models/{model_name}.pth'
        print(f'checkpoint_path: {checkpoint_path}')

        # 保存的路径为新建的文件夹，保存图像文件和data_box
        save_path = self.original_pre_trained_test[model_name][data_type][scene]

        # 配置预训练模型测试的变量
        self.config['original_pre_trained_test'] = self.pre_trained_test_config(checkpoint_path, save_path)

        # load RAFT model
        model = torch.nn.DataParallel(RAFT(self.config['original_pre_trained_test']))
        # 加载模型, strict=False表示可以不按照完全相同的模型架构来加载，一般用在迁移学习，但是我们没有对模型架构进行过调整，所以用哪个都行
        model.load_state_dict(torch.load(self.config['original_pre_trained_test'].model), strict=False)
        model.cuda()
        model.eval()

        # 源图片
        img_file_path = self.original_dataset_dict[data_type][scene]
        # 存放光流数据的文件
        Data_Box = []

        # 分别创建存放图片和光流数据的文件
        Data_Box_path = os.path.join(save_path, 'Data_Box')
        image_path = os.path.join(save_path, 'image')
        self.make_dir(Data_Box_path)
        self.make_dir(image_path)

        # 使用dataset加载后优化代码
        dataset_structure = render_datasets.Structure(img_file_path)
        with torch.no_grad():
            for val_id in tqdm(range(len(dataset_structure) - 1)):
                img1, img2, _, _ = dataset_structure[val_id]
                img1 = img1[None].cuda()
                img2 = img2[None].cuda()

                padder = InputPadder(img1.shape)
                img1, img2 = padder.pad(img1, img2)

                flow_low, flow_up = model(img1, img2, iters=self.config['original_pre_trained_test'].iters,
                                          test_mode=True)
                flow_data = flow_up[0].permute(1, 2, 0).cpu().numpy()
                flo = flow_viz.flow_to_image(flow_data)
                cv2.imwrite(os.path.join(save_path, 'image', f'flow_up_{val_id}.png'), flo)
                Data_Box.append(flow_data)
                print(f'Data_box_size: {len(Data_Box)}, type: {type(Data_Box)}')

                if (val_id + 1) % 200 == 0 and val_id > 0:
                    try:
                        # 保存文件名比finetune的多了一个original
                        npz_path = os.path.join(Data_Box_path, f'flow_up_original_data_chunk_{val_id // 200 + 1}.npz')
                        np.savez(npz_path, arr=Data_Box)
                        # 每次存完之后就清空数据
                        Data_Box.clear()
                        print("Section saved array using Numpy's savez method successfully.")
                        print("Data_Box cleared successfully.")
                    except Exception as e:
                        print("An error occurred:", e)

                # 释放GPU内存
                torch.cuda.empty_cache()

    @print_function_name_decorator
    def compare_original_and_finetune(self):
        for flag in self.flag:
            for model_name in self.model_list:
                for data_type, scenes in self.total_scenes.items():
                    # 钢尺的暂时不需要重做
                    if data_type == 'steel_ruler':
                        pass
                    else:
                        for scene in scenes:
                            # 分了5轮训练的结果
                            for round_num in range(self.round_sum):
                                # 选择图片
                                self.select_image(model_name, data_type, scene, round_num+1)
                                # 标点coordinate
                                self.coordinate(model_name, data_type, scene, round_num+1)
                                # 计算缩放因子calc_factor
                                scale_factor = self.calc_factor(model_name, data_type, scene, round_num+1)
                                # 与excel比较compare_excel
                                self.compare_excel(model_name, data_type, scene, scale_factor, round_num+1)

    @print_function_name_decorator
    def select_image(self, model_name, data_type, scene, round_num):
        # 现在每一轮都测试一次
        each_path = os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}')

        items = os.listdir(each_path)

        # 目前默认选择第一张就好，对于两种方法都一样
        if 'select_image' not in items:
            # 后面的方法是：直接用预测图片序列的第一张，来取区域
            select_image_path = os.path.join(each_path, 'select_image')
            self.make_dir(select_image_path)

            # 把预测的图像序列第一张放到select_image文件路径下: original_dataset
            source_path = os.path.join(self.original_dataset_dict[data_type][scene],
                                       os.listdir(self.original_dataset_dict[data_type][scene])[0])  # 源文件路径
            destination_path = os.path.join(select_image_path, 'select_image.png')  # 目标文件路径
            print(f'source_path: {source_path}, destination_path: {destination_path}')
            shutil.copyfile(source_path, destination_path)  # 复制图片
            print("图片已成功复制到目标路径！")

    @print_function_name_decorator
    def coordinate(self, model_name, data_type, scene, round_num):
        # 这次的都选定比较好
        points = {}
        if data_type == 'steel_ruler':
            # 一开始选择的还行的结果
            points = {'segment1': {'point1': [125, 110], 'point2': [167, 160]},
                      'segment2': {'point1': [647, 136], 'point2': [694, 182]},
                      'segment3': {'point1': [1172, 126], 'point2': [1220, 167]}}
            # 比较好的结果
            # points = {'segment1': {'point1': [106, 86], 'point2': [167, 161]},
            #           'segment2': {'point1': [625, 114], 'point2': [693, 182]},
            #           'segment3': {'point1': [1146, 104], 'point2': [1220, 165]}}
        elif data_type == 'five_floors_frameworks':
            if scene == '0-4_0%':
                # 整个标定板
                points = {'segment1': {'point1': [144, 267], 'point2': [357, 477]},
                          'segment2': {'point1': [135, 606], 'point2': [351, 819]},
                          'segment3': {'point1': [150, 945], 'point2': [360, 1164]},
                          'segment4': {'point1': [132, 1308], 'point2': [357, 1506]},
                          'segment5': {'point1': [156, 1653], 'point2': [366, 1869]}}
                # 左边一小块
                # points = {'segment1': {'point1': [123, 378], 'point2': [153, 405]},
                #           'segment2': {'point1': [123, 693], 'point2': [153, 726]},
                #           'segment3': {'point1': [123, 1029], 'point2': [153, 1077]},
                #           'segment4': {'point1': [123, 1389], 'point2': [153, 1425]},
                #           'segment5': {'point1': [123, 1746], 'point2': [153, 1770]}}
                # 右边一小块：比较好的结果
                # points = {'segment1': {'point1': [360, 360], 'point2': [370, 370]},
                #           'segment2': {'point1': [360, 700], 'point2': [370, 710]},
                #           'segment3': {'point1': [365, 1055], 'point2': [375, 1065]},
                #           'segment4': {'point1': [365, 1410], 'point2': [375, 1420]},
                #           'segment5': {'point1': [370, 1750], 'point2': [380, 1760]}}
            elif scene == '01-02_5%':
                points = {'segment1': {'point1': [114, 195], 'point2': [321, 399]},
                          'segment2': {'point1': [105, 537], 'point2': [321, 753]},
                          'segment3': {'point1': [114, 882], 'point2': [324, 1089]},
                          'segment4': {'point1': [102, 1242], 'point2': [324, 1446]},
                          'segment5': {'point1': [93, 1596], 'point2': [309, 1812]}}

        # 画矩形
        img = cv2.imread(
            os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}', 'select_image',
                         'select_image.png'))
        for i in range(0, len(points)):
            pt1 = points[f'segment{i + 1}']['point1']
            pt2 = points[f'segment{i + 1}']['point2']
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), thickness=3)
            # 不需要标出坐标
            # cv2.putText(img, f"{pt1[0]},{pt1[1]}", (pt1[0] - 20, pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6, (0, 0, 0), thickness=2)
            # cv2.putText(img, f"{pt2[0]},{pt2[1]}", (pt1[0] - 20, pt1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6, (0, 0, 0), thickness=2)
        # 保存标了点的图片
        cv2.imwrite(
            os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}', 'select_image',
                         'coordinate.png'), img)
        print(f'保存coordinate.png成功')

        # points.json, 编码格式为 UTF-8
        # ensure_ascii=False 参数确保在保存 JSON 文件时不会将非 ASCII 字符转换为 Unicode 转义序列
        points_path = os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}',
                                   'select_image', 'points.json')
        with open(points_path, 'w', encoding='utf-8') as f:
            json.dump(points, f, ensure_ascii=False)
        # 打印保存路径
        print(f"JSON 文件已保存到：{points_path}")

    @print_function_name_decorator
    def calc_factor(self, model_name, data_type, scene, round_num):
        # 组装点
        physical_distance = 0
        points = {}
        if data_type == 'steel_ruler':
            physical_distance = 45
            points = {'point1': [600, 200], 'point2': [710, 200]}
        elif data_type == 'five_floors_frameworks':
            physical_distance = 140
            if scene == '0-4_0%' or scene == '0-4_0%_all':
                points = {'point1': [140, 477], 'point2': [360, 477]}
            elif scene == '01-02_5%' or scene == '01-02_5%_all':
                points = {'point1': [110, 402], 'point2': [330, 402]}

        # 画线
        img = cv2.imread(
            os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}', 'select_image',
                         'select_image.png'))
        cv2.line(img, points['point1'], points['point2'], (255, 0, 0), 3)
        scale_factor_path = os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}',
                                         'select_image', 'scale_factor.png')
        cv2.imwrite(scale_factor_path, img)
        print(f'scale_factor.png保存成功: {scale_factor_path}')

        # 输入已知的物理距离
        pixel_distance = ((points['point1'][0] - points['point2'][0]) ** 2 + (
                points['point1'][1] - points['point2'][1]) ** 2) ** 0.5
        # 计算比例因子
        scale_factor = physical_distance / pixel_distance
        print(f'data_type: {data_type}, scene: {scene}, 缩放因子: {scale_factor}')

        return scale_factor

    @print_function_name_decorator
    def compare_excel(self, model_name, data_type, scene, scale_factor):

        # 加载光流数据
        flow_data = self.get_flow_data(model_name, data_type, scene, scale_factor)
        # 加载excel数据
        excel, optical_flow_time = self.get_excel_data(model_name, data_type, scene)
        # 画位移图
        self.visualize_displacement(model_name, data_type, scene, flow_data, excel, optical_flow_time)
        # 计算RMSE, NRMSE
        self.calc_error(model_name, data_type, scene, flow_data, excel)
        # 画误差图
        self.visualize_error(model_name, data_type, scene)

    @print_function_name_decorator
    def compare_excel(self, model_name, data_type, scene, scale_factor, round_num):

        # 加载光流数据
        flow_data = self.get_flow_data(model_name, data_type, scene, scale_factor, round_num)
        # 加载excel数据
        excel, optical_flow_time = self.get_excel_data(model_name, data_type, scene)
        # 画位移图
        self.visualize_displacement(model_name, data_type, scene, flow_data, excel, optical_flow_time, round_num)
        # 计算RMSE, NRMSE
        self.calc_error(model_name, data_type, scene, flow_data, excel, round_num)
        # 画误差图
        self.visualize_error(model_name, data_type, scene, round_num)

    @print_function_name_decorator
    def get_flow_data(self, model_name, data_type, scene, scale_factor, round_num):
        # 钢尺试验都是测量dy, 五层框架测量dx
        # 数据结构：flow_data = {'original': {'segment1': {'dx': [], 'dy': []}, 'segment2': {}, 'segment3': {}}, 'finetune': {}}
        flow_data = {}
        # 读取npz文件
        data_box_dict = {'original': [], 'finetune': []}

        # 读取points.json
        with open(os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}',
                               'select_image', 'points.json'), 'r', encoding='utf-8') as f:
            points_dict = json.load(f)
        print(f'读取points_dict成功: {points_dict}')

        x_coordinate = {}
        y_coordinate = {}

        for position, points in points_dict.items():
            # 组装每个position的x, y坐标
            x_coordinate[position] = []
            y_coordinate[position] = []
            for point, coordinate_list in points.items():
                x_coordinate[position].append(coordinate_list[0])
                y_coordinate[position].append(coordinate_list[1])

        # 读取npz文件，区分开是否经过RealFlow优化的模型得出的结果
        npz_items_pre_trained = os.listdir(
            os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}', 'Data_Box'))

        # 分开读取original和pre_trained_test的npz的文件夹
        # pre_trained_test的npz的文件夹
        for npz_item in npz_items_pre_trained:
            npz_file_path = os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}',
                                         'Data_Box', npz_item)
            data_box_dict['finetune'].append(npz_file_path)

        # 读取original文件夹下的npz
        npz_items_original = os.listdir(
            os.path.join(self.original_pre_trained_test[model_name][data_type][scene], 'Data_Box'))
        for npz_item in npz_items_original:
            npz_file_path = os.path.join(self.original_pre_trained_test[model_name][data_type][scene], 'Data_Box',
                                         npz_item)
            data_box_dict['original'].append(npz_file_path)

        # 排序
        data_box_dict['original'] = sorted(data_box_dict['original'],
                                           key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        data_box_dict['finetune'] = sorted(data_box_dict['finetune'],
                                           key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

        # 依次遍历点的坐标
        for stage, data_box_list in data_box_dict.items():

            flow_data[stage] = {}

            for position in points_dict.keys():
                # flow_arr(200, 600, 1920, 2)
                # flow_data数据结构: {'segment1': {'dx': [], 'dy': []}}
                flow_data[stage][position] = {}

                concatenated_arrays_dx = []
                concatenated_arrays_dy = []

                for data_box_item in data_box_list:
                    npz_data = np.load(data_box_item)
                    flow_arr = npz_data['arr']

                    # (201, 1920, 600, 2) 先取y, 再取x
                    dx_three_dimension = flow_arr[:, min(y_coordinate[position]):max(y_coordinate[position]),
                                         min(x_coordinate[position]):max(x_coordinate[position]), 0]
                    dy_three_dimension = flow_arr[:, min(y_coordinate[position]):max(y_coordinate[position]),
                                         min(x_coordinate[position]):max(x_coordinate[position]), 1]
                    dx_mean_array = np.mean(dx_three_dimension, axis=(1, 2))
                    dy_mean_array = np.mean(dy_three_dimension, axis=(1, 2))

                    print(f'position: {position}, shape: {dx_mean_array.shape}')
                    print(f'position: {position}, shape: {dy_mean_array.shape}')

                    # 组装数组
                    concatenated_arrays_dx.append(dx_mean_array)
                    concatenated_arrays_dy.append(dy_mean_array)

                # Industrial_camera_5%_all特殊处理, excel的点数不够，只能取1978个点
                if scene == 'Industrial_camera_5%_all':
                    flow_data[stage][position]['dx'] = np.concatenate(concatenated_arrays_dx, axis=0)
                    flow_data[stage][position]['dy'] = np.concatenate(concatenated_arrays_dy, axis=0)
                    flow_data[stage][position]['dx'] = flow_data[stage][position]['dx'][:1978]
                    flow_data[stage][position]['dy'] = flow_data[stage][position]['dy'][:1978]
                else:
                    flow_data[stage][position]['dx'] = np.concatenate(concatenated_arrays_dx, axis=0)
                    flow_data[stage][position]['dy'] = np.concatenate(concatenated_arrays_dy, axis=0)
                # 乘以缩放系数
                flow_data[stage][position]['dx'] = scale_factor * flow_data[stage][position]['dx']
                flow_data[stage][position]['dy'] = scale_factor * flow_data[stage][position]['dy']

        return flow_data

    @print_function_name_decorator
    def get_excel_data(self, model_name, data_type, scene):
        excel_folder_path = os.path.join(model_name, 'excel_data')
        excel_path = None
        optical_flow_time = None
        excel = {}
        if data_type == 'steel_ruler':
            if scene == 'Industrial_camera_0%':
                excel_path = os.path.join(excel_folder_path, '2023-12-22(Industrial_camera_0%).xlsx')
            elif scene == 'Industrial_camera_5%':
                excel_path = os.path.join(excel_folder_path, '2023-12-22(Industrial_camera_5%).xlsx')
        elif data_type == 'five_floors_frameworks':
            if scene == '0-4_0%':
                excel_path = os.path.join(excel_folder_path, 'five_floors_frameworks_0-4_0%.xlsx')
            elif scene == '01-02_5%':
                excel_path = os.path.join(excel_folder_path, 'five_floors_frameworks_01-02-1_5%.xlsx')

        df = pd.read_excel(excel_path)

        if data_type == 'steel_ruler':
            if scene == 'Industrial_camera_0%':
                # 0.3s-4.3s => 0-200张
                # (0.3+60/100)s=0.9s <= 60张
                # 工业相机100fps, excel200fps
                excel['time'] = df.iloc[185:585, 0]
                optical_flow_time = np.linspace(0.88, 2.88, num=200, endpoint=True)
                excel['segment1'] = df.iloc[185:585, 1]
                excel['segment2'] = df.iloc[185:585, 2]
                excel['segment3'] = df.iloc[185:585, 3]
                excel['excitation'] = df.iloc[185:585, 4]
            elif scene == 'Industrial_camera_5%':
                # 1.05s-5.05s => 0-200张
                # (1.05+278/100)s=3.83s <= 278张
                # 工业相机100fps, excel200fps
                excel['time'] = df.iloc[788:1188, 0]
                optical_flow_time = np.linspace(3.83, 5.83, num=200, endpoint=True)
                excel['segment1'] = df.iloc[788:1188, 1]
                excel['segment2'] = df.iloc[788:1188, 2]
                excel['segment3'] = df.iloc[788:1188, 3]
                excel['excitation'] = df.iloc[788:1188, 4]
        elif data_type == 'five_floors_frameworks':
            if scene == '0-4_0%':
                excel['time'] = df.iloc[17:602, 0]
                optical_flow_time = np.linspace(0, 6, num=600, endpoint=True)
                excel['segment1'] = df.iloc[2:587, 11]
                excel['segment2'] = df.iloc[2:587, 7]
                excel['segment3'] = df.iloc[2:587, 8]
                excel['segment4'] = df.iloc[2:587, 9]
                excel['segment5'] = df.iloc[2:587, 10]
            elif scene == '01-02_5%':
                excel['time'] = df.iloc[89:689, 0]
                optical_flow_time = np.linspace(0.87, 6.87, num=600, endpoint=True)
                excel['segment1'] = df.iloc[89:689, 11]
                excel['segment2'] = df.iloc[89:689, 7]
                excel['segment3'] = df.iloc[89:689, 8]
                excel['segment4'] = df.iloc[89:689, 9]
                excel['segment5'] = df.iloc[89:689, 10]

        return excel, optical_flow_time

    @print_function_name_decorator
    def visualize_displacement(self, model_name, data_type, scene, flow_data, excel, optical_flow_time, round_num):
        if data_type == 'steel_ruler':
            plt.figure(figsize=(len(flow_data['original'])*6, 6))
        elif data_type == 'five_floors_frameworks':
            plt.figure(figsize=(10, 25))

        plt.rcParams['font.sans-serif'] = 'Times New Roman'  # 设置Times New Roman
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

        colors = {'excel': 'black', 'optical_flow': {'original': 'red', 'finetune': 'green'}}
        linestyles = {'excel': '-', 'optical_flow': {'original': '-.', 'finetune': '-.'}}
        linewidths = {'excel': 2, 'optical_flow': {'original': 2, 'finetune': 2}}
        segments = [f'segment{i+1}' for i in range(len(flow_data['original']))]
        sub_titles = {'steel_ruler': ['Left', 'Middle', 'Right'], 'five_floors_frameworks': [f'story{i}' for i in range(len(segments), 0, -1)]}
        stages = ['original', 'finetune']
        direction = {'steel_ruler': 'dy', 'five_floors_frameworks': 'dx'}

        for i, position in enumerate(segments):
            if data_type == 'steel_ruler':
                plt.subplot(1, len(segments), i+1)
            elif data_type == 'five_floors_frameworks':
                plt.subplot(len(segments), 1, i+1)

            plt.plot(excel['time'], excel[position], label='LVDT', linewidth=linewidths['excel'], linestyle='-', color=colors['excel'])

            for j, stage in enumerate(stages):
                plt.plot(optical_flow_time, flow_data[stage][position][direction[data_type]] * (-1),
                         label=stage, linewidth=linewidths['optical_flow'][stage],
                         linestyle=linestyles['optical_flow'][stage], color=colors['optical_flow'][stage])

            plt.ylabel('Disp[mm]', fontsize=16, fontweight='bold')
            plt.xlabel('times[s]', fontsize=16, fontweight='bold')
            # 设置坐标刻度范围
            if data_type == 'steel_ruler':
                plt.yticks(np.linspace(-30, 30, num=7), fontsize=14)
                plt.legend(loc='upper right', fontsize=14)
                if scene == 'Industrial_camera_0%':
                    plt.xticks(np.linspace(0.5, 3.0, num=6), fontsize=14)
                elif scene == 'Industrial_camera_5%':
                    plt.xticks(np.linspace(3.5, 6.0, num=6), fontsize=14)
            elif data_type == 'five_floors_frameworks':
                plt.yticks(np.linspace(-50, 50, num=11), fontsize=14)
                plt.legend(loc='upper left', fontsize=14)
                if scene == '0-4_0%':
                    plt.xticks(np.linspace(0, 6, num=7), fontsize=14)
                elif scene == '01-02_5%':
                    plt.xticks(np.linspace(0, 7, num=8), fontsize=14)

            # 设置子图的标题
            plt.title(sub_titles[data_type][i], loc='center', fontsize=18, fontweight='bold')

        # 设置整体标题
        if data_type == 'steel_ruler':
            plt.suptitle(f'Displacement_{model_name}: LVDT vs {stages[0]} vs {stages[1]}', fontsize=18)
        elif data_type == 'five_floors_frameworks':
            pass

        # 调整子图之间的距离
        plt.tight_layout()
        plt.savefig(os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}', 'displacement.png'))
        plt.show()

    @print_function_name_decorator
    def calc_error(self, model_name, data_type, scene, flow_data, excel, round_num):
        '''
            计算excel的每几位的平均值作为一个值
            Industrial_camera: 100fps, 0.01s/frame
            phone: 100fps, 0.01s/frame
            SLR_camera: 1600 => 400
            LVDT: 200fps, 0.005s/frame
        '''
        # 数据采样处理：把LVDT的数据和光流数据的个数变成一样
        excel_processed = {}
        RMSE = {}
        NRMSE = {}
        segments = [f'segment{i + 1}' for i in range(len(flow_data['original']))]
        stages = ['original', 'finetune']
        if data_type == 'steel_ruler':
            if scene == 'Industrial_camera_0%':
                for position in segments:
                    # 使用scipy.signal.resample函数进行重新采样
                    Fs = 400  # 原始采样率
                    Fs_new = 200  # 目标采样率
                    excel_processed[position] = resample(excel[position], int(len(excel[position]) * Fs_new / Fs))
            elif scene == 'Industrial_camera_5%':
                for position in segments:
                    Fs = 400  # 原始采样率
                    Fs_new = 200  # 目标采样率
                    excel_processed[position] = resample(excel[position], int(len(excel[position]) * Fs_new / Fs))

        elif data_type == 'five_floors_frameworks':
            if scene == '0-4_0%':
                for position in segments:
                    Fs = 585  # 原始采样率
                    Fs_new = 600  # 目标采样率
                    excel_processed[position] = resample(excel[position], int(len(excel[position]) * Fs_new / Fs))
            elif scene == '01-02_5%':
                for position in segments:
                    Fs = 600  # 原始采样率
                    Fs_new = 600  # 目标采样率
                    excel_processed[position] = resample(excel[position], int(len(excel[position]) * Fs_new / Fs))

        # 计算RMSE和NRMSE
        if data_type == 'steel_ruler':
            for stage in stages:
                RMSE[stage] = {}
                NRMSE[stage] = {}
                for position in segments:
                    RMSE[stage][position] = np.sqrt(np.mean(((excel_processed[position] - flow_data[stage][position]['dy'] * (-1)) ** 2)))
                    NRMSE[stage][position] = RMSE[stage][position] / (np.max(excel_processed[position]) - np.min(excel_processed[position]))

                print(f'model: {model_name}, data_type:{data_type}, scene: {scene}, stage: {stage}, RMSE: {RMSE[stage]}, NRMSE: {NRMSE[stage]}')

        elif data_type == 'five_floors_frameworks':
            for stage in stages:
                RMSE[stage] = {}
                NRMSE[stage] = {}
                for position in segments:
                    RMSE[stage][position] = np.sqrt(np.mean(((excel_processed[position] - flow_data[stage][position]['dx'] * (-1)) ** 2)))
                    NRMSE[stage][position] = RMSE[stage][position] / (np.max(excel_processed[position]) - np.min(excel_processed[position]))

                print(f'model: {model_name}, data_type:{data_type}, scene: {scene}, stage: {stage}, RMSE: {RMSE[stage]}, NRMSE: {NRMSE[stage]}')

        # 保存json文件
        json_RMSE_path = os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}', 'RMSE.json')
        json_NRMSE_path = os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}', 'NRMSE.json')
        with open(json_RMSE_path, "w") as json_file:
            json.dump(RMSE, json_file)
        with open(json_NRMSE_path, "w") as json_file:
            json.dump(NRMSE, json_file)

    @print_function_name_decorator
    def visualize_error(self, model_name, data_type, scene, round_num):

        # 每个模型的的original, finetune比较
        self.sub_model_error(model_name, data_type, scene, round_num)

    @print_function_name_decorator
    def sub_model_error(self, model_name, data_type, scene, round_num):
        # 读取json文件
        json_RMSE_path = os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}',
                                      'RMSE.json')
        json_NRMSE_path = os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}',
                                       'NRMSE.json')
        with open(json_RMSE_path, "r") as json_file:
            RMSE = json.load(json_file)
        with open(json_NRMSE_path, "r") as json_file:
            NRMSE = json.load(json_file)

        error = {'NRMSE': NRMSE, 'RMSE': RMSE}

        # 创建子图
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'  # 设置英文Times New Roman
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        segments = [f'segment{i + 1}' for i in range(len(RMSE['original']))]
        stages = ['original', 'finetune']
        colors = ['red', 'green', 'blue']  # 柱子颜色
        x_ticks = {'steel_ruler': ['Left', 'Middle', 'Right'], 'five_floors_frameworks': [f'story{i}' for i in range(len(RMSE['original']), 0, -1)]}

        # 计算柱状图的宽度
        bar_width = 0.25
        x = 0

        for i, evaluation in enumerate(error):

            for j, stage in enumerate(stages):

                # 计算每个柱状图的位置
                x = np.arange(len(segments))
                # 绘制柱状图
                axs[i].bar(x + bar_width * j, list(error[evaluation][stage].values()), width=bar_width, label=stage, color=colors[j])
                if len(stages) == 3:
                    axs[i].set_xticks(x + bar_width)  # 将刻度设置在柱状图的中心
                elif len(stages) == 2:
                    axs[i].set_xticks(x + bar_width / 2)  # 将刻度设置在柱状图的中心
                axs[i].set_xticklabels(x_ticks[data_type], fontsize=14)  # 设置刻度标签
                axs[i].tick_params(axis='both', which='major', labelsize=14)

                # 在每个柱子上方添加数值标签
                for k, position in enumerate(error[evaluation][stage]):
                    axs[i].text(x[k] + bar_width * j, error[evaluation][stage][position], f'{error[evaluation][stage][position]:.2f}', ha='center', va='bottom',
                                fontsize=10, fontweight='bold')

            axs[i].set_xlabel('position', fontsize=16, fontweight='bold')
            axs[i].set_ylabel(f'{evaluation}', fontsize=16, fontweight='bold')
            axs[i].legend(loc='upper left', fontsize=14)
            if evaluation == 'RMSE':
                axs[i].set_yticks(np.linspace(0, 16, num=5))
            elif evaluation == 'NRMSE':
                axs[i].set_yticks(np.linspace(0, 0.5, num=11))

        plt.suptitle(f'error_{model_name}: {stages[0]} vs {stages[1]}', fontsize=18)  # 设置整体标题
        plt.tight_layout()
        plt.savefig(os.path.join(self.pre_trained_test[model_name][data_type][scene], f'round{round_num}', 'error.png'))
        plt.show()


class GeneratedDataset:

    @print_function_name_decorator
    def __init__(self, path):
        self.path = path
        self.splits = ['training', 'test']
        self.file_types = ['image1', 'new_image2', 'flow', 'flow_rgb']

    @print_function_name_decorator
    def make_dir_for_dataset(self, save_path):

        for split in self.splits:

            split_path = os.path.join(save_path, split)
            self.make_dir(split_path)

            for file_type in self.file_types:
                file_type_path = os.path.join(split_path, file_type)
                self.make_dir(file_type_path)

    @print_function_name_decorator
    def make_dir(self, path):
        if not os.path.exists(path):
            # 如果文件路径不存在，则创建它
            os.makedirs(path)
            print(f"文件路径 {path} 不存在，已成功创建")
        else:
            print(f"文件路径 {path} 已存在")


def make_dir(path):
    if not os.path.exists(path):
        # 如果文件路径不存在，则创建它
        os.makedirs(path)
        print(f"文件路径 {path} 不存在，已成功创建")
    else:
        print(f"文件路径 {path} 已存在")


if __name__ == '__main__':

    # 原来直接用预训练模型来渲染和训练，出来的效果不好
    # 四个数据，两个模型，一个帧不更新
    total_scenes = {'steel_ruler': ['Industrial_camera_0%'], 'five_floors_frameworks': ['0-4_0%']}
    model_list = ['raft-chairs', 'raft-things']
    flag = ['notUpdated']
    total_steps = {'steel_ruler': 400, 'five_floors_frameworks': 1000}
    round_num = 5

    max_flow = 400  # 最大光流大小
    train_summary_frequency = 10  # 训练状态打印的频率
    validation_frequency = 100  # 验证频率 & 保存checkpoints的频率

    experiment = Experiment(total_scenes, model_list, flag, round_num, total_steps,
                            max_flow, train_summary_frequency, validation_frequency)
    experiment.control()
