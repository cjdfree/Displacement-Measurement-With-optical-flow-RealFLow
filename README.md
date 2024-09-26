<<<<<<< HEAD
# README

## Introduction
- 基于**[RealFlow](https://github.com/megvii-research/RealFlow)**生成光流数据集，实现对结构位移测量数据集的制作，并微调训练结果。
- 原光流基准模型基于**[RAFT](https://github.com/princeton-vl/RAFT)**。

## Requirements
- torch>=1.8.1
- torchvision>=0.9.1
- opencv-python>=4.5.2
- timm>=0.4.5
- cupy>=5.0.0
- numpy>=1.15.0

## Rendered Datasets
![results](https://user-images.githubusercontent.com/1344482/180913871-cbbce758-8b03-46b5-b3a4-b07f0b229f82.JPG)

## 文件目录说明

- `experiment_realflow`文件夹存放项目编写代码，其他代码来自项目**[RealFlow](https://github.com/megvii-research/RealFlow)**。

- `experiment_realflow/experiment.py`运行，自动化，批量生成数据集与对光流模型进行模型微调，训练选择的数据集场景，使用的光流模型以及参数如下，可以根据自己的修改：

  ```python
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
  ```

基于RealFlow生成数据集的，再进行光流模型微调的方法，在结构位移测量的场景使用效果并不理想，生成数据集后训练的结果并不理想。欢迎社区伙伴探讨和发现问题。

## 数据集资源

已试验的数据集和试验结果可以访问链接，永久有效。

通过网盘分享的文件：RealFlow
链接: https://pan.baidu.com/s/1gStBbgmB1SFgYoj44d5rgg?pwd=y3tm 提取码: y3tm 
--来自百度网盘超级会员v5的分享

试验数据位于目录，每一个文件夹代表一次试验，试验数据独立。

- `6.13_experiment`
- `5.30_experiment`
- `5.29_experiment`
- `5.20_experiment`
=======
# Displacement-Measurement-With-optical-flow-RealFLow
>>>>>>> 7636da3a0fb4ada6190fe2b503c76e42bc5530a1
