import argparse

import torch
import onnx
import onnxruntime as ort
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from RAFT.core.utils import frame_utils
from RAFT.core.utils.utils import InputPadder
from RAFT.core.raft import RAFT


# 输入的处理
def preprocess(batch_size, height, width):
    img1 = torch.randn(batch_size, 3, height, width).cuda()
    img2 = torch.randn(batch_size, 3, height, width).cuda()
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    return img1, img2


def config(checkpoint_path):
    parser = argparse.ArgumentParser()
    # RAFT parameters
    parser.add_argument('--model', help="restore checkpoint", default=checkpoint_path)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=True)
    parser.add_argument('--save_location', help="save the results in local or oss", default='local')
    parser.add_argument('--save_path', help=" local path to save the result")
    parser.add_argument('--iters', help=" kitti 24, sintel 32", default=12)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])

    args = parser.parse_args()

    return args


# 加载输入
# 按照工业相机的分辨率
img1, img2 = preprocess(1, 1280, 600)

# 导入模型
args = config(r'RAFT\models\raft-things.pth')
model = RAFT(args)
model.load_state_dict(torch.load(args.model), strict=False)
model.cuda()
model.eval()

print(f'model: {model}')

# 模型转换为ONNX格式并保存
with torch.no_grad():
    torch.onnx.export(
        model,                                   # 要转换的模型
        (img1, img2),                            # 模型的任意一组输入，对于光流来说一般是两张图片
        'raft-things.onnx',       # 导出的 ONNX 文件名
        opset_version=16,                        # ONNX 算子集版本，从16开始支持
        input_names=['input'],                   # 输入 Tensor 的名称（自己起名字）
        output_names=['output']                  # 输出 Tensor 的名称（自己起名字）
    )


