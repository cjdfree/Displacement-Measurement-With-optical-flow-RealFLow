import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

import RAFT.core.datasets as datasets
from RAFT.core.utils.frame_utils import writeFlow
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
from RAFT.core.utils.flow_viz import flow_to_image

from torchvision.utils import save_image
from utils.tools import FlowReversal
from softmax_splatting import softsplat
from DPT.dpt.models import DPTDepthModel
import cv2


@torch.no_grad()
def render_local(flow_net, dataset, data_type, scene, flag, save_path, alpha, splatting, iters):

    # load DPT depth model, using pretrain DPT model
    depth_model_path = "../DPT/model/dpt_large-midas-2f21e586.pt"
    DPT = DPTDepthModel(
        path=depth_model_path,
        backbone="vitl16_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    DPT.cuda()
    DPT.eval()

    for val_id in tqdm(range(len(dataset)-1)):
        image1, image2, _, _ = dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, 8)
        image1, image2 = padder.pad(image1, image2)

        # estimate bi-directional flow
        with torch.no_grad():
            _, flow_forward = flow_net(image1, image2, iters=iters, test_mode=True)
            _, flow_back = flow_net(image2, image1, iters=iters, test_mode=True)

        flow_fw = padder.unpad(flow_forward)
        image1 = padder.unpad(image1).contiguous()
        image2 = padder.unpad(image2)
        flow_bw = padder.unpad(flow_back)

        # setting alpha
        linspace = alpha
        flow_fw = flow_fw * linspace
        flow_bw = flow_bw * (1 - linspace)

        # occ check
        with torch.no_grad():
            fw = FlowReversal()
            _, occ = fw.forward(image1, flow_fw)
            occ = torch.clamp(occ, 0, 1)

        # dilated occ mask
        occ = occ.squeeze(0).permute(1, 2, 0).cpu().numpy()
        occ = (1-occ)*255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(occ, kernel)/255
        occ = 1-torch.from_numpy(dilated).permute(2, 0, 1).unsqueeze(0).cuda()

        padder = InputPadder(image1.shape, mode='sintel', divisible=32)
        input, input2, flow_fw, flow_bw = padder.pad(image1 / 255, image2 / 255, flow_fw, flow_bw)

        # estimate depth and splatting
        with torch.no_grad():

            # estimate depth and normalize
            tenMetric = DPT(input.cuda())
            tenMetric = (tenMetric - tenMetric.min()) / (tenMetric.max() - tenMetric.min())

            # splatting can choose: softmax, max, summation
            output1 = softsplat.FunctionSoftsplat(tenInput=input, tenFlow=flow_fw,
                                                 tenMetric=tenMetric.unsqueeze(0),
                                                 strType=splatting)

            tenMetric2 = DPT(input2.cuda())
            tenMetric2 = (tenMetric2 - tenMetric2.min()) / (tenMetric2.max() - tenMetric2.min())
            output2 = softsplat.FunctionSoftsplat(tenInput=input2, tenFlow=flow_bw,
                                                  tenMetric=tenMetric2.unsqueeze(0),
                                                  strType=splatting)
        # fuse the result
        output = padder.unpad(output1) * occ + (1 - occ) * padder.unpad(output2)
        input = padder.unpad(input)
        flow = padder.unpad(flow_fw).squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 保存结果
        if flag == 'Updated':
            if data_type == 'steel_ruler':
                if scene == 'Industrial_camera_0%' or scene == 'Industrial_camera_5%':
                    # 保存文件，前150对的数据，保存到training，后面50对的数据保存到test
                    if val_id < 150:
                        save_image(input, os.path.join(save_path, 'training', 'image1', f'{val_id}.png'))
                        save_image(output, os.path.join(save_path, 'training', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'training', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'training', 'flow_rgb', f'{val_id}.png'), flow_image)
                    elif val_id >= 150:
                        save_image(input, os.path.join(save_path, 'test', 'image1', f'{val_id}.png'))
                        save_image(output, os.path.join(save_path, 'test', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'test', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'test', 'flow_rgb', f'{val_id}.png'), flow_image)

                elif scene == 'Industrial_camera_0%_0-400' or scene == 'Industrial_camera_5%_0-400':
                    # 保存文件，前300对的数据，保存到training，后面100对的数据保存到test
                    if val_id < 300:
                        save_image(input, os.path.join(save_path, 'training', 'image1', f'{val_id}.png'))
                        save_image(output, os.path.join(save_path, 'training', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'training', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'training', 'flow_rgb', f'{val_id}.png'), flow_image)
                    elif val_id >= 300:
                        save_image(input, os.path.join(save_path, 'test', 'image1', f'{val_id}.png'))
                        save_image(output, os.path.join(save_path, 'test', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'test', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'test', 'flow_rgb', f'{val_id}.png'), flow_image)

            elif data_type == 'five_floors_frameworks':
                # 保存文件，前150对的数据，保存到training，后面50对的数据保存到test
                if val_id < 450:
                    save_image(input, os.path.join(save_path, 'training', 'image1', f'{val_id}.png'))
                    save_image(output, os.path.join(save_path, 'training', 'new_image2', f'{val_id}.png'))
                    writeFlow(os.path.join(save_path, 'training', 'flow', f'{val_id}.flo'), flow)
                    flow_image = flow_to_image(flow)
                    cv2.imwrite(os.path.join(save_path, 'training', 'flow_rgb', f'{val_id}.png'), flow_image)
                elif val_id >= 450:
                    save_image(input, os.path.join(save_path, 'test', 'image1', f'{val_id}.png'))
                    save_image(output, os.path.join(save_path, 'test', 'new_image2', f'{val_id}.png'))
                    writeFlow(os.path.join(save_path, 'test', 'flow', f'{val_id}.flo'), flow)
                    flow_image = flow_to_image(flow)
                    cv2.imwrite(os.path.join(save_path, 'test', 'flow_rgb', f'{val_id}.png'), flow_image)

        elif flag == 'notUpdated':

            save_image(input, os.path.join(save_path, 'training', 'image1', f'img1.png'))

            if data_type == 'steel_ruler':
                if scene == 'Industrial_camera_0%' or scene == 'Industrial_camera_5%':
                    # 保存文件，前150对的数据，保存到training，后面50对的数据保存到test
                    if val_id < 150:
                        save_image(output, os.path.join(save_path, 'training', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'training', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'training', 'flow_rgb', f'{val_id}.png'), flow_image)
                    elif val_id >= 150:
                        save_image(output, os.path.join(save_path, 'test', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'test', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'test', 'flow_rgb', f'{val_id}.png'), flow_image)

                elif scene == 'Industrial_camera_0%_all' or scene == 'Industrial_camera_5%_all':
                    # 按9:1划分训练集和测试集
                    if val_id < 1800:
                        save_image(output, os.path.join(save_path, 'training', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'training', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'training', 'flow_rgb', f'{val_id}.png'), flow_image)
                    elif val_id >= 1800:
                        save_image(output, os.path.join(save_path, 'test', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'test', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'test', 'flow_rgb', f'{val_id}.png'), flow_image)

                elif scene == 'Industrial_camera_0%_0-400' or scene == 'Industrial_camera_5%_0-400':
                    # 保存文件，前300对的数据，保存到training，后面100对的数据保存到test
                    if val_id < 300:
                        save_image(output, os.path.join(save_path, 'training', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'training', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'training', 'flow_rgb', f'{val_id}.png'), flow_image)
                    elif val_id >= 300:
                        save_image(output, os.path.join(save_path, 'test', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'test', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'test', 'flow_rgb', f'{val_id}.png'), flow_image)

            elif data_type == 'five_floors_frameworks':
                if scene == '0-4_0%' or scene == '01-02_5%':
                    # 保存文件，前450对的数据，保存到training，后面150对的数据保存到test
                    if val_id < 450:
                        save_image(output, os.path.join(save_path, 'training', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'training', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'training', 'flow_rgb', f'{val_id}.png'), flow_image)
                    elif val_id >= 450:
                        save_image(output, os.path.join(save_path, 'test', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'test', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'test', 'flow_rgb', f'{val_id}.png'), flow_image)
                elif scene == '0-4_0%_all' or scene == '01-02_5%_all':
                    # 按9:1划分训练集和测试集
                    if val_id < 1800:
                        save_image(output, os.path.join(save_path, 'training', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'training', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'training', 'flow_rgb', f'{val_id}.png'), flow_image)
                    elif val_id >= 1800:
                        save_image(output, os.path.join(save_path, 'test', 'new_image2', f'{val_id}.png'))
                        writeFlow(os.path.join(save_path, 'test', 'flow', f'{val_id}.flo'), flow)
                        flow_image = flow_to_image(flow)
                        cv2.imwrite(os.path.join(save_path, 'test', 'flow_rgb', f'{val_id}.png'), flow_image)

        # 尝试释放内存
        torch.cuda.empty_cache()

