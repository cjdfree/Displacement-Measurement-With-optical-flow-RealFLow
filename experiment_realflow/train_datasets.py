# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import os
import random
from glob import glob
import os.path as osp
import cv2
from RAFT.core.utils import frame_utils
from RAFT.core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
import math


# 基类：根据Pytorch的dataset类来创建的一个自己的数据集
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        # 基类默认不增强数据集，计算稠密光流，后面每个数据集都有自己的增强操作
        self.augmentor = None
        self.sparse = sparse
        # 如果传进来的数据增强参数不是None，那就去数据增强
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        # 默认是：不是测试集
        self.is_test = False
        self.init_seed = False
        '''
            self.flow_list: ['01.flo', '02.flo', '03.flo', ...],  储存光流ground truth的，.flo文件路径
            self.image_list: [[img1, img2], [img2, img3], [img3, img4], [], ..., []], 里面是图像的路径
            self.extra_info: [['alley', 0], ['alley', 1], ..., [scene, index], ...]
        '''
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

        self.dataclean = None
        self.type = 'chairs'
        self.depth = False

    # 这个方法在进去训练的时候，循环再调用
    # for i_batch, data_blob in enumerate(train_loader):
    # index是外面循环取数据的下标，从0, 1, 2, ...
    def __getitem__(self, index):

        # 测试集返回的取数据方式: img1, img2, self.extra_info[index]
        # 每次拿到两张图片img1, img2，并且拿到extra_info，把他们读取出来后，全部转换成tensor的形式返回
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        # 下面就是训练集的返回数据方式
        # 设置随机种子，以确保在多线程数据加载时每个线程使用不同的随机种子，避免在多线程下数据重复
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        # 这个部分作者RealFlow作者在RAFT的源码上面做了一定的修改
        # things, 0706
        # 判断当前数据是否是dict类型，如果是dict类型，说明数据来源于self.dataclean对象，通过getiterm的方法来获取样本数据
        # RAFT源码都是以list格式的格式存储，这里是用dict存储的时候读取的方法
        if isinstance(self.image_list[index], dict):
            sample = self.dataclean.getiterm(self.image_list[index])
            img1, img2, flow, valid = sample['im1'], sample['im2'], sample['flow'], sample['valid']
            # img1, img2 = sample['im1'], sample['im2']
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            valid = torch.from_numpy(valid).permute(2, 0, 1).float()
            return img1, img2, flow, valid

        # 这部分是RAFT的源代码
        else:
            index = index % len(self.image_list)
            valid = None
            if self.sparse:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            else:
                flow = frame_utils.read_gen(self.flow_list[index])

            # RAFT源码不涉及depth的计算
            if self.depth == True:
                depth = frame_utils.depth_read(self.depth_list[index])
                depth = np.array(depth).astype(np.float32)
                depth = torch.from_numpy(depth).unsqueeze(0).float()

            # 读取图片和光流
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            # grayscale images
            if len(img1.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
                img2 = np.tile(img2[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]

        # 数据增强操作
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    # 重载了乘法运算符 *，用于批量操作。将光流列表和图像列表按照给定的倍数扩展。
    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    # 返回数据集的长度
    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/data/Sintel', dstype='clean'):

        # 一般都是在拿训练集
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        # 里面分场景
        for scene in sorted(os.listdir(image_root)):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                # 图片两两拿出来
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))
        depth_root = osp.join('/data/Sintel_depth', split, 'depth')
        self.depth_list = []
        for scene in os.listdir(image_root):
            self.depth_list += sorted(glob(osp.join(depth_root, scene, '*.dpt')))[:-1]
        self.depth = True


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/data/Optical_Flow_all/datasets/KITTI_data/data_scene_flow/'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1


# 我们自己的数据集，需要继承FlowDataset
class SteelRuler(FlowDataset):
    def __init__(self, flag, data_type, scene, root, aug_params=None, split='training'):

        if flag == 'Updated':
            # 一般都是在拿训练集
            super(SteelRuler, self).__init__(aug_params)
            flow_root = osp.join(root, split, 'flow')
            image_root_1 = osp.join(root, split, 'image1')
            image_root_2 = osp.join(root, split, 'new_image2')

            if split == 'test':
                self.is_test = True

            image_list_1 = sorted(glob(osp.join(image_root_1, '*.png')))
            image_list_2 = sorted(glob(osp.join(image_root_2, '*.png')))
            for i in range(len(image_list_1)):
                # 图片两两拿出来
                self.image_list += [[image_list_1[i], image_list_2[i]]]
                self.extra_info += [(scene, i)]  # scene and frame_id
            # 其他数据集都是只用training来测试，我这里的话，就同时尝试在training和test上面都训练
            # if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, '*.flo')))

        # 不更新，只保存了一张image1
        elif flag == 'notUpdated':
            # 一般都是在拿训练集
            super(SteelRuler, self).__init__(aug_params)
            flow_root = osp.join(root, split, 'flow')
            image_root_1 = osp.join(root, split, 'image1')
            image_root_2 = osp.join(root, split, 'new_image2')

            if split == 'test':
                self.is_test = True

            image_list_1 = sorted(glob(osp.join(image_root_1, '*.png')))
            image_list_2 = sorted(glob(osp.join(image_root_2, '*.png')))
            for i in range(len(image_list_2)):
                # 图片两两拿出来
                self.image_list += [[image_list_1[0], image_list_2[i]]]
                self.extra_info += [(scene, i)]  # scene and frame_id
            # 其他数据集都是只用training来测试，我这里的话，就同时尝试在training和test上面都训练
            # if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, '*.flo')))


class FiveFloorsFrameworks(FlowDataset):
    def __init__(self, flag, data_type, scene, root, aug_params=None, split='training'):

        if flag == 'Updated':
            # 一般都是在拿训练集
            super(FiveFloorsFrameworks, self).__init__(aug_params)
            flow_root = osp.join(root, split, 'flow')
            image_root_1 = osp.join(root, split, 'image1')
            image_root_2 = osp.join(root, split, 'new_image2')

            if split == 'test':
                self.is_test = True

            image_list_1 = sorted(glob(osp.join(image_root_1, '*.png')))
            image_list_2 = sorted(glob(osp.join(image_root_2, '*.png')))
            for i in range(len(image_list_1)):
                # 图片两两拿出来
                self.image_list += [[image_list_1[i], image_list_2[i]]]
                self.extra_info += [(scene, i)]  # scene and frame_id
            # 其他数据集都是只用training来测试，我这里的话，就同时尝试在training和test上面都训练
            # if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, '*.flo')))

        elif flag == 'notUpdated':
            # 一般都是在拿训练集
            super(FiveFloorsFrameworks, self).__init__(aug_params)
            flow_root = osp.join(root, split, 'flow')
            image_root_1 = osp.join(root, split, 'image1')
            image_root_2 = osp.join(root, split, 'new_image2')

            if split == 'test':
                self.is_test = True

            image_list_1 = sorted(glob(osp.join(image_root_1, '*.png')))
            image_list_2 = sorted(glob(osp.join(image_root_2, '*.png')))
            for i in range(len(image_list_2)):
                # 图片两两拿出来
                self.image_list += [[image_list_1[0], image_list_2[i]]]
                self.extra_info += [(scene, i)]  # scene and frame_id
                # 其他数据集都是只用training来测试，我这里的话，就同时尝试在training和test上面都训练
                # if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, '*.flo')))


def fetch_dataloader(args, generated_dataset_path, flag, data_type, scene, TRAIN_DS='C+T+K/S'):
    """ Create the data loader for the corresponding training set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        # train_dataset = FlyingChairs(aug_params, split='training')
        train_dataset = FlyingChairs(aug_params)
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        # clean_dataset = FlyingThings3D_Nori(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = final_dataset
    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100 * sintel_clean + 100 * sintel_final + things
    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    # 我们自己的数据集
    elif args.stage == 'steel_ruler':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = SteelRuler(flag, data_type, scene, generated_dataset_path, aug_params, split='training')

    elif args.stage == 'five_floors_frameworks':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = FiveFloorsFrameworks(flag, data_type, scene, generated_dataset_path, aug_params, split='training')

    else:
        raise ValueError('')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader



