import numpy as np
import torch
import random
from typing import Dict


# 对点云进行随机旋转
class PointcloudRotate(object):
    def __init__(self, angle=2 * np.pi):
        self.angle = angle

    def __call__(self, item_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rotation_angle = np.random.uniform(0, self.angle)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = torch.Tensor([[cosval, -sinval], [sinval, cosval]])
        for key in item_dict.keys():
            pc = item_dict[key]
            pc_2d = pc[:, :2]
            rotated_2d = torch.matmul(pc_2d, rotation_matrix.T)
            rotated_3d = torch.cat((rotated_2d, pc[:, 2].view(-1, 1)), dim=1)
            pc = rotated_3d.to(dtype=torch.float32, device=pc.device)
            item_dict[key] = pc
        return item_dict


# 对点云进行随机缩放
class PointcloudScale(object):
    def __init__(self, scale_low=0.8, scale_high=1.2):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, item_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        xyz = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        xyz = torch.Tensor(xyz)
        for key in item_dict.keys():
            pc = item_dict[key]
            pc[:, 0:3] = torch.mul(pc[:, 0:3], xyz)
            item_dict[key] = pc
        return item_dict


# 对点云进行随机平移
class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, item_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        xyz = np.random.uniform(
            low=-self.translate_range, high=self.translate_range, size=[3]
        )
        xyz = torch.Tensor(xyz)
        for key in item_dict.keys():
            pc = item_dict[key]
            pc[:, 0:3] = pc[:, 0:3] + xyz
            item_dict[key] = pc
        return item_dict


# 中心化、归一化点云
class PointcloudCenterNormalization(object):
    def normalize_pc(self, pc: torch.Tensor) -> torch.Tensor:
        pc_min, pc_max = torch.min(pc), torch.max(pc)
        scale = torch.sqrt(torch.sum((pc_max - pc_min) ** 2))
        pc_scale = pc / scale
        return pc_scale

    def center_pc(self, pc: torch.Tensor) -> torch.Tensor:
        pc_min, pc_max = torch.min(pc), torch.max(pc)
        center = 0.5 * (pc_min + pc_max)
        pc_center = pc - center
        return pc_center

    def __call__(self, item_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in item_dict.keys():
            pc = item_dict[key]
            pc_center = self.center_pc(pc)
            pc_center_scale = self.normalize_pc(pc_center)
            item_dict[key] = pc_center_scale
        return item_dict


# 对点云进行随机缩放和平移
class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=0.8, scale_high=1.2, translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, item_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        xyz2 = np.random.uniform(
            low=-self.translate_range, high=self.translate_range, size=[3]
        )
        for key in item_dict.keys():
            pc = item_dict[key]
            xyz1 = torch.from_numpy(xyz1).to(dtype=torch.float32, device=pc.device)
            xyz2 = torch.from_numpy(xyz2).to(dtype=torch.float32, device=pc.device)
            pc[:, 0:3] = torch.mul(pc[:, 0:3], xyz1) + xyz2
            item_dict[key] = pc
        return item_dict


# 对点云中的每个点添加随机扰动--->无法保证pc与vertices扰动一致
class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, item_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in item_dict.keys():
            pc = item_dict[key]
            jittered_data = (
                pc.new(pc.size(1), 3)
                .normal_(mean=0.0, std=self.std)
                .clamp_(-self.clip, self.clip)
            )
            pc[:, 0:3] += jittered_data
            item_dict[key] = pc
        return item_dict


# 对点云中的每个点添加随机扰动(在点云尺度范围内)--->无法保证pc与vertices扰动一致
class PointcloudJitterAdapt(object):
    def __init__(self, max_jitter_fraction=0.1):
        self.max_jitter_fraction = max_jitter_fraction

    def __call__(self, item_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in item_dict.keys():
            pc = item_dict[key]
            # 计算点云在各个维度上的直径
            max_vals = torch.max(pc, dim=0)[0]
            min_vals = torch.min(pc, dim=0)[0]
            diameters = max_vals - min_vals

            # 避免除以零的情况，如果直径为零则设置为1
            diameters[diameters == 0] = 1

            # 计算每个维度上顶点可以移动的最大距离
            max_jitter_distances = diameters * self.max_jitter_fraction
            std = max_jitter_distances[None, :]

            # 生成随机扰动
            jittered_data = torch.normal(mean=torch.zeros_like(std), std=std)
            pc[:, 0:3] = pc + jittered_data[:, 0:3]
            item_dict[key] = pc
        return item_dict


# 以下还未修改
# 随机丢弃点云中的点，模拟点云的不完整性
class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.5):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            cur_pc = pc[:, :]
            cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(
                len(drop_idx), 1
            )  # set to the first point
            pc[:, :] = cur_pc

        return pc


# 对点云进行随机水平翻转
class RandomHorizontalFlip(object):
    def __init__(self, upright_axis="z", is_temporal=False):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {"x": 0, "y": 1, "z": 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = torch.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]
        return coords
