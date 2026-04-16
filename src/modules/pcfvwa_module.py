import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.pointnet_stack_utils import *
from src.utils.model_utils import *
from src.modules.pointcloud_encoder_face import PointCloudEncoderFace
import numpy as np


class StackSAModuleMSG(nn.Module):
    def __init__(self, radii, nsamples, mlps, pool_method="max_pool"):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super(StackSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(QueryAndGroup(radius, nsample, use_xyz=True))
            mlp_spec = mlps[i]
            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend(
                    [
                        nn.Conv2d(
                            mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False
                        ),
                        nn.BatchNorm2d(mlp_spec[k + 1]),
                        nn.ReLU(),
                    ]
                )
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        xyz,
        xyz_batch_cnt,
        new_xyz,
        new_xyz_batch_cnt,
        features=None,
    ):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, C, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(
                dim=0
            )  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[k](new_features)  # (1, C, M1 + M2 ..., nsample)
            if self.pool_method == "max_pool":
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(
                    dim=-1
                )  # (1, C, M1 + M2 ...)
            elif self.pool_method == "avg_pool":
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(
                    dim=-1
                )  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features


class PCFVWAModule(nn.Module):
    def __init__(self, config, output_dim=384):
        super(PCFVWAModule, self).__init__()
        self.pointcloud_encoder = PointCloudEncoderFace(output_dim)
        self.local_feature_module = StackSAModuleMSG(
            radii=config["radii"],
            nsamples=config["nsamples"],
            mlps=config["mlps"],
            pool_method="max_pool",
        )
        self.drop = nn.Dropout(config["drop_radio"])
        input_dim = np.sum(np.array(config["mlps"])[:, 1])
        self.shared_fc = LinearBN(input_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, pc_sparse, vs_sparse, batch_size):
        pc_features = self.pointcloud_encoder(pc_sparse, batch_size)
        pc_coords_sparse, vs_coords_sparse = pc_sparse.C, vs_sparse.C
        pts_coords = pc_coords_sparse[:, 1:].contiguous().to(torch.float32)
        vs_coords = vs_coords_sparse[:, 1:].contiguous().to(torch.float32)
        pts_batch_idx = pc_coords_sparse[:, 0].to(torch.int32)
        vs_batch_idx = vs_coords_sparse[:, 0].to(torch.int32)
        pts_batch_cnt = torch.bincount(pts_batch_idx, minlength=batch_size).to(
            torch.int32
        )
        vs_batch_cnt = torch.bincount(vs_batch_idx, minlength=batch_size).to(
            torch.int32
        )
        features = torch.cat(pc_features, dim=0)

        assert pts_coords.shape[0] == features.shape[0] == pts_batch_cnt.sum()

        vs_xyz, vs_new_features = self.local_feature_module(
            pts_coords, pts_batch_cnt, vs_coords, vs_batch_cnt, features
        )
        vs_new_features = self.drop(self.shared_fc(vs_new_features))

        vs_batch_cnt = tuple(vs_batch_cnt.tolist())
        vs_xyz = torch.split(vs_xyz, vs_batch_cnt)
        vs_new_features = torch.split(vs_new_features, vs_batch_cnt)
        return vs_xyz, vs_new_features
