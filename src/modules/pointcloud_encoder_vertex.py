import math
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.utils import kaiming_normal_


class BasicBlockInstanceNorm(nn.Module):
    expansion = 1
    NORM = ME.MinkowskiInstanceNorm

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        dimension=-1,
    ):
        super(BasicBlockInstanceNorm, self).__init__()
        assert dimension > 0
        self.conv1 = ME.MinkowskiConvolution(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
        )
        self.norm1 = self.NORM(planes)
        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
        )
        self.norm2 = self.NORM(planes)
        self.relu = ME.MinkowskiLeakyReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNetBase(nn.Module):
    BLOCK = None
    NORM = None
    LAYERS = None
    PLANES = None

    def __init__(self, in_channels, out_channels, D=3, pooling=True):
        nn.Module.__init__(self)
        self.D = D

        assert self.BLOCK is not None, "BLOCK Module should be not None"
        assert self.NORM is not None, "NORM Module should be not None"
        assert self.LAYERS is not None, "LAYERS List should be not None"
        assert self.PLANES is not None, "PLANES List should be not None"

        self.pooling = pooling
        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels,
                self.PLANES[0],
                kernel_size=3,
                stride=1,
                dimension=D,
                expand_coordinates=False,
            ),
            self.NORM(self.PLANES[0]),
            ME.MinkowskiLeakyReLU(inplace=True),
        )
        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.PLANES[1], self.LAYERS[0], stride=1
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.PLANES[2], self.LAYERS[1], stride=1
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], out_channels, self.LAYERS[2], stride=1
        )

        self.layer4 = self.BLOCK(
            out_channels,
            out_channels,
            stride=1,
            dimension=self.D,
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiInstanceNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        in_planes,
        out_planes,
        blocks,
        stride=1,
        dilation=1,
    ):
        layers = []
        for _ in range(0, blocks):  # residual blocks
            layers.append(
                block(
                    in_planes, in_planes, stride=1, dilation=dilation, dimension=self.D
                )
            )
        layers.append(
            nn.Sequential(  # conv1x1 to increase feature size
                ME.MinkowskiConvolution(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                self.NORM(out_planes),
                ME.MinkowskiLeakyReLU(inplace=True),
            )
        )
        # pooling to reduce spatial resolution
        if self.pooling:
            layers.append(
                ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=self.D)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# 自定义的ResNet层，比ResNet18更小
class ResNetCustom(ResNetBase):
    BLOCK = BasicBlockInstanceNorm
    NORM = ME.MinkowskiInstanceNorm
    LAYERS = (3, 3, 3)
    PLANES = (64, 128, 256)


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        voxel_dim,
        num_pos_feats=32,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.voxel_dim = voxel_dim
        assert voxel_dim > 0

    def forward(self, voxel_coord):
        if self.normalize:
            voxel_coord = self.scale * voxel_coord / (self.voxel_dim - 1)

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=voxel_coord.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos = voxel_coord[:, :, None] / dim_t
        pos_x = pos[:, 0]
        pos_y = pos[:, 1]
        pos_z = pos[:, 2]  # in shape[n, pos_feature_dim]
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_z = torch.stack(
            (pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos = torch.cat((pos_x, pos_y, pos_z), dim=1)
        return pos


class SparseBackboneMinkowski(ME.MinkowskiNetwork):
    def __init__(self, output_dim=384, pooling=True):
        super(SparseBackboneMinkowski, self).__init__(3)  # dimension = 3
        self.sparseModel = ResNetCustom(
            in_channels=4,
            out_channels=output_dim,
            D=3,
            pooling=pooling,
        )

    def forward(self, x):
        input = ME.SparseTensor(coordinates=x[0], features=x[1])
        out = self.sparseModel(input)
        return out


class PointCloudEncoderVertex(nn.Module):
    def __init__(self, output_dim: int = 384, pooling: bool = True) -> None:
        super(PointCloudEncoderVertex, self).__init__()
        self.emb_net = SparseBackboneMinkowski(output_dim, pooling)
        assert (
            output_dim % 3 == 0
        ), "output_dim must be divisible by 3 for position encoding"
        self.position_encoding = PositionEmbeddingSine3D(
            32, output_dim // 3, normalize=True
        )

    def forward(self, coords, feats, batch_size):
        output = self.emb_net([coords, feats])

        sparse_locations = output.C
        sparse_features = output.F

        batch_idx = sparse_locations[:, 0]
        sparse_locations = sparse_locations[:, 1:] // output.tensor_stride[0]

        voxel_pos_embedding = self.position_encoding(sparse_locations)

        batch_number_samples = torch.bincount(batch_idx, minlength=batch_size).to(
            torch.int32
        )
        batch_number_samples = tuple(batch_number_samples.tolist())
        voxel_features = torch.split(sparse_features, batch_number_samples)
        voxel_pos_embedding = torch.split(voxel_pos_embedding, batch_number_samples)

        pointcloud_features = list(map(torch.add, voxel_features, voxel_pos_embedding))

        return pointcloud_features
