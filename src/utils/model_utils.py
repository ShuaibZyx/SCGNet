import os
import copy
import pc_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.autograd import Function


class Conv2ds(nn.Sequential):
    def __init__(self, cns):
        super().__init__()
        for i in range(len(cns) - 1):
            in_cn, out_cn = cns[i], cns[i + 1]
            self.add_module("conv%d" % (i + 1), Conv2dBN(in_cn, out_cn))


class Conv2dBN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x), inplace=True))


class Conv1ds(nn.Sequential):
    def __init__(self, cns):
        super().__init__()
        for i in range(len(cns) - 1):
            in_cn, out_cn = cns[i], cns[i + 1]
            self.add_module("conv%d" % (i + 1), Conv1dBN(in_cn, out_cn))


class Conv1dBN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.bn = nn.BatchNorm1d(out_channel)
        self.conv = nn.Conv1d(in_channel, out_channel, 1)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x), inplace=True))


class Linears(nn.Sequential):
    def __init__(self, cns):
        super().__init__()
        for i in range(len(cns) - 1):
            in_cn, out_cn = cns[i], cns[i + 1]
            self.add_module("linear%d" % (i + 1), LinearBN(in_cn, out_cn))


class LinearBN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.bn = nn.BatchNorm1d(out_channel)
        self.conv = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x), inplace=True))


def load_params_with_optimizer(
    net, filename, to_cpu=False, optimizer=None, logger=None
):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    logger.info("==> Loading parameters from checkpoint")
    checkpoint = torch.load(filename)
    epoch = checkpoint.get("epoch", -1)
    it = checkpoint.get("it", 0.0)

    net.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        logger.info("==> Loading optimizer parameters from checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    logger.info("==> Done")

    return it, epoch


def load_params(net, filename, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError
    if logger is not None:
        logger.info("==> Loading parameters from checkpoint")
    checkpoint = torch.load(filename)

    net.load_state_dict(checkpoint["model_state"])
    if logger is not None:
        logger.info("==> Done")


class DBSCANCluster(Function):

    @staticmethod
    def forward(ctx, eps: float, min_pts: int, point: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param eps: float, dbscan eps
        :param min_pts: int, dbscan core point threshold
        :param point: (B, N, 3) xyz coordinates of the points
        :return:
            idx: (B, N) cluster idx
        """
        point = point.contiguous()

        B, N, _ = point.size()
        idx = torch.cuda.IntTensor(B, N).zero_() - 1

        pc_util.dbscan_wrapper(B, N, eps, min_pts, point, idx)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, grad_out):
        return ()


dbscan_cluster = DBSCANCluster.apply


class GetClusterPts(Function):

    @staticmethod
    def forward(ctx, point: torch.Tensor, cluster_idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param point: (B, N, 3) xyz coordinates of the points
        :param cluster_idx: (B, N) cluster idx
        :return:
            key_pts: (B, M, 3) cluster center pts, M is max_num_cluster_class
            num_cluster: (B, M) cluster num, num of pts in each cluster class
        """
        cluster_idx = cluster_idx.contiguous()

        B, N = cluster_idx.size()
        M = torch.max(cluster_idx) + 1
        key_pts = torch.cuda.FloatTensor(B, M, 3).zero_()
        num_cluster = torch.cuda.IntTensor(B, M).zero_()
        pc_util.cluster_pts_wrapper(B, N, M, point, cluster_idx, key_pts, num_cluster)
        key_pts[key_pts * 1e4 == 0] = -1e1
        ctx.mark_non_differentiable(key_pts)
        ctx.mark_non_differentiable(num_cluster)
        return key_pts, num_cluster

    @staticmethod
    def backward(ctx, grad_out):
        return ()


get_cluster_pts = GetClusterPts.apply


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Masks logits such that logits not in top-k are small

    Args:
        logits: tensor representing network predictions
        k: how many logits to not filter out

    Returns:
        logits: logits with top-k logits remaining intact
    """

    if k == 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        k_largest = torch.min(values)
        logits = torch.where(
            torch.le(logits, k_largest), torch.ones_like(logits) * -1e9, logits
        )
        return logits


def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Masks logits using nucleus (top-p) sampling

    Args:
        logits: Network predictions
        top-p: What probability of the predictions we want to keep unmasked
    Returns:
        logits: logits with top-p prob mass remaining intact
    """

    if p == 1:
        return logits
    else:
        logit_shape = logits.shape
        seq, dim = logit_shape[1], logit_shape[2]
        logits = torch.reshape(logits, [-1, dim])
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cumulative_probs = torch.roll(cumulative_probs, 1, -1)
        cumulative_probs[:, 0] = 0
        sorted_indices_to_remove = (cumulative_probs > p).to(logits.dtype)
        logits_ordered = sorted_logits - sorted_indices_to_remove * 1e9
        logits = logits_ordered.gather(1, sorted_indices.argsort(-1))
        return torch.reshape(logits, [-1, seq, dim])


def get_clones(module: nn.Module, N: int) -> ModuleList:
    """Clone a module n-times

    Args:
        module: module to clone
        N: how many times to clone the module
    Returns:
        module_list: ModuleList that contains module n times
    """
    return ModuleList([copy.deepcopy(module) for _ in range(N)])


def embedding_to_padding(emb: torch.Tensor) -> torch.Tensor:
    """
    Calculates the padding mask based on which all embeddings are all zero.
    Args:
        emb: A Tensor with shape [..., depth]
    Returns:
        A float tensor with shape [...]. Each element is 1 if its corresponding embedding vector is all zero, and is 0 otherwise.
    """
    emb_sum = torch.sum(torch.abs(emb), dim=-1)
    float_emb_sum = emb_sum.to(torch.float32)
    return (float_emb_sum == 0.0).transpose(0, 1)
