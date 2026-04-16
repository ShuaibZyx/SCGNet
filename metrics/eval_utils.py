import sys

sys.path.append("../")
import csv
import torch
import trimesh
import numpy as np
import open3d
from extensions.emd.emd_module import emdModule

emd_module_fun = emdModule()


def get_pc_open3d(pc):
    ptcloud = open3d.geometry.PointCloud()
    ptcloud.points = open3d.utility.Vector3dVector(pc)
    return ptcloud


def get_pc_tensor(pc):
    pc = torch.from_numpy(pc).to(dtype=torch.float32, device="cuda")
    return pc


def load_xyz(xyz_file):
    """Load point cloud file"""
    pts = []
    with open(xyz_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            xyz = line.split(" ")
            pts.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    return np.array(pts).astype(np.float32)


def load_obj(poly_file):
    vs = []
    fs = []
    with open(poly_file) as f:
        for line in f.readlines():
            if line[0] == "v":
                vxyz = line.strip().split(" ")
                vs.append([float(vxyz[1]), float(vxyz[2]), float(vxyz[3])])
            elif line[0] == "f":
                a = line.strip().split(" ")
                fs.append([int(f) - 1 for f in a[1:]])
    return np.array(vs), fs


def pc_fscore(gt_pc, pred_pc, threshold=0.01):
    gt_pc_open3d = get_pc_open3d(gt_pc)
    pred_pc_open3d = get_pc_open3d(pred_pc)
    # 计算顶点的度量

    D_S1 = gt_pc_open3d.compute_point_cloud_distance(pred_pc_open3d)
    D_S2 = pred_pc_open3d.compute_point_cloud_distance(gt_pc_open3d)

    # 距离阈值 threshold 的精度
    Pd = (sum(d <= threshold for d in D_S1) / float(len(D_S1))) if len(D_S1) > 0 else 0
    # 距离阈值 threshold 的召回率
    Rd = (sum(d <= threshold for d in D_S2) / float(len(D_S2))) if len(D_S2) > 0 else 0

    F_Score = (2 * Pd * Rd / (Pd + Rd)) if (Pd + Rd) > 0 else 0

    return F_Score, Pd, Rd


def compute_rmse_open3d(pts, pred_pc):
    pts_open3d = get_pc_open3d(pts)
    pred_pc_open3d = get_pc_open3d(pred_pc)
    distance = pts_open3d.compute_point_cloud_distance(pred_pc_open3d)
    rmse = np.sqrt(np.mean(np.array(distance) ** 2))
    return rmse


def emd_distance(gt_pc, pred_pc, eps=0.005, iterations=100):
    gt_pc = get_pc_tensor(gt_pc).unsqueeze(0)
    pred_pc = get_pc_tensor(pred_pc).unsqueeze(0)
    dist, _ = emd_module_fun(pred_pc, gt_pc, eps, iterations)
    emd_out = torch.mean(torch.sqrt(dist)).item()
    return emd_out


def apply_normalize(mesh):
    """
    normalize mesh to [-1, 1]
    """
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale * 0.95)

    return mesh


def normalize_points(points):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    center = (min_coords + max_coords) / 2
    scale = (max_coords - min_coords).max()

    normalized = (points - center) / scale * 0.95
    return normalized


def sample_pc(mesh_path, pc_num, with_normal=False):

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    mesh = apply_normalize(mesh)

    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points

    points, face_idx = mesh.sample(50000, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]

    return np.array(pc_normal)


# 保存至csv文件
def save_results_to_csv(results, filename, accelerated_cd=True):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = list(results.keys())
        writer = csv.writer(csvfile)
        # 写入列名
        writer.writerow(["指标", "原始值", "处理精度"])
        # 获取每个指标的值，如果是tensor则获取其数值
        for key in fieldnames:
            value = results[key]
            if isinstance(value, torch.Tensor):
                # 如果是加速模式且tensor在GPU上，先将其移到CPU
                if accelerated_cd and value.is_cuda:
                    value = value.cpu()
                value_item = value.item()
            else:
                value_item = value
            truncated_precision = round(value_item, 4)
            writer.writerow([key, value_item, truncated_precision])
