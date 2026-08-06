import os
from os.path import join, splitext, basename
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import trimesh
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import pickle
from glob import glob
import torch
import networkx as nx
from datetime import date
from scipy.spatial import KDTree
from plyfile import PlyData, PlyElement


def test_mesh_xyz_match():
    mesh_file = "/home/shuaib/work/Vaihingen/00000.obj"
    pts_file = "/home/shuaib/work/Vaihingen/00000.ply"
    vertices, faces = read_polygon_mesh(mesh_file)
    pts = read_pts_ply(pts_file)

    centered_vertices, vertices_center = center_vertices(vertices)
    scaled_vertices, vertices_scale = normalize_vertices(centered_vertices)
    scaled_pts = (pts - vertices_center) / vertices_scale

    print(f"pts = {pts.shape}")
    mesh = trimesh.load_mesh(mesh_file)
    mesh.vertices = (mesh.vertices - vertices_center) / vertices_scale

    # 为Mesh添加缩放变换-放大1.1倍
    scale_matrix = trimesh.transformations.scale_matrix(1.1)
    mesh = mesh.apply_transform(scale_matrix)

    # 检查哪些点位于Mesh内部
    pts_contain_mask = mesh.contains(scaled_pts)
    # 获取位于Mesh内部的点云数据
    pts_contain = scaled_pts[pts_contain_mask]
    print(f"pts_contain = {pts_contain.shape}")
    percentage_contain = (pts_contain.shape[0] / pts.shape[0]) * 100
    print(f"Percentage of points contain mesh: {percentage_contain:.2f}%")

    # 恢复原始大小
    mesh = trimesh.load_mesh(mesh_file)
    mesh.vertices = (mesh.vertices - vertices_center) / vertices_scale

    surface_samples, _ = trimesh.sample.sample_surface(mesh, 10000)

    # 计算原始点云数据与采样点之间的距离
    distances = cdist(pts_contain, surface_samples, metric="euclidean")

    # 保留仅在Mesh平面上的点云数据
    points_on_surface = pts_contain[np.any(distances < 0.015, axis=1)]
    print(f"points_on_surface = {points_on_surface.shape}")

    # 计算位于Mesh内部的点云数据占总点云数据的百分比
    percentage_surface = (points_on_surface.shape[0] / pts.shape[0]) * 100

    print(f"Percentage of points surface mesh: {percentage_surface:.2f}%")

    visualize_points(surface_samples, points_on_surface, "bounds.obj")


# 加载Mesh文件
def read_polygon_mesh(mesh_file, face_split=False):
    vertices, faces = [], []
    with open(mesh_file) as file:
        for line in file.readlines():
            line = line.strip().split(" ")
            if line[0] == "v":
                vertices.append(line[1:])
            elif line[0] == "f":
                if face_split:
                    faces.append([int(f.split("//")[0]) - 1 for f in line[1:]])
                else:
                    faces.append([int(f) - 1 for f in line[1:]])
    vertices = np.array(vertices, dtype=np.float32)
    return vertices, faces


def read_triangle_mesh(mesh_file):
    """Load mesh file"""
    mesh = trimesh.load_mesh(mesh_file)
    vertices = mesh.vertices
    faces = mesh.faces
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def read_edge_mesh(mesh_file):
    vertices, edges = [], set()
    with open(mesh_file) as file:
        for line in file.readlines():
            line = line.strip().split(" ")
            if line[0] == "v":
                vertices.append(line[1:])
            elif line[0] == "f":
                face = [int(f) - 1 for f in line[1:]]
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                    edges.add(edge)

    edges = np.array(list(edges))
    vertices = np.array(vertices, dtype=np.float32)
    return vertices, edges


def read_pts_xyz(xyz_file):
    """Load pointcloud file"""
    pts = []
    with open(xyz_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            xyz = line.split(" ")
            pts.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    return np.array(pts, dtype=np.float32)


def read_pts_ply(ply_file):
    """Load pointcloud file"""
    mesh = trimesh.load(ply_file)
    return np.array(mesh.vertices, dtype=np.float32)


def xyz_to_ply(xyz_file_path, ply_save_path):
    pts = read_pts_xyz(xyz_file_path)
    vertex = np.array(pts, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyElement.describe(vertex, "vertex", comments=["vertices"])
    PlyData([el], text=True).write(ply_save_path)


def ply_to_xyz(ply_file_path, xyz_save_path):
    mesh = trimesh.load(ply_file_path)
    pts = mesh.vertices
    write_pts(pts, xyz_save_path)


def center_vertices_ground(vertices):
    # 计算 X 和 Y 的包围盒中心
    x_center, y_center = np.mean(vertices[:, :2], axis=0)

    # 计算 Z 的最小值
    z_min = np.min(vertices[:, 2])

    # 平移顶点
    vertices[:, :2] -= [x_center, y_center]  # 对齐 X 和 Y 中心到 (0, 0)
    vertices[:, 2] -= z_min  # 对齐 Z 最小值到 0

    # 返回对齐后的顶点和移动信息
    center_info = (x_center, y_center, z_min)
    return vertices, center_info


def center_vertices(vertices):
    """Translate vertices so that the bounding box is centered at zero

    Args:
        vertices: np array of shape (num_vertices, 3)

    Returns:
        centered_vertices: centered vertices in array of shape (num_vertices, 3)
    """
    vert_min = np.min(vertices, axis=0)
    vert_max = np.max(vertices, axis=0)
    center = 0.5 * (vert_min + vert_max)
    centered_vertices = vertices - center
    return centered_vertices, center


def normalize_vertices(vertices):
    """Scale vertices so that the long diagonal of the bounding box is one

    Args:
        vertices: unscaled vertices of shape (num_vertices, 3)
    Returns:
        scaled_vertices: scaled vertices of shape (num_vertices, 3)
    """
    vert_min = np.min(vertices, axis=0)
    vert_max = np.max(vertices, axis=0)
    extents = vert_max - vert_min
    scale = np.sqrt(np.sum(extents**2))
    scaled_vertices = vertices / scale
    return scaled_vertices, scale


def write_obj(vertices, faces, file_path) -> None:
    """Writes vertices and faces to .obj file to represent 3D object
    Args:
        vertices: array of shape (num_vertices, 3) representing vertex indices
        faces: List of vertex indices representing vertex connectivity
        file_path: Where to save .obj file
    """
    if faces is not None:
        if min(min(faces)) == 0:
            f_add = 1
        else:
            f_add = 0
    with open(file_path, "w") as f:
        for v in vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            line = "f"
            for i in face:
                line += " {}".format(i + f_add)
            line += "\n"
            f.write(line)


def write_pts(pts, file_path):
    with open(file_path, "w") as f:
        [f.write(f"{v1} {v2} {v3}\n") for v1, v2, v3 in pts[:, :3]]


def farthest_pts_sample(point: np.ndarray, npoint: int = 2048):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def compute_testset_vs2pts_distance(dataset_root_dir):
    distance = []
    dataset_mesh_path = join(dataset_root_dir, "meshes")
    dataset_pts_path = join(dataset_root_dir, "pointclouds")
    testset_txt = join(dataset_root_dir, "val.txt")
    with open(testset_txt, "r") as tf:
        # 读取每一行的文件名
        testset_files = [line.strip() for line in tf.readlines()]
        for i, filename in tqdm(
            enumerate(testset_files),
            total=len(testset_files),
            desc="Compute Testset VS2PTS Distance",
        ):
            mesh_file = join(dataset_mesh_path, f"{filename}.obj")
            pts_file = join(dataset_pts_path, f"{filename}.xyz")

            vs, _ = read_polygon_mesh(mesh_file)
            pts = read_pts_xyz(pts_file)

            pts_max_z = np.max(pts[:, 2])
            vs_max_z = np.max(vs[:, 2])
            distance.append(pts_max_z - vs_max_z)
    return -np.mean(distance)


def add_pts_bottom(
    pts: np.array,
    vs: np.array = None,
    split: str = "train",
    fps: bool = False,
    fps_radio: int = 100,
) -> np.array:
    if split == "train" and vs is not None:
        minz = -np.max(vs[:, 2])
    else:
        minz = -np.max(pts[:, 2])
    added_pts = pts.copy()
    added_pts[:, 2] = minz
    if fps:
        add_pts_num = max(round(pts.shape[0] / fps_radio), 104)
        added_pts = farthest_pts_sample(added_pts, add_pts_num)
    pts = np.vstack((pts, added_pts))
    return pts


def face_to_cycles(faces):
    g = nx.Graph()
    for v in range(len(faces) - 1):
        g.add_edge(faces[v], faces[v + 1])
    g.add_edge(faces[-1], faces[0])
    return list(nx.cycle_basis(g))


def torch_lexsort(a, dim=-1):
    assert dim == -1
    assert a.ndim == 2
    a_unq, inv = torch.unique(a.flip(0), dim=dim, sorted=True, return_inverse=True)
    return torch.argsort(inv)


def argmin(arr):
    return min(range(len(arr)), key=lambda x: arr[x])


def remove_face_cycles(faces):
    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g. [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        f_list = f if isinstance(f, list) else f.tolist()
        cliques = face_to_cycles(f_list)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts
            if c_length > 2:
                d = argmin(c)
                # Cyclically permute faces so that the first index is the smallest
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    return sub_faces


def process_mesh(vertices, faces):
    vertices = torch.from_numpy(vertices)

    # 移除重复的顶点并获取逆索引
    vertices, inv = torch.unique(vertices, dim=0, return_inverse=True)

    # 重新索引面以匹配重新排序的顶点
    faces = [torch.argsort(inv[f]) for f in faces]
    faces = remove_face_cycles(faces)

    # 按照最低顶点索引对面进行排序
    faces.sort(key=lambda f: tuple(sorted(f)))
    faces = [torch.Tensor(f).to(torch.int64) for f in faces]

    # 移除退化的面后，一些顶点可能不再被引用
    num_verts = vertices.shape[0]
    vert_connected = torch.eq(
        torch.arange(num_verts)[:, None], torch.hstack(faces)[None]
    ).any(dim=-1)
    vertices = vertices[vert_connected]

    # 重新索引面以匹配重新排序的顶点
    vert_indices = torch.arange(num_verts) - torch.cumsum(
        (1 - vert_connected.to(torch.int32)), dim=-1
    )
    faces = [vert_indices[f].tolist() for f in faces]

    vertices = vertices.numpy()
    return vertices, faces


def visualize_points(points, vertices, vis_path):
    with Path(vis_path).open("w") as f:
        # 写入点云，颜色设置为白色（255, 255, 255）
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]} 255 255 255 \n")

        # 写入顶点，颜色设置为红色（255, 0, 0）
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]} 255 0 0 \n")


def make_dataset_split_random(dataset_save_dir, val_ratio=0.1):
    dataset_json_path = join(dataset_save_dir, "info.json")
    with open(dataset_json_path, "r") as dj:
        dataset_json = json.load(dj)
        filenames = list(dataset_json.keys())

        np.random.shuffle(filenames)
        val_size = int(len(filenames) * val_ratio)

        val_filenames = filenames[:val_size]
        train_filenames = filenames[val_size:]

        # 排序文件名
        val_filenames.sort()
        train_filenames.sort()

        val_file = join(dataset_save_dir, "val.txt")
        train_file = join(dataset_save_dir, "train.txt")

        with open(train_file, "w") as tf:
            for filename in train_filenames:
                tf.write(filename + "\n")

        with open(val_file, "w") as vf:
            for filename in val_filenames:
                vf.write(filename + "\n")

    return len(train_filenames), len(val_filenames)


def make_dataset_figure(dataset_save_dir):
    dataset_name = basename(dataset_save_dir).split("_")[0]
    dataset_json_path = join(dataset_save_dir, "info.json")
    vertex_count = []
    face_count = []
    pts_count = []
    with open(dataset_json_path, "r") as dj:
        dataset_json = json.load(dj)
    for filename, info in dataset_json.items():
        # 统计数据集的顶点数量分布、面片数量分布、点云数量分布
        vertex_count.append(info["count_verts"])
        face_count.append(info["count_faces"])
        pts_count.append(info["count_pts"])

    # 创建图表
    plt.figure(figsize=(18, 6))

    # 绘制顶点数量分布图
    plt.subplot(1, 3, 1)
    plt.hist(vertex_count, bins=30, color="skyblue", edgecolor="black")
    plt.xlabel("Vertex Count", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 绘制面片数量分布图
    plt.subplot(1, 3, 2)
    plt.hist(face_count, bins=30, color="lightgreen", edgecolor="black")
    plt.xlabel("Face Count", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 绘制点云数量分布图
    plt.subplot(1, 3, 3)
    plt.hist(pts_count, bins=30, color="salmon", edgecolor="black")
    plt.xlabel("Point Cloud Count", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 设置图表间距
    plt.tight_layout()
    figure_save_path = join(dataset_save_dir, "dataset_distribution.png")
    plt.savefig(figure_save_path, dpi=300)


def visualize_city(dataset_root_dir):
    mesh_files = glob(join(dataset_root_dir, "meshes", "*.obj"))
    mesh_list = []
    with open(join(dataset_root_dir, "info.json")) as dj:
        dataset_json = json.load(dj)

    # 收集所有顶点和面
    for mesh_file in tqdm(mesh_files, total=len(mesh_files), desc="Visualize City"):
        filename = splitext(basename(mesh_file))[0]
        mesh = trimesh.load_mesh(mesh_file)
        mesh_info = dataset_json[filename]
        mesh.vertices = mesh.vertices * mesh_info["scale"] + mesh_info["center"]
        mesh_list.append(mesh)

    dataset_mesh = trimesh.util.concatenate(mesh_list)
    dataset_mesh.fix_normals()
    dataset_mesh.remove_unreferenced_vertices()
    dataset_mesh.update_faces(dataset_mesh.unique_faces())
    dataset_mesh.fill_holes()
    dataset_mesh.export(join(dataset_root_dir, "dataset_mesh.obj"))


def visualize_city_split(dataset_root_dir, split="val"):
    with open(join(dataset_root_dir, f"{split}.txt")) as dvt:
        mesh_files = [
            join(dataset_root_dir, "meshes", f"{line.strip()}.obj")
            for line in dvt.readlines()
        ]
    mesh_list = []
    with open(join(dataset_root_dir, "info.json")) as dj:
        dataset_json = json.load(dj)

    # 收集所有顶点和面
    for mesh_file in tqdm(
        mesh_files, total=len(mesh_files), desc=f"Visualize {split} City"
    ):
        filename = splitext(basename(mesh_file))[0]
        mesh = trimesh.load_mesh(mesh_file)
        mesh_info = dataset_json[filename]
        mesh.vertices = mesh.vertices * mesh_info["scale"] + mesh_info["center"]
        mesh_list.append(mesh)

    dataset_mesh = trimesh.util.concatenate(mesh_list)
    dataset_mesh.fix_normals()
    dataset_mesh.remove_unreferenced_vertices()
    dataset_mesh.update_faces(dataset_mesh.unique_faces())
    dataset_mesh.fill_holes()
    dataset_mesh.export(join(dataset_root_dir, f"dataset_{split}_mesh.obj"))


def make_dataset_split_region(
    dataset_root_dir,
    need_val=True,
    val_grid_index=None,
):
    dataset_name = basename(dataset_root_dir).split("_")[0]
    mesh_files = glob(join(dataset_root_dir, "model", "*.obj"))
    filename_list = [splitext(basename(mesh_file))[0] for mesh_file in mesh_files]
    vertices_xy_list = []
    with open(join(dataset_root_dir, "info.json")) as dj:
        dataset_json = json.load(dj)

    # 收集所有顶点和面
    for mesh_file in tqdm(
        mesh_files, total=len(mesh_files), desc="Collect Vertices XY For Split Region"
    ):
        filename = splitext(basename(mesh_file))[0]
        vertices = read_polygon_mesh(mesh_file)[0]
        mesh_info = dataset_json[filename]
        vertices = vertices * mesh_info["scale"] + mesh_info["center"]
        vertices_xy_list.append(vertices[:, :2])

    vertices_xy_stack = np.vstack([vertices_xy for vertices_xy in vertices_xy_list])

    # 计算边界
    min_x, max_x = vertices_xy_stack[:, 0].min(), vertices_xy_stack[:, 0].max()
    min_y, max_y = vertices_xy_stack[:, 1].min(), vertices_xy_stack[:, 1].max()

    # 定义网格切割参数
    num_grid_x = 3  # X方向上的网格数量
    num_grid_y = 3  # Y方向上的网格数量

    # 计算每个网格的大小
    width = (max_x - min_x) / num_grid_x
    height = (max_y - min_y) / num_grid_y

    # 创建一个列表来存储每个网格的左上角和右下角坐标
    grids = [
        (
            (min_x + i * width, min_y + j * height),
            (min_x + (i + 1) * width, min_y + (j + 1) * height),
        )
        for i in range(num_grid_x)
        for j in range(num_grid_y)
    ]

    # 获取属于验证区域的Mesh文件
    val_files = []
    if need_val:
        (val_left_x, val_top_y), (val_right_x, val_bottom_y) = grids[val_grid_index]
        for vertices_xy, filename in tqdm(
            zip(vertices_xy_list, filename_list),
            total=len(vertices_xy_list),
            desc="Get Val Region Files",
        ):
            # 顶点在验证区域大于50%的个数则划分到验证集
            if (
                (vertices_xy[:, 0] >= val_left_x)
                & (vertices_xy[:, 0] <= val_right_x)
                & (vertices_xy[:, 1] >= val_top_y)
                & (vertices_xy[:, 1] <= val_bottom_y)
            ).sum() / vertices_xy.shape[0] > 0.5:
                val_files.append(filename)

    # print(f"total_files = {len(filename_list)}")
    # val_files.sort()
    # val_count = len(val_files)
    # val_radio = f"{val_count / len(mesh_files) * 100:.2f}%"
    # print(f"val_grid_index = {val_grid_index}")
    # print(f"val_mesh_files = {val_count}")
    # print(f"val radio = {val_radio}")

    # train_files = list(set(filename_list) - set(val_files))
    # train_files.sort()
    # train_count = len(train_files)
    # train_radio = f"{train_count / len(mesh_files) * 100:.2f}%"
    # print(f"train_mesh_files = {train_count}")
    # print(f"train radio = {train_radio}")

    # # 将验证集文件名写入txt文件
    # val_txt_path = join(dataset_root_dir, "val.txt")
    # with open(val_txt_path, "w") as vt:
    #     for filename in val_files:
    #         vt.write(filename + "\n")

    # # 将训练集文件名写入txt文件
    # train_txt_path = join(dataset_root_dir, "train.txt")
    # with open(train_txt_path, "w") as tt:
    #     for filename in train_files:
    #         tt.write(filename + "\n")

    # 可视化划分结果
    plt.figure(figsize=(20, 20), dpi=300)
    # 获取当前Axes对象
    ax = plt.gca()
    # 绘制所有顶点(#5D98D7)
    ax.scatter(vertices_xy_stack[:, 0], vertices_xy_stack[:, 1], color="#404040", s=1)
    # 绘制网格边界
    # for (left_x, top_y), (right_x, bottom_y) in grids:
    #     plt.plot(
    #         [left_x, right_x], [top_y, top_y], color="black", dashes=(10, 2)
    #     )  # 上边界
    #     plt.plot(
    #         [left_x, right_x], [bottom_y, bottom_y], color="black", dashes=(10, 2)
    #     )  # 下边界
    #     plt.plot(
    #         [left_x, left_x], [top_y, bottom_y], color="black", dashes=(10, 2)
    #     )  # 左边界
    #     plt.plot(
    #         [right_x, right_x], [top_y, bottom_y], color="black", dashes=(10, 2)
    #     )  # 右边界

    # 标记测试集区域
    if need_val:
        plt.plot(
            [val_left_x, val_right_x],
            [val_top_y, val_top_y],
            color="black",
            dashes=(10, 2),  # 上边界
        )
        plt.plot(
            [val_left_x, val_right_x],
            [val_bottom_y, val_bottom_y],
            color="black",
            dashes=(10, 2),  # 下边界
        )
        plt.plot(
            [val_left_x, val_left_x],
            [val_top_y, val_bottom_y],
            color="black",
            dashes=(10, 2),  # 左边界
        )
        plt.plot(
            [val_right_x, val_right_x],
            [val_top_y, val_bottom_y],
            color="black",
            dashes=(10, 2),  # 右边界
        )
        plt.fill_between(
            [val_left_x, val_right_x],
            val_top_y,
            val_bottom_y,
            color="red",
            alpha=0.2,
            label="Val/Test",
        )

    # 隐藏所有刻度线和标签
    ax.set_xticks([])  # 隐藏x轴刻度
    ax.set_yticks([])  # 隐藏y轴刻度

    # 隐藏所有边框（四边）
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 优化显示效果
    plt.tight_layout()
    plt.savefig(join(dataset_root_dir, "dataset_region.png"), dpi=300)

    # return train_count, train_radio, val_count, val_radio


def delete_dataset_by_region(dataset_root_dir, val_grid_index):
    mesh_files = glob(join(dataset_root_dir, "meshes", "*.obj"))
    filename_list = [splitext(basename(mesh_file))[0] for mesh_file in mesh_files]
    vertices_xy_list = []
    with open(join(dataset_root_dir, "info.json")) as dj:
        dataset_json = json.load(dj)

    # 收集所有顶点和面
    for mesh_file in tqdm(
        mesh_files, total=len(mesh_files), desc="Collect Vertices XY For Split Region"
    ):
        filename = splitext(basename(mesh_file))[0]
        vertices = read_polygon_mesh(mesh_file)[0]
        mesh_info = dataset_json[filename]
        vertices = vertices * mesh_info["scale"] + mesh_info["center"]
        vertices_xy_list.append(vertices[:, :2])

    vertices_xy_stack = np.vstack([vertices_xy for vertices_xy in vertices_xy_list])

    # 计算边界
    min_x, max_x = vertices_xy_stack[:, 0].min(), vertices_xy_stack[:, 0].max()
    min_y, max_y = vertices_xy_stack[:, 1].min(), vertices_xy_stack[:, 1].max()

    # 定义网格切割参数
    num_grid_x = 3  # X方向上的网格数量
    num_grid_y = 3  # Y方向上的网格数量

    # 计算每个网格的大小
    width = (max_x - min_x) / num_grid_x
    height = (max_y - min_y) / num_grid_y

    # 创建一个列表来存储每个网格的左上角和右下角坐标
    grids = [
        (
            (min_x + i * width, min_y + j * height),
            (min_x + (i + 1) * width, min_y + (j + 1) * height),
        )
        for i in range(num_grid_x)
        for j in range(num_grid_y)
    ]

    (val_left_x, val_top_y), (val_right_x, val_bottom_y) = grids[val_grid_index]
    delete_files = []
    for vertices_xy, filename in tqdm(
        zip(vertices_xy_list, filename_list),
        total=len(vertices_xy_list),
        desc="Get Val Region Files",
    ):
        # 顶点在验证区域大于50%的个数则划分到验证集
        if (
            (vertices_xy[:, 0] >= val_left_x)
            & (vertices_xy[:, 0] <= val_right_x)
            & (vertices_xy[:, 1] >= val_top_y)
            & (vertices_xy[:, 1] <= val_bottom_y)
        ).sum() / vertices_xy.shape[0] > 0.5:
            # 删除Mesh以及点云文件
            delete_files.append(filename)

    face_max_count = 0

    with open(join(dataset_root_dir, "delete.txt"), "w") as dt:
        for filename in delete_files:
            dt.write(filename + "\n")

    for filename in delete_files:
        mesh_info = dataset_json[filename]
        count_faces = mesh_info["count_faces"]
        if count_faces > face_max_count:
            face_max_count = count_faces
        dataset_json.pop(filename)
        mesh_file = join(dataset_root_dir, "meshes", f"{filename}.obj")
        pts_file = join(dataset_root_dir, "pointclouds", f"{filename}.xyz")
        os.remove(mesh_file)
        os.remove(pts_file)

    print(f"face_max_count = {face_max_count}")
    print(f"delete_files = {len(delete_files)}")
    print("done")


def preprocess_pkl(
    dataset_root: str,
    mesh_type: str = "Polygon",
    add_bottom: bool = True,
    npoints: int = 4096,
    test_split: str = "test"
):
    # 初始化字典以保存数据
    data = {
        "name_train": [],
        "vertices_train": [],
        "faces_train": [],
        "pts_train": [],
        "name_val": [],
        "vertices_val": [],
        "faces_val": [],
        "pts_val": [],
    }
    dataset_name = basename(dataset_root)
    # 训练集
    dataset_train_txt_path = join(dataset_root, "split", "train.txt")
    dataset_val_txt_path = join(dataset_root, "split", f"{test_split}.txt")

    pkl_save_path = join(dataset_root, f"{dataset_name}_Processed.pkl")

    dataset_mesh_path = join(dataset_root, "model")
    dataset_pts_path = join(dataset_root, "pointclouds")

    # 训练集
    with open(dataset_train_txt_path) as dtt:
        dataset_train = [line.strip() for line in dtt.readlines()]
        for filename in tqdm(dataset_train, desc="Make Dataset PKL-Train"):
            mesh_file = join(dataset_mesh_path, f"{filename}.obj")
            pts_file = join(dataset_pts_path, f"{filename}.xyz")

            if mesh_type == "Triangle":
                vertices, faces = read_triangle_mesh(mesh_file)
            else:
                vertices, faces = read_polygon_mesh(mesh_file)

            pts = read_pts_xyz(pts_file)

            if pts.shape[0] > npoints:
                idx = np.random.randint(0, pts.shape[0], npoints)
                np.random.shuffle(idx)
                pts = pts[idx]

            if add_bottom:
                pts = add_pts_bottom(pts, vertices, split="train", fps=False)

            # 保存数据
            data["name_train"].append(filename)
            data["vertices_train"].append(vertices)
            data["faces_train"].append(faces)
            data["pts_train"].append(pts)

    # 验证集
    with open(dataset_val_txt_path) as dvt:
        dataset_val = [line.strip() for line in dvt.readlines()]
        for filename in tqdm(dataset_val, desc="Make Dataset PKL-Val"):
            mesh_file = join(dataset_mesh_path, f"{filename}.obj")
            pts_file = join(dataset_pts_path, f"{filename}.xyz")

            if mesh_type == "Triangle":
                vertices, faces = read_triangle_mesh(mesh_file)
            else:
                vertices, faces = read_polygon_mesh(mesh_file)

            pts = read_pts_xyz(pts_file)

            if pts.shape[0] > npoints:
                idx = np.random.randint(0, pts.shape[0], npoints)
                np.random.shuffle(idx)
                pts = pts[idx]

            if add_bottom:
                pts = add_pts_bottom(pts, vertices, split="train", fps=False)

            # 保存数据
            data["name_val"].append(filename)
            data["vertices_val"].append(vertices)
            data["faces_val"].append(faces)
            data["pts_val"].append(pts)

    with open(pkl_save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"{dataset_name} pkl数据已保存到 {pkl_save_path}")


def preprocess_pkl_for_edge(
    dataset_root: str,
    type: str = "Edge",
    bottom: bool = True,
    fps: bool = False,
    npoints: int = 2048,
):
    # 初始化字典以保存数据
    data = {
        "name_train": [],
        "vertices_train": [],
        "edges_train": [],
        "pts_train": [],
        "name_val": [],
        "vertices_val": [],
        "edges_val": [],
        "pts_val": [],
    }
    dataset_name = basename(dataset_root)
    # 训练集
    dataset_train_txt_path = join(dataset_root, "train.txt")
    dataset_val_txt_path = join(dataset_root, "val.txt")

    pkl_save_path = join(dataset_root, f"{dataset_name}_{type}_Processed.pkl")

    dataset_mesh_path = join(dataset_root, "meshes")
    dataset_pts_path = join(dataset_root, "pointclouds")

    # 训练集
    with open(dataset_train_txt_path) as dtt:
        dataset_train = [line.strip() for line in dtt.readlines()]
    for filename in tqdm(dataset_train, desc="Make Dataset PKL-Train"):
        mesh_file = join(dataset_mesh_path, f"{filename}.obj")
        pts_file = join(dataset_pts_path, f"{filename}.xyz")

        vertices, edges = read_edge_mesh(mesh_file)
        pts = read_pts_xyz(pts_file)

        if bottom:
            pts = add_pts_bottom(pts, vertices, split="train", fps=True)

        if fps:
            pts = farthest_pts_sample(pts, npoints)

        # 保存数据
        data["name_train"].append(filename)
        data["vertices_train"].append(vertices)
        data["edges_train"].append(edges)
        data["pts_train"].append(pts)

    # 验证集
    with open(dataset_val_txt_path) as dvt:
        dataset_val = [line.strip() for line in dvt.readlines()]
    for filename in tqdm(dataset_val, desc="Make Dataset PKL-Val"):
        mesh_file = join(dataset_mesh_path, f"{filename}.obj")
        pts_file = join(dataset_pts_path, f"{filename}.xyz")

        vertices, edges = read_edge_mesh(mesh_file)
        pts = read_pts_xyz(pts_file)

        if bottom:
            pts = add_pts_bottom(pts, vertices, split="val", fps=fps)

        if fps:
            pts = farthest_pts_sample(pts, npoints)

        # 保存数据
        data["name_val"].append(filename)
        data["vertices_val"].append(vertices)
        data["edges_val"].append(edges)
        data["pts_val"].append(pts)

    with open(pkl_save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"{dataset_name} pkl数据已保存到 {pkl_save_path}")


def Make_Tallinn_Dataset(dataset_root_dir):
    json_dict = {}
    drop_files = []

    dataset_name = basename(dataset_root_dir)
    dataset_save_dir = Path(dataset_root_dir).parent / f"{dataset_name}_Processed"

    dataset_mesh_dir = join(dataset_root_dir, "meshes")
    dataset_pts_dir = join(dataset_root_dir, "pointclouds")

    mesh_save_dir = Path(join(dataset_save_dir, "meshes"))
    mesh_save_dir.mkdir(parents=True, exist_ok=True)
    pts_save_dir = Path(join(dataset_save_dir, "pointclouds"))
    pts_save_dir.mkdir(parents=True, exist_ok=True)

    face_max_count = 0
    face_vertex_max_count = 0
    face_vertex_sum_max_count = 0
    vertex_max_count = 0
    pts_max_count = 0

    for i, file in tqdm(
        enumerate(os.listdir(dataset_mesh_dir)),
        total=len(os.listdir(dataset_mesh_dir)),
        desc="Make Tallinn Dataset",
    ):
        # if i > 10:
        #     break
        if file.endswith(".obj"):
            filename = splitext(file)[0]
            pts_save_path = join(pts_save_dir, f"{int(filename):05d}.xyz")
            mesh_save_path = join(mesh_save_dir, f"{int(filename):05d}.obj")

            mesh_file = join(dataset_mesh_dir, f"{filename}.obj")
            pts_file = join(dataset_pts_dir, f"{filename}.xyz")

            # 获取Mesh文件的顶点和平面
            vertices, faces = read_polygon_mesh(mesh_file, face_split=True)
            # 获取点云数据
            pts = read_pts_xyz(pts_file)

            # 过滤数据(顶点与平面数量, 平面或顶点小于4不构成三维模型)
            if vertices.shape[0] > 100 or vertices.shape[0] < 4 or len(faces) < 4:
                drop_files.append(filename)
                continue

            # 将顶点中心化、归一化
            centered_vertices, vertices_center = center_vertices(vertices)
            scaled_vertices, vertices_scale = normalize_vertices(centered_vertices)

            # 将点云与顶点对齐中心化、归一化
            scaled_pts = (pts - vertices_center) / vertices_scale

            mesh = trimesh.load_mesh(mesh_file)
            mesh.vertices = (mesh.vertices - vertices_center) / vertices_scale

            # 将Mesh缩放变换(1.1倍),尽可能的包含更多的原始点云数据
            scale_matrix = trimesh.transformations.scale_matrix(1.1)
            mesh = mesh.apply_transform(scale_matrix)
            # 检查哪些点位于Mesh内部
            pts_contain_mask = mesh.contains(scaled_pts)
            # 获取位于Mesh内部的点云数据
            pts_contain = scaled_pts[pts_contain_mask]

            # 过滤采样点云数据
            if pts_contain.shape[0] > 4096:
                idx = np.random.randint(0, pts_contain.shape[0], 4096)
                np.random.shuffle(idx)
                pts_contain = pts_contain[idx]

            # 重新获取原始Mesh数据
            mesh = trimesh.load_mesh(mesh_file)
            mesh.vertices = (mesh.vertices - vertices_center) / vertices_scale

            # 采样原始Mesh表面点云数据
            mesh_surface_samples, _ = trimesh.sample.sample_surface(mesh, 10000)
            # 计算Mesh内部的点云数据与表面点云数据之间的距离
            distances = cdist(pts_contain, mesh_surface_samples, metric="euclidean")
            # 保留仅在Mesh平面一定距离阈值内的点云数据
            pts_on_surface = pts_contain[np.any(distances < 0.015, axis=1)]

            # 检查点云数量
            if pts_on_surface.shape[0] < 24:
                drop_files.append(filename)
                continue

            if len(faces) > face_max_count:
                face_max_count = len(faces)
            if max([len(f) for f in faces]) > face_vertex_max_count:
                face_vertex_max_count = max([len(f) for f in faces])
            if sum([len(f) for f in faces]) > face_vertex_sum_max_count:
                face_vertex_sum_max_count = sum([len(f) for f in faces])
            if scaled_vertices.shape[0] > vertex_max_count:
                vertex_max_count = scaled_vertices.shape[0]
            if pts_on_surface.shape[0] > pts_max_count:
                pts_max_count = pts_on_surface.shape[0]

            write_obj(scaled_vertices, faces, mesh_save_path)
            write_pts(pts_on_surface, pts_save_path)

            json_dict[f"{int(filename):05d}"] = {
                "count_verts": scaled_vertices.shape[0],
                "count_faces": len(faces),
                "center": vertices_center.tolist(),
                "scale": float(vertices_scale),
                "count_pts": pts_on_surface.shape[0],
            }

    print(f"Dataset save path = {dataset_save_dir}")
    json_save_path = join(dataset_save_dir, "info.json")
    json_data = json.dumps(json_dict, indent=4)
    with open(json_save_path, "w") as json_file:
        json_file.write(json_data)

    # 按区域分割数据集
    train_count, train_radio, val_count, val_radio = make_dataset_split_region(
        dataset_save_dir, need_val=True, val_grid_index=8
    )
    # 数据集分布作图
    make_dataset_figure(dataset_save_dir)
    # 可视化数据集
    visualize_city(dataset_save_dir)

    print(f"drop_files = {len(drop_files)}")
    dataset_info_path = join(dataset_save_dir, "dataset_info.txt")
    original_file_count = len(json_dict) + len(drop_files)
    handle_time = date.today().strftime("%Y-%m-%d")
    with open(dataset_info_path, "w") as dft:
        dft.write(f"{dataset_name} Dataset Extra Info\n")
        dft.write(f"handle time : {handle_time}\n")
        dft.write(f"handle author : ShuaibZyx(email:2631667689@qq.com)\n")
        dft.write(f"original_file_count = {original_file_count}\n")
        dft.write(f"file_count = {len(json_dict)}\n")
        dft.write(f"drop_count = {len(drop_files)}\n")
        dft.write(f"train_count = {train_count}\n")
        dft.write(f"train_radio = {train_radio}\n")
        dft.write(f"val_count = {val_count}\n")
        dft.write(f"val_radio = {val_radio}\n")
        dft.write(f"face_max_count = {face_max_count}\n")
        dft.write(f"face_vertex_max_count = {face_vertex_max_count}\n")
        dft.write(f"face_vertex_sum_max_count = {face_vertex_sum_max_count}\n")
        dft.write(f"vertex_max_count = {vertex_max_count}\n")
        dft.write(f"pts_max_count = {pts_max_count}\n")

    drop_file_path = join(dataset_save_dir, "drop_files.txt")
    with open(drop_file_path, "w") as dft:
        for file in drop_files:
            dft.write(file + "\n")

    # create dataset pkl
    preprocess_pkl(
        dataset_root=dataset_save_dir,
        type="Polygon",
        bottom=True,
        fps=False,
        npoints=2048,
    )
    preprocess_pkl(
        dataset_root=dataset_save_dir,
        type="Triangle",
        bottom=True,
        fps=False,
        npoints=2048,
    )
    print("done")


def rename_zurich_dataset(dataset_dir):
    dataset_mesh_path = join(dataset_dir, "meshes")
    dataset_pts_path = join(dataset_dir, "pointclouds")
    dataset_json_path = join(dataset_dir, "info.json")
    with open(dataset_json_path, "r") as dj:
        dataset_json = json.load(dj)
        for i, file in tqdm(
            enumerate(os.listdir(dataset_mesh_path)),
            total=len(os.listdir(dataset_mesh_path)),
            desc="Rename Zurich Dataset",
        ):
            filename = splitext(file)[0]
            mesh_file = join(dataset_mesh_path, f"{filename}.obj")
            pts_file = join(dataset_pts_path, f"{filename}.xyz")
            mesh_file_new = join(dataset_mesh_path, f"{i:05d}.obj")
            pts_file_new = join(dataset_pts_path, f"{i:05d}.xyz")
            os.rename(mesh_file, mesh_file_new)
            os.rename(pts_file, pts_file_new)
            dataset_json[f"{i:05d}"] = dataset_json.pop(filename)
    with open(dataset_json_path, "w") as dj:
        json.dump(dataset_json, dj, indent=4)
    print("done")


def Make_Zurich_Dataset(dataset_root_dir: str):
    json_dict = {}
    drop_files = []

    dataset_name = basename(dataset_root_dir)
    dataset_save_dir = Path(dataset_root_dir).parent / f"{dataset_name}_Filtered"
    dataset_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"dataset_save_dir = {dataset_save_dir}")

    dataset_pts_save_dir = dataset_save_dir / "pointclouds"
    dataset_pts_save_dir.mkdir(parents=True, exist_ok=True)
    dataset_meshes_save_dir = dataset_save_dir / "meshes"
    dataset_meshes_save_dir.mkdir(parents=True, exist_ok=True)

    dataset_train_path = join(dataset_root_dir, "trainset", "meshes")
    dataset_test_path = join(dataset_root_dir, "testset", "meshes")

    train_json_path = join(dataset_root_dir, "trainset", "info.json")
    test_json_path = join(dataset_root_dir, "testset", "info.json")

    face_max_count = 0
    face_vertex_max_count = 0
    face_vertex_sum_max_count = 0
    vertex_max_count = 0
    pts_max_count = 0

    with open(train_json_path) as trjp:
        train_json = json.load(trjp)
        for i, file in tqdm(
            enumerate(os.listdir(dataset_train_path)),
            total=len(os.listdir(dataset_train_path)),
            desc="Make Zurich Dataset-Train",
        ):
            # if i > 10:
            #     break
            if file.endswith(".obj"):
                # 获取文件信息
                filename = splitext(file)[0]
                pts_save_path = join(dataset_pts_save_dir, f"{filename}.xyz")
                mesh_save_path = join(dataset_meshes_save_dir, f"{filename}.obj")

                mesh_file_path = join(dataset_train_path, file)
                pts_file_path = mesh_file_path.replace("meshes", "pointclouds").replace(
                    "obj", "xyz"
                )

                # 加载数据
                vertices, faces = read_polygon_mesh(mesh_file_path)
                pts = read_pts_xyz(pts_file_path)
                json_info = train_json[filename]

                # 过滤数据(顶点与平面数量, 平面或顶点小于4不构成三维模型)
                if vertices.shape[0] > 100 or vertices.shape[0] < 4 or len(faces) < 4:
                    drop_files.append(filename)
                    continue

                # 恢复原始数据
                vertices = vertices * json_info["scale"] + json_info["center"]
                pts = pts * json_info["scale"] + json_info["center"]

                # 中心化、归一化数据
                centered_vertices, vertices_center = center_vertices(vertices)
                scaled_vertices, vertices_scale = normalize_vertices(centered_vertices)
                scaled_pts = (pts - vertices_center) / vertices_scale

                # 加载Mesh对象
                mesh = trimesh.load_mesh(mesh_file_path)
                mesh.vertices = mesh.vertices * json_info["scale"] + json_info["center"]
                mesh.vertices = (mesh.vertices - vertices_center) / vertices_scale

                # 将Mesh缩放变换(1.1倍),尽可能的包含更多的原始点云数据
                scale_matrix = trimesh.transformations.scale_matrix(1.1)
                mesh = mesh.apply_transform(scale_matrix)

                # 检查哪些点位于Mesh内部
                pts_contain_mask = mesh.contains(scaled_pts)
                # 获取位于Mesh内部的点云数据
                pts_contain = scaled_pts[pts_contain_mask]

                # 过滤采样点云数据
                if pts_contain.shape[0] > 4096:
                    idx = np.random.randint(0, pts_contain.shape[0], 4096)
                    np.random.shuffle(idx)
                    pts_contain = pts_contain[idx]

                # 重新获取原始Mesh数据
                mesh = trimesh.load_mesh(mesh_file_path)
                mesh.vertices = mesh.vertices * json_info["scale"] + json_info["center"]
                mesh.vertices = (mesh.vertices - vertices_center) / vertices_scale

                # 采样原始Mesh表面点云数据
                mesh_surface_samples, _ = trimesh.sample.sample_surface(mesh, 10000)
                # 计算Mesh内部的点云数据与表面点云数据之间的距离
                distances = cdist(pts_contain, mesh_surface_samples, metric="euclidean")
                # 保留仅在Mesh平面一定距离阈值内的点云数据
                pts_on_surface = pts_contain[np.any(distances < 0.015, axis=1)]

                # 检查点云数量(避免过滤结果太小或为空)
                if pts_on_surface.shape[0] < 24:
                    drop_files.append(filename)
                    continue

                if len(faces) > face_max_count:
                    face_max_count = len(faces)
                if max([len(f) for f in faces]) > face_vertex_max_count:
                    face_vertex_max_count = max([len(f) for f in faces])
                if sum([len(f) for f in faces]) > face_vertex_sum_max_count:
                    face_vertex_sum_max_count = sum([len(f) for f in faces])
                if scaled_vertices.shape[0] > vertex_max_count:
                    vertex_max_count = scaled_vertices.shape[0]
                if pts_on_surface.shape[0] > pts_max_count:
                    pts_max_count = pts_on_surface.shape[0]

                write_obj(scaled_vertices, faces, mesh_save_path)
                write_pts(pts_on_surface, pts_save_path)

                json_dict[filename] = {
                    "count_verts": scaled_vertices.shape[0],
                    "count_faces": len(faces),
                    "center": vertices_center.tolist(),
                    "scale": float(vertices_scale),
                    "count_pts": pts_on_surface.shape[0],
                }

    with open(test_json_path) as tejp:
        test_json = json.load(tejp)
        for i, file in tqdm(
            enumerate(os.listdir(dataset_test_path)),
            total=len(os.listdir(dataset_test_path)),
            desc="Make Zurich Dataset-Test",
        ):
            # if i > 10:
            #     break
            if file.endswith(".obj"):
                # 获取文件信息
                filename = splitext(file)[0]
                pts_save_path = join(dataset_pts_save_dir, f"{filename}.xyz")
                mesh_save_path = join(dataset_meshes_save_dir, f"{filename}.obj")
                mesh_file_path = join(dataset_test_path, file)
                pts_file_path = mesh_file_path.replace("meshes", "pointclouds").replace(
                    "obj", "xyz"
                )

                # 加载数据
                vertices, faces = read_polygon_mesh(mesh_file_path)
                pts = read_pts_xyz(pts_file_path)
                json_info = test_json[filename]

                # 过滤数据(顶点与平面数量, 平面或顶点小于4不构成三维模型)
                if vertices.shape[0] < 4 or vertices.shape[0] > 100 or len(faces) < 4:
                    drop_files.append(filename)
                    continue

                # 恢复原始数据
                vertices = vertices * json_info["scale"] + json_info["center"]
                pts = pts * json_info["scale"] + json_info["center"]

                # 中心化、归一化数据
                centered_vertices, vertices_center = center_vertices(vertices)
                scaled_vertices, vertices_scale = normalize_vertices(centered_vertices)
                scaled_pts = (pts - vertices_center) / vertices_scale

                # 加载Mesh对象
                mesh = trimesh.load_mesh(mesh_file_path)
                mesh.vertices = mesh.vertices * json_info["scale"] + json_info["center"]
                mesh.vertices = (mesh.vertices - vertices_center) / vertices_scale

                # 将Mesh缩放变换(1.1倍),尽可能的包含更多的原始点云数据
                scale_matrix = trimesh.transformations.scale_matrix(1.1)
                mesh = mesh.apply_transform(scale_matrix)

                # 检查哪些点位于Mesh内部
                pts_contain_mask = mesh.contains(scaled_pts)
                # 获取位于Mesh内部的点云数据
                pts_contain = scaled_pts[pts_contain_mask]

                # 过滤采样点云数据
                if pts_contain.shape[0] > 4096:
                    idx = np.random.randint(0, pts_contain.shape[0], 4096)
                    np.random.shuffle(idx)
                    pts_contain = pts_contain[idx]

                # 重新获取原始Mesh数据
                mesh = trimesh.load_mesh(mesh_file_path)
                mesh.vertices = mesh.vertices * json_info["scale"] + json_info["center"]
                mesh.vertices = (mesh.vertices - vertices_center) / vertices_scale

                # 采样原始Mesh表面点云数据
                mesh_surface_samples, _ = trimesh.sample.sample_surface(mesh, 10000)
                # 计算Mesh内部的点云数据与表面点云数据之间的距离
                distances = cdist(pts_contain, mesh_surface_samples, metric="euclidean")
                # 保留仅在Mesh平面一定距离阈值内的点云数据
                pts_on_surface = pts_contain[np.any(distances < 0.015, axis=1)]

                # 检查点云数量(避免过滤结果太小或为空)
                if pts_on_surface.shape[0] < 24:
                    drop_files.append(filename)
                    continue

                if len(faces) > face_max_count:
                    face_max_count = len(faces)
                if max([len(f) for f in faces]) > face_vertex_max_count:
                    face_vertex_max_count = max([len(f) for f in faces])
                if sum([len(f) for f in faces]) > face_vertex_sum_max_count:
                    face_vertex_sum_max_count = sum([len(f) for f in faces])
                if scaled_vertices.shape[0] > vertex_max_count:
                    vertex_max_count = scaled_vertices.shape[0]
                if pts_on_surface.shape[0] > pts_max_count:
                    pts_max_count = pts_on_surface.shape[0]

                write_obj(scaled_vertices, faces, mesh_save_path)
                write_pts(pts_on_surface, pts_save_path)

                json_dict[filename] = {
                    "count_verts": scaled_vertices.shape[0],
                    "count_faces": len(faces),
                    "center": vertices_center.tolist(),
                    "scale": float(vertices_scale),
                    "count_pts": pts_on_surface.shape[0],
                }

    print(f"Dataset save path = {dataset_save_dir}")
    json_save_path = join(dataset_save_dir, "info.json")
    json_data = json.dumps(json_dict, indent=4)
    with open(json_save_path, "w") as json_file:
        json_file.write(json_data)

    # 重命名数据集
    rename_zurich_dataset(dataset_save_dir)
    # 按区域分割数据集
    train_count, train_radio, val_count, val_radio = make_dataset_split_region(
        dataset_save_dir, need_val=True, val_grid_index=8
    )
    # 数据集分布作图
    make_dataset_figure(dataset_save_dir)
    # 可视化数据集
    visualize_city(dataset_save_dir)

    print(f"drop_files = {len(drop_files)}")
    dataset_info_path = join(dataset_save_dir, "dataset_info.txt")
    original_file_count = len(json_dict) + len(drop_files)
    handle_time = date.today().strftime("%Y-%m-%d")
    with open(dataset_info_path, "w") as dft:
        dft.write(f"{dataset_name} Dataset Extra Info\n")
        dft.write(f"handle time : {handle_time}\n")
        dft.write(f"handle author : ShuaibZyx(email:2631667689@qq.com)\n")
        dft.write(f"original_file_count = {original_file_count}\n")
        dft.write(f"file_count = {len(json_dict)}\n")
        dft.write(f"drop_count = {len(drop_files)}\n")
        dft.write(f"train_count = {train_count}\n")
        dft.write(f"train_radio = {train_radio}\n")
        dft.write(f"val_count = {val_count}\n")
        dft.write(f"val_radio = {val_radio}\n")
        dft.write(f"face_max_count = {face_max_count}\n")
        dft.write(f"face_vertex_max_count = {face_vertex_max_count}\n")
        dft.write(f"face_vertex_sum_max_count = {face_vertex_sum_max_count}\n")
        dft.write(f"vertex_max_count = {vertex_max_count}\n")
        dft.write(f"pts_max_count = {pts_max_count}\n")

    drop_file_path = join(dataset_save_dir, "drop_files.txt")
    with open(drop_file_path, "w") as dft:
        for file in drop_files:
            dft.write(file + "\n")

    # create dataset pkl
    preprocess_pkl(
        dataset_root=dataset_save_dir,
        type="Polygon",
        bottom=True,
        fps=False,
        npoints=2048,
    )
    preprocess_pkl(
        dataset_root=dataset_save_dir,
        type="Triangle",
        bottom=True,
        fps=False,
        npoints=2048,
    )
    print("done")


def Make_City3D_Dataset(dataset_root_dir):
    json_dict = {}
    drop_files = []

    dataset_name = basename(dataset_root_dir)
    dataset_save_dir = Path(dataset_root_dir).parent / f"{dataset_name}_Filtered"
    dataset_save_dir.mkdir(parents=True, exist_ok=True)

    mesh_save_dir = Path(join(dataset_save_dir, "meshes"))
    mesh_save_dir.mkdir(parents=True, exist_ok=True)
    pts_save_dir = Path(join(dataset_save_dir, "pointclouds"))
    pts_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset_save_dir = {dataset_save_dir}")

    face_max_count = 0
    face_vertex_max_count = 0
    face_vertex_sum_max_count = 0
    vertex_max_count = 0
    pts_max_count = 0

    for i, file in tqdm(
        enumerate(os.listdir(dataset_root_dir)),
        total=len(os.listdir(dataset_root_dir)),
        desc="Make City3D Dataset",
    ):
        # if i > 10:
        #     break
        if file.endswith(".obj"):
            filename = splitext(file)[0]
            mesh_file = join(dataset_root_dir, f"{filename}.obj")
            pts_file = join(dataset_root_dir, f"{filename}.ply")

            mesh_save_path = join(mesh_save_dir, f"{filename}.obj")
            pts_save_path = join(pts_save_dir, f"{filename}.xyz")

            vertices, faces = read_polygon_mesh(mesh_file, False)
            pts = read_pts_ply(pts_file)

            # 将顶点中心化、归一化
            centered_vertices, vertices_center = center_vertices(vertices)
            scaled_vertices, vertices_scale = normalize_vertices(centered_vertices)

            # 将点云与顶点对齐中心化、归一化
            scaled_pts = (pts - vertices_center) / vertices_scale

            mesh = trimesh.load_mesh(mesh_file)
            mesh.vertices = (mesh.vertices - vertices_center) / vertices_scale

            # 将Mesh缩放变换(1.1倍),尽可能的包含更多的原始点云数据
            scale_matrix = trimesh.transformations.scale_matrix(1.1)
            mesh = mesh.apply_transform(scale_matrix)
            # 检查哪些点位于Mesh内部
            pts_contain_mask = mesh.contains(scaled_pts)
            # 获取位于Mesh内部的点云数据
            pts_contain = scaled_pts[pts_contain_mask]

            # 过滤采样点云数据
            if pts_contain.shape[0] > 4096:
                idx = np.random.randint(0, pts_contain.shape[0], 4096)
                np.random.shuffle(idx)
                pts_contain = pts_contain[idx]

            # 重新获取原始Mesh数据
            mesh = trimesh.load_mesh(mesh_file)
            mesh.vertices = (mesh.vertices - vertices_center) / vertices_scale

            # 采样原始Mesh表面点云数据
            mesh_surface_samples, _ = trimesh.sample.sample_surface(mesh, 10000)
            # 计算Mesh内部的点云数据与表面点云数据之间的距离
            distances = cdist(pts_contain, mesh_surface_samples, metric="euclidean")
            # 保留仅在Mesh平面一定距离阈值内的点云数据
            pts_on_surface = pts_contain[np.any(distances < 0.015, axis=1)]

            # 检查点云数量
            if pts_on_surface.shape[0] < 24:
                drop_files.append(filename)
                continue

            if len(faces) > face_max_count:
                face_max_count = len(faces)
            if max([len(f) for f in faces]) > face_vertex_max_count:
                face_vertex_max_count = max([len(f) for f in faces])
            if sum([len(f) for f in faces]) > face_vertex_sum_max_count:
                face_vertex_sum_max_count = sum([len(f) for f in faces])
            if scaled_vertices.shape[0] > vertex_max_count:
                vertex_max_count = scaled_vertices.shape[0]
            if pts_on_surface.shape[0] > pts_max_count:
                pts_max_count = pts_on_surface.shape[0]

            write_obj(scaled_vertices, faces, mesh_save_path)
            write_pts(pts_on_surface, pts_save_path)

            json_dict[filename] = {
                "count_verts": scaled_vertices.shape[0],
                "count_faces": len(faces),
                "center": vertices_center.tolist(),
                "scale": float(vertices_scale),
                "count_pts": pts_on_surface.shape[0],
            }

    print(f"Dataset save path = {dataset_save_dir}")
    json_save_path = join(dataset_save_dir, "info.json")
    json_data = json.dumps(json_dict, indent=4)
    with open(json_save_path, "w") as json_file:
        json_file.write(json_data)

    # 按区域分割数据集
    train_count, train_radio, val_count, val_radio = make_dataset_split_region(
        dataset_save_dir, need_val=True, val_grid_index=8
    )
    # 数据集分布作图
    make_dataset_figure(dataset_save_dir)
    # 可视化数据集
    visualize_city(dataset_save_dir)
    visualize_city_split(dataset_save_dir, split="val")

    print(f"drop_files = {len(drop_files)}")
    dataset_info_path = join(dataset_save_dir, "dataset_info.txt")
    original_file_count = len(json_dict) + len(drop_files)
    handle_time = date.today().strftime("%Y-%m-%d")
    with open(dataset_info_path, "w") as dft:
        dft.write(f"{dataset_name} Dataset Extra Info\n")
        dft.write(f"handle time : {handle_time}\n")
        dft.write(f"handle author : ShuaibZyx(email:2631667689@qq.com)\n")
        dft.write(f"original_file_count = {original_file_count}\n")
        dft.write(f"file_count = {len(json_dict)}\n")
        dft.write(f"drop_count = {len(drop_files)}\n")
        dft.write(f"train_count = {train_count}\n")
        dft.write(f"train_radio = {train_radio}\n")
        dft.write(f"val_count = {val_count}\n")
        dft.write(f"val_radio = {val_radio}\n")
        dft.write(f"face_max_count = {face_max_count}\n")
        dft.write(f"face_vertex_max_count = {face_vertex_max_count}\n")
        dft.write(f"face_vertex_sum_max_count = {face_vertex_sum_max_count}\n")
        dft.write(f"vertex_max_count = {vertex_max_count}\n")
        dft.write(f"pts_max_count = {pts_max_count}\n")

    drop_file_path = join(dataset_save_dir, "drop_files.txt")
    with open(drop_file_path, "w") as dft:
        for file in drop_files:
            dft.write(file + "\n")

    # create dataset pkl
    preprocess_pkl(
        dataset_root=dataset_save_dir,
        type="Polygon",
        bottom=True,
        fps=False,
        npoints=2048,
    )
    preprocess_pkl(
        dataset_root=dataset_save_dir,
        type="Triangle",
        bottom=True,
        fps=False,
        npoints=2048,
    )

    make_val_dual(dataset_save_dir, "val")
    make_val_city3d(dataset_save_dir, "val")
    print("done")


def write_pts_dual(pts, ground, file_path):
    with open(file_path, "w") as f:
        f.write(f"# ground {ground}\n")
        [f.write(f"{v1} {v2} {v3}\n") for v1, v2, v3 in pts[:, :3]]


def make_val_city3d(dataset_root_dir, split="val"):
    dataset_name = basename(dataset_root_dir)
    val_save_path = join(dataset_root_dir, "city3d_val")
    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)
    with open(join(dataset_root_dir, "info.json")) as djp:
        dataset_json = json.load(djp)
    with open(join(dataset_root_dir, f"{split}.txt")) as vt:
        val_list = [line.strip() for line in vt.readlines()]
    for file in tqdm(val_list, desc=f"Make {dataset_name} Val City3D"):
        xyz_file_path = join(dataset_root_dir, "pointclouds", f"{file}.xyz")
        ply_save_path = join(val_save_path, f"{file}.ply")
        file_json = dataset_json[file]
        pts = read_pts_xyz(xyz_file_path)
        pts = add_pts_bottom(pts, split="val", fps=True, fps_radio=100)
        pts = pts * file_json["scale"]
        pts_cuple = [(pts[i, 0], pts[i, 1], pts[i, 2]) for i in range(pts.shape[0])]
        vertex = np.array(pts_cuple, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        el = PlyElement.describe(vertex, "vertex", comments=["vertices"])
        PlyData([el], text=True).write(ply_save_path)
    print("done")


def make_val_dual(dataset_root_dir, split="val"):
    dataset_name = basename(dataset_root_dir)
    val_save_path = join(dataset_root_dir, "dual_val")
    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)
    with open(join(dataset_root_dir, f"{split}.txt")) as vt:
        val_list = [line.strip() for line in vt.readlines()]
    with open(join(dataset_root_dir, "info.json")) as dj:
        dataset_json = json.load(dj)
    for file in tqdm(val_list, desc=f"Make {dataset_name} Val 2.5D Dual"):
        pts_file = join(dataset_root_dir, "pointclouds", f"{file}.xyz")
        file_json = dataset_json[file]
        pts = read_pts_xyz(pts_file)
        pts = add_pts_bottom(pts, split="val", fps=True, fps_radio=100)
        pts = pts * file_json["scale"]
        ground = np.min(pts[:, 2])
        pts_coord_save_path = join(val_save_path, f"{file}.xyzn")
        write_pts_dual(pts, ground, pts_coord_save_path)
    print("done")


def point_density(points, vertices, radius=0.05):
    """计算顶点radius周围平均点云密度"""
    tree = KDTree(points)  # 构建 KDTree 用于最近邻搜索
    densities = []
    for v in vertices:
        indices = tree.query_ball_point(v, radius)
        densities.append(len(indices))
    return np.mean(np.array(densities))


def dataset_point_density(dataset_root_dir):
    dataset_name = basename(dataset_root_dir)
    dataset_mesh_dir = join(dataset_root_dir, "meshes")
    dataset_pts_dir = join(dataset_root_dir, "pointclouds")
    dataset_densities = []
    for i, file in tqdm(
        enumerate(os.listdir(dataset_mesh_dir)),
        total=len(os.listdir(dataset_mesh_dir)),
        desc="Dataset Point Density",
    ):
        # if i > 100:
        #     break
        if file.endswith(".obj"):
            filename = splitext(file)[0]
            mesh_file = join(dataset_mesh_dir, f"{filename}.obj")
            pts_file = join(dataset_pts_dir, f"{filename}.xyz")
            vertices, _ = read_polygon_mesh(mesh_file)
            pts = read_pts_xyz(pts_file)
            pts = add_pts_bottom(pts, vertices, split="train", fps=False)
            density = point_density(pts, vertices, 0.05)
            dataset_densities.append(density)
    mean_dataset_density = np.mean(np.array(dataset_densities))
    print(f"{dataset_name} dataset point density mean = {mean_dataset_density}")


if __name__ == "__main__":
    preprocess_pkl(
        dataset_root="",
        mesh_type="Polygon",
        add_bottom=True,
        npoints=4096,
        test_split="test",
    )

