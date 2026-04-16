import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from pathlib import Path
from PIL import Image, ImageDraw
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import trimesh
from os.path import join, dirname, basename, splitext, exists
import os
import math
from src.utils.pointnet_stack_utils import ball_query
import torch
matplotlib.use("Agg")


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
    edges = set()
    with open(poly_file, "r") as f:
        for line in f:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                face = [int(v.split("/")[0]) - 1 for v in line.split()[1:]]
                fs.append(face)
                # 提取去重边
                for i in range(len(face)):
                    a, b = face[i], face[(i + 1) % len(face)]
                    edges.add(tuple(sorted((a, b))))  # 排序避免重复
    return np.array(vs), fs, list(edges)


def plot_points_axes3D(pts, output_path, multiple, color, transpose: bool = False):
    pts = pts[:, [1, 2, 0]] if transpose else pts
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    lim = 0.3 * multiple

    plt.autoscale(False)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=60, alpha=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.tick_params(axis="y", which="both", labelleft=False)
    ax.tick_params(axis="z", which="both", labelleft=False)

    ax.grid(color="black")  # Set the grid line color to black

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")


def visualize_points(points, vis_path, colors=None):
    if colors is None:
        Path(vis_path).write_text(
            "\n".join(f"v {p[0]} {p[1]} {p[2]} 127 127 127" for p in points)
        )
    else:
        Path(vis_path).write_text(
            "\n".join(
                f"v {p[0]} {p[1]} {p[2]} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
                for i, p in enumerate(points)
            )
        )


def plot_pts(
    pts_path, output_dir, color="#158bb8", view=[25, 180, 0], alpha=1, widths=3
):
    pts = load_xyz(pts_path)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=widths, alpha=alpha)

    # 自动调整坐标轴范围
    ax.auto_scale_xyz(pts[:, 0], pts[:, 1], pts[:, 2])

    # 设置视角
    ax.view_init(view[0], view[1], view[2])

    # 关闭坐标轴
    ax.set_axis_off()

    filename = splitext(basename(pts_path))[0]
    output_path = join(output_dir, f"{filename}.png")

    if not exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=450, transparent=True)
    plt.close(fig)


def plot_pts_for_pvfa(pts_path, output_dir, color="#158bb8"):
    pts = load_xyz(pts_path)
    vs, fs, _ = load_obj(
        pts_path.replace("pointclouds", "meshes").replace("xyz", "obj")
    )
    # minz_p = -np.max(vs[:, 2])
    # added_pts = pts.copy()
    # added_pts[:, 2] = minz_p
    # pts = np.vstack((pts, added_pts))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    fig.set_facecolor("black")
    ax.set_facecolor("black")

    xyz = torch.from_numpy(pts).to(dtype=torch.float32, device="cuda").contiguous()
    xyz_batch_cnt = torch.Tensor([pts.shape[0]]).to(dtype=torch.int32, device="cuda")
    new_xyz = torch.from_numpy(vs).to(dtype=torch.float32, device="cuda").contiguous()
    new_xyz_batch_cnt = torch.Tensor([vs.shape[0]]).to(dtype=torch.int32, device="cuda")

    radius = 0.05
    nsample = 64

    idx, empty_ball_mask = ball_query(
        radius, nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt
    )
    vs_has_neighbor = new_xyz[~empty_ball_mask]
    vs_no_neighbor = new_xyz[empty_ball_mask]

    pts_neighbor_mask = torch.zeros(xyz.shape[0], dtype=torch.bool)
    unique_indices = torch.unique(idx.view(-1))
    pts_neighbor_mask[unique_indices] = True

    pts_is_neighbor = pts[pts_neighbor_mask]
    pts_not_neighbor = pts[~pts_neighbor_mask]

    vs_has_neighbor = vs_has_neighbor.cpu().numpy()
    vs_no_neighbor = vs_no_neighbor.cpu().numpy()

    ax.scatter(
        vs_has_neighbor[:, 0],
        vs_has_neighbor[:, 1],
        vs_has_neighbor[:, 2],
        c="red",
        s=3,
        alpha=1,
    )
    ax.scatter(
        vs_no_neighbor[:, 0],
        vs_no_neighbor[:, 1],
        vs_no_neighbor[:, 2],
        c="red",
        s=3,
        alpha=1,
    )
    ax.scatter(
        pts_is_neighbor[:, 0],
        pts_is_neighbor[:, 1],
        pts_is_neighbor[:, 2],
        c="#FDF9C0",
        s=3,
        alpha=1,
    )
    ax.scatter(
        pts_not_neighbor[:, 0],
        pts_not_neighbor[:, 1],
        pts_not_neighbor[:, 2],
        c="#4BA6C8",
        s=3,
        alpha=1,
    )

    # 自动调整坐标轴范围
    ax.auto_scale_xyz(pts[:, 0], pts[:, 1], pts[:, 2])

    # 设置视角
    ax.view_init(25, 180, 0)

    # 关闭坐标轴
    ax.set_axis_off()

    filename = splitext(basename(pts_path))[0]
    output_path = join(output_dir, f"{filename}.png")
    print(f"Saving to {output_path}")

    if not exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=450)
    plt.close(fig)


def plot_vertices(mesh_path, color="#158bb8"):
    vertices, _, _ = load_obj(mesh_path)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=color, s=10, alpha=1)

    # 自动调整坐标轴范围
    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    # 设置视角
    ax.view_init(25, 180, 0)

    # 关闭坐标轴
    ax.set_axis_off()

    output_dir = dirname(mesh_path)
    filename = splitext(basename(mesh_path))[0]
    output_path = join(dirname(mesh_path), f"{filename}.png")

    if not exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=450, transparent=True)
    plt.close(fig)


def plot_vertices_and_faces(
    vertices,
    faces,
    multiple,
    vertices_color,
    face_color,
    output_path,
    transpose: bool = False,
):
    vertices = vertices[:, [1, 2, 0]] if transpose else vertices
    ngons = [[vertices[v, :].tolist() for v in f] for f in faces]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    lim = 0.25 * multiple
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=vertices_color, s=20)
    polygon_collection = Poly3DCollection(ngons)
    polygon_collection.set_alpha(0.3)
    polygon_collection.set_color(face_color)
    ax.add_collection(polygon_collection)
    ax.set_zlim(-lim, lim)
    ax.view_init(25, 180, 0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")


def plot_mesh(
    mesh_path,
    output_dir,
    color=["yellow", "white", "skyblue"],  # 点线面
    view=[25, 180, 0],  # 上下、左右、相机自身旋转
    alpha=[1, 0.2, 0.8],  # 点线面
    widths=[15, 1, 0.5],  # 点线面
):
    # 加载顶点和边
    vertices, _, edges = load_obj(mesh_path)

    # 边
    lines = [[vertices[v1, :].tolist(), vertices[v2, :].tolist()] for v1, v2 in edges]

    mesh = trimesh.load(mesh_path)
    vertices, faces = mesh.vertices, mesh.faces

    # 面
    ngons = np.array([vertices[face] for face in faces])

    # 灯光
    ls = LightSource(azdeg=225, altdeg=60)

    # 创建图形
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 面：0.8 线0.2

    # 绘制面
    polygon_collection = Poly3DCollection(
        ngons,
        alpha=alpha[2],
        linewidths=widths[2],
        facecolors=color[2],
        edgecolors=color[1],
        shade=True,
        lightsource=ls,
    )
    ax.add_collection3d(polygon_collection)

    # # 绘制线
    # line_collection = Line3DCollection(lines, colors=color[1], linewidths=widths[1], alpha=alpha[1])
    # ax.add_collection3d(line_collection)

    # 绘制点
    # ax.scatter(
    #     vertices[:, 0],
    #     vertices[:, 1],
    #     vertices[:, 2],
    #     c=color[0],
    #     s=widths[0],
    #     edgecolors="red",
    #     alpha=alpha[0]
    # )

    # 自动调整坐标轴范围
    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    # 设置视角
    # 上下、左右、相机自身旋转
    ax.view_init(view[0], view[1], view[2])

    # 关闭坐标轴
    ax.set_axis_off()

    filename = splitext(basename(mesh_path))[0]
    output_path = join(output_dir, f"{filename}.png")

    if not exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=450, transparent=True)
    plt.close(fig)  # 关闭当前图形


def visualize_mesh_vertices_gif(vertices, output_dir, transpose: bool = False):
    vertices = vertices[:, [1, 2, 0]] if transpose else vertices
    for i in range(1, len(vertices), 1):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        plt.xlim(-0.35, 0.35)
        plt.ylim(-0.35, 0.35)
        # Don't mess with the limits!
        plt.autoscale(False)
        ax.set_axis_off()
        ax.scatter(vertices[:i, 0], vertices[:i, 1], vertices[:i, 2], c="g", s=10)
        ax.set_zlim(-0.35, 0.35)
        ax.view_init(25, -120, 0)
        plt.tight_layout()
        plt.savefig(output_dir / f"{i:05d}.png")
        plt.close("all")
    create_gif(output_dir, 40, output_dir / "vis.gif")


def visualize_mesh_vertices_and_faces_gif(vertices, faces, output_dir):
    ngons = [[vertices[v, :].tolist() for v in f] for f in faces]
    for i in range(1, len(ngons) + 1, 1):
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")
        plt.xlim(-0.35, 0.35)
        plt.ylim(-0.35, 0.35)
        # Don't mess with the limits!
        plt.autoscale(False)
        ax.set_axis_off()
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c="black", s=10)
        polygon_collection = Poly3DCollection(ngons[:i])
        polygon_collection.set_alpha(0.3)
        polygon_collection.set_color("b")
        ax.add_collection(polygon_collection)
        ax.set_zlim(-0.35, 0.35)
        ax.view_init(25, -120, 0)
        plt.tight_layout()
        plt.savefig(output_dir / f"{len(vertices) + i:05d}.png")
        plt.close("all")
    create_gif(output_dir, 40, output_dir / "vis.gif")


def create_gif(folder, fps, output_path):
    collection_rgb = []
    for f in sorted(
        [x for x in folder.iterdir() if x.suffix == ".png" or x.suffix == ".jpg"]
    ):
        img_rgb = np.array(Image.open(f).resize((384, 384)))
        collection_rgb.append(img_rgb)
    clip = ImageSequenceClip(collection_rgb, fps=fps)
    clip.write_gif(output_path, verbose=False, logger=None)


def crop_center(image, crop_ratio=0.8):
    """裁剪图片中心部分。"""
    img_width, img_height = image.size
    crop_size = (img_width * crop_ratio, img_height * crop_ratio)
    left = (img_width - crop_size[0]) / 2
    upper = (img_height - crop_size[1]) / 2
    right = left + crop_size[0]
    lower = upper + crop_size[1]
    return image.crop((left, upper, right, lower))


def create_placeholder_image(size, margin_ratio = 0.2):
    """
    生成一个带对角线的透明占位图，表示图像加载失败。
    """
    img = Image.new("RGBA", size, (255, 255, 255, 0))  # 透明背景
    draw = ImageDraw.Draw(img)
    x_margin = int(size[0] * margin_ratio)
    y_margin = int(size[1] * margin_ratio)

    draw.line(
        (x_margin, y_margin, size[0] - x_margin, size[1] - y_margin),
        fill=(75, 166, 200, 255),
        width=10,
    )
    return img


def create_collage(
    image_paths,
    output_path,
    images_per_row=5,
    margin_x=20,
    margin_y=20,
    crop_ratio=0.65,
    dpi=(450, 450),
    vertical=False,
):
    if not image_paths:
        print("图片列表为空。")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 获取图像尺寸（尝试第一个有效图像）
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img_width, img_height = img.size
                break
        except:
            continue
    else:
        print("无有效图片，拼图未生成。")
        return

    crop_size = (math.ceil(img_width * crop_ratio), math.ceil(img_height * crop_ratio))

    num_images = len(image_paths)
    if vertical:
        num_cols = (num_images + images_per_row - 1) // images_per_row
        num_rows = images_per_row
    else:
        num_rows = (num_images + images_per_row - 1) // images_per_row
        num_cols = images_per_row

    total_width = num_cols * crop_size[0] + (num_cols + 1) * margin_x
    total_height = num_rows * crop_size[1] + (num_rows + 1) * margin_y
    collage = Image.new("RGBA", (total_width, total_height), (0, 0, 0, 0))

    for idx, img_path in enumerate(image_paths):
        if vertical:
            col, row = divmod(idx, images_per_row)
        else:
            row, col = divmod(idx, images_per_row)

        x = margin_x + col * (crop_size[0] + margin_x)
        y = margin_y + row * (crop_size[1] + margin_y)

        try:
            with Image.open(img_path).convert("RGBA") as img:
                cropped_img = crop_center(img, crop_ratio)
                if cropped_img.size != crop_size:
                    cropped_img = cropped_img.resize(
                        crop_size, Image.Resampling.BICUBIC
                    )
        except Exception:
            cropped_img = create_placeholder_image(crop_size)

        collage.paste(cropped_img, (x, y), cropped_img)

    collage.save(output_path, dpi=dpi)
    print(f"拼图已保存到 {output_path} (DPI: {dpi})")
