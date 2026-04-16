import os
import trimesh
import numpy as np
import open3d as o3d
from tqdm.auto import tqdm
from glob import glob
import pickle
from os.path import join, dirname, basename, splitext, exists
import argparse


# 加载生成的模型
def load_meshes(results_path, new_meshes=False):
    cache_file = join(results_path, "predict_meshes.pkl")
    if not new_meshes and exists(cache_file):
        with open(cache_file, "rb") as f:
            pred_meshes = pickle.load(f)
    else:
        pred_meshes = glob(join(results_path, "*.obj"))
        with open(cache_file, "wb") as f:
            pickle.dump(pred_meshes, f)
    return pred_meshes


# 渲染点云图像
def draw_point_cloud(vis_points, path):
    width, height = 800, 600
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vis_points)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path)
    vis.destroy_window()


# 创建文件夹
def create_folder(path):
    os.makedirs(path, exist_ok=True)


# 保存点云数据并渲染图像
def save_point_cloud(points, vis_points, file_path, vis):
    np.save(file_path, points)
    if vis:
        img_path = join(
            dirname(file_path).replace("npy", "img"),
            f"{splitext(basename(file_path))[0]}.png",
        )
        create_folder(dirname(img_path))
        if vis_points is not None:
            draw_point_cloud(vis_points, img_path)


# 采样点云数据
def sample_point_cloud(save_path, pred_meshes, num_points=10000, vis=True, test=False):
    create_folder(save_path)
    for i, pred_mesh_path in tqdm(
        enumerate(pred_meshes), total=len(pred_meshes), desc="sample"
    ):
        # print(f"handle:{pred_mesh_path}")
        pred_mesh = trimesh.load_mesh(pred_mesh_path)
        pred_points = pred_mesh.sample(num_points)
        vis_points = None
        if vis:
            vis_mesh = pred_mesh.copy()
            vis_mesh.vertices = vis_mesh.vertices[:, [1, 2, 0]]
            vis_points = vis_mesh.sample(num_points)
        file_path = join(save_path, f"{splitext(basename(pred_mesh_path))[0]}.npy")
        save_point_cloud(pred_points, vis_points, file_path, vis)
        if test and i >= 10:  # 如果test为True且已处理10个模型，则停止
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample Meshes")
    parser.add_argument(
        "--exp_tag",
        type=str,
        required=True,
        help="exp_tag",
    )
    parser.add_argument(
        "--city", type=str, default="Zurich", help="Choose Whitch City."
    )
    parser.add_argument(
        "--num_points", type=int, default=10000, help="Sample Number of points"
    )
    parser.add_argument(
        "--vis", type=bool, default=False, help="Visualize the vis point cloud Or Not"
    )
    parser.add_argument("--test", type=bool, default=False, help="Test the code Or Not")
    args = parser.parse_args()

    # testset sample
    # testset_path = join("../datasets", args.city, "testset/meshes")
    # save_path = join("../datasets", args.city, "testset_simple/npy")

    results_path = join("../results", args.exp_tag, args.city, "meshes")
    save_path = join("../results", args.exp_tag, args.city, "sample", "npy")

    pred_meshes = load_meshes(results_path, new_meshes=True)

    sample_point_cloud(
        save_path=save_path,
        pred_meshes=pred_meshes,
        num_points=args.num_points,
        vis=args.vis,
        test=args.test,
    )
