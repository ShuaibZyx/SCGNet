import sys

sys.path.append("..")
import os
from os.path import join, dirname, splitext, exists
from tqdm import tqdm
import json
import torch
import src.utils.data_utils as data_utils
import argparse
from inference.generate_utils import *
from pathlib import Path
import csv
import time


def generate_mesh_check(
    vertex_model, face_model, batch, file_data_info, args, cuda_device, cpu_device
):

    save_vertices = None
    save_faces = None

    run_v_id = 0
    while run_v_id < 10:
        pred_v = sample_from_vertex_model(
            vertex_model, batch, args.top_p_v, args.sample_mask, cuda_device
        )
        if not v_have_stop_token(pred_v):
            run_v_id += 1
            continue

        run_f_id = 0
        f_batch = preprocess_v(
            pred_v,
            batch["pc_sparse"],
            cuda_device,
        )
        while run_f_id < 10:
            # print('run {} iterations'.format(run_v_id*10+run_f_id))
            pred_f = sample_from_face_model(
                face_model, f_batch, args.top_p_f, args.sample_mask, cuda_device
            )

            if not f_have_stop_token(pred_f):
                run_f_id += 1
                continue

            if_floor_cover, vs_floor_inds = is_floor_covering_pointcloudxy(
                pred_v.to(cpu_device),
                data_utils.dequantize_verts(f_batch["pc_sparse"].C[:, 1:], 8)
                .detach()
                .to(cpu_device),
                file_data_info,
            )

            if vs_floor_inds is None:
                break

            if not if_floor_cover:
                if are_missing_floor_vertices(pred_v.to(cpu_device), vs_floor_inds):
                    break
                elif are_missing_floor_faces(
                    pred_v.to(cpu_device), pred_f, vs_floor_inds
                ):
                    run_f_id += 1
                    continue

            save_vertices = (
                data_utils.dequantize_verts(f_batch["vs_sparse"].C[:, 1:])
                .detach()
                .to(cpu_device)
            )
            save_faces = pred_f

            run_v_id = 10
            break

        run_v_id += 1
    return save_vertices, save_faces


def generate_mesh_single(
    vertex_model, face_model, batch, args, cuda_device, cpu_device
):
    save_vertices = None
    save_faces = None

    pred_v = sample_from_vertex_model(
        vertex_model, batch, args.top_p_v, args.sample_mask, cuda_device
    )
    f_batch = preprocess_v(
        pred_v,
        batch["pc_sparse"],
        cuda_device,
    )
    pred_f = sample_from_face_model(
        face_model, f_batch, args.top_p_f, args.sample_mask, cuda_device
    )

    save_vertices = (
        data_utils.dequantize_verts(f_batch["vs_sparse"].C[:, 1:])
        .detach()
        .to(cpu_device)
    )
    save_faces = pred_f
    return save_vertices, save_faces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Meshes")
    parser.add_argument("--flag", type=str, required=True, help="实验标识")
    parser.add_argument(
        "--vc",
        type=str,
        default="02251006_Vertex-Full-TorchSparse-Tallinn_brass-sitrep",
        help="要加载的顶点模型权重名称",
    )
    parser.add_argument(
        "--fc",
        type=str,
        default="03232127_Full-Tallinn-Best_frayed-marsanne",
        help="要加载的平面模型权重名称",
    )
    parser.add_argument(
        "--city", type=str, default="Zurich", help="选中哪个城市的数据集"
    )
    parser.add_argument(
        "--top_p_v", type=float, default=0.9, help="顶点模型核采样参数top_p"
    )
    parser.add_argument(
        "--top_p_f", type=float, default=0.9, help="平面模型核采样参数top_p"
    )
    parser.add_argument(
        "--sample_mask", type=bool, default=True, help="是否使用sample_mask采样方法"
    )
    parser.add_argument(
        "--single_inference", type=bool, default=False, help="是否使用单次推理"
    )
    args = parser.parse_args()

    project_path = Path(__file__).resolve().parents[1]
    print(project_path)
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")
    city = args.city

    test_data_path = f"/home/kemove/devdata1/zyx/datasets/{city}_Filtered/{city}_Filtered_Polygon_Processed.pkl"
    test_json_path = join(dirname(test_data_path), "info.json")

    with open(test_json_path) as f:
        data_info = json.load(f)

    vertex_checkpoint_name = args.vc
    face_checkpoint_name = args.fc

    vertex_checkpoint_path = join(
        project_path, "runs", vertex_checkpoint_name, "checkpoints", "last.ckpt"
    )
    face_checkpoint_path = join(
        project_path, "runs", face_checkpoint_name, "checkpoints", "last.ckpt"
    )

    print(f"vertex_checkpoint_path: {vertex_checkpoint_path}")
    print(f"face_checkpoint_path: {face_checkpoint_path}")

    vertex_model = load_vertex_model(vertex_checkpoint_path, cuda_device)
    face_model = load_face_model(face_checkpoint_path, cuda_device)

    test_dataloader = load_test_dataloader(test_data_path)

    mesh_save_dir = join(
        project_path,
        "results",
        f"{args.flag}",
        city,
        "meshes",
    )
    os.makedirs(mesh_save_dir, exist_ok=True)
    generate_info_txt = join(dirname(mesh_save_dir), "lack_files.txt")
    generate_info_csv = join(dirname(mesh_save_dir), "generate_info.csv")
    lack_files = []
    error_files = []

    # 检查 CSV 文件是否存在，如果不存在则写入表头
    file_exists = os.path.isfile(generate_info_csv)

    with open(generate_info_csv, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["model_id", "vertices", "faces", "seconds"]
        )

        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()

        for i, test_batch in tqdm(
            enumerate(test_dataloader),
            total=len(test_dataloader),
            desc="Generate Mesh",
        ):
            # 初始化数据
            batch_to_device(test_batch, cuda_device)
            filename = splitext(test_batch["filename"][0])[0]

            mesh_file = join(mesh_save_dir, f"{filename}.obj")

            if exists(mesh_file):
                continue

            file_data_info = data_info[filename]

            try:
                start_time = time.perf_counter()
                # 采样顶点和平面
                if args.single_inference:
                    save_vertices, save_faces = generate_mesh_single(
                        vertex_model,
                        face_model,
                        test_batch,
                        args,
                        cuda_device,
                        cpu_device,
                    )
                else:
                    save_vertices, save_faces = generate_mesh_check(
                        vertex_model,
                        face_model,
                        test_batch,
                        file_data_info,
                        args,
                        cuda_device,
                        cpu_device,
                    )

                # 处理结果
                if save_vertices is not None and save_faces is not None:
                    data_utils.write_obj(save_vertices, save_faces, mesh_file)
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    writer.writerow(
                        {
                            "model_id": filename,
                            "vertices": len(save_vertices),
                            "faces": len(save_faces),
                            "seconds": round(elapsed_time, 2),
                        }
                    )
                else:
                    lack_files.append(filename)
            except Exception as e:
                error_files.append(filename)
                print(f"Error generating mesh for {filename}: {e}\n")

    # 记录实验信息以及缺失的文件信息
    with open(generate_info_txt, "w") as generate_info_txt:
        generate_info_txt.write(f"vertex model: {vertex_checkpoint_name}\n")
        generate_info_txt.write(f"face model: {face_checkpoint_name}\n")
        if len(lack_files) > 0:
            generate_info_txt.write("Below are the sample check fail documents:\n")
        for lack_file in lack_files:
            generate_info_txt.write(lack_file + "\n")
        if len(error_files) > 0:
            generate_info_txt.write("Below are the error documents:\n")
        for error_file in error_files:
            generate_info_txt.write(error_file + "\n")
