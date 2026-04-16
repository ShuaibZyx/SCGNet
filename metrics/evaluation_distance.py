import argparse
from tqdm import tqdm
import os
from os.path import join, splitext, dirname
import csv
import point_cloud_utils as pcu
from eval_utils import *
from time import time
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration Parameters")
    parser.add_argument("--flag", type=str, default="Test")
    parser.add_argument("--city", type=str, default="Zurich")
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--multiple", type=int, default=100)
    args = parser.parse_args()

    now = int(time())
    result_dir = join(
        Path(__file__).resolve().parents[1], "results", args.flag, args.city
    )

    city = args.city

    pred_dir = f"/home/kemove/devdata1/zyx/P2B-PCFVWA-TorchSparse/results/{args.flag}/{args.city}/meshes"
    gt_dir = f"/home/kemove/devdata1/zyx/datasets/{city}_Filtered/meshes"
    pts_dir = f"/home/kemove/devdata1/zyx/datasets/{city}_Filtered/pointclouds"

    # pred_dir = join(result_dir, "meshes")
    # gt_dir = join(result_dir, "meshes_GT")
    # pts_dir = join(result_dir, "pointclouds")

    print(f"pred_mesh_dir = {pred_dir}")

    with open(join(dirname(gt_dir), "val.txt"), "r") as vt:
        mesh_list = [mesh.strip() for mesh in vt.readlines()]
    # mesh_list = [name for name in os.listdir(gt_dir) if name.endswith(".obj")]

    complexity_dict = {"Average Vertices": 0, "Average Faces": 0, "Failure Number": 0}

    result_dict = {
        "HD": 0,
        "CD(inp->rec)": 0,
        "CD(rec->ref)": 0,
        "RMSE(inp->rec)": 0,
        "EMD": 0,
        "F1-Score": 0,
        "Precision": 0,
        "Recall": 0,
    }

    metrics_info_csv = f"distance_result_detail/{args.flag}/{args.city}-{args.npoints}-{args.threshold}-{args.multiple}-{now}.csv"
    os.makedirs(dirname(metrics_info_csv), exist_ok=True)
    with open(metrics_info_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Mesh_Id",
                "HD",
                "CD(inp->rec)",
                "CD(rec->ref)",
                "RMSE(inp->rec)",
                "EMD",
                "F1-Score",
                "Precision",
                "Recall",
            ],
        )
        # 写入列标题
        writer.writeheader()
        lack_files = []
        for i, mesh_name in enumerate(tqdm(mesh_list, desc="Compute Metrics")):
            try:
                mesh_id = splitext(mesh_name)[0]
                pred_path = join(pred_dir, mesh_id + ".obj")
                gt_path = join(gt_dir, mesh_id + ".obj")
                pts_path = join(pts_dir, mesh_id + ".xyz")

                # print(f"pred_path = {pred_path}")
                # print(f"gt_path = {gt_path}")
                # print(f"pts_path = {pts_path}")

                if not os.path.exists(pred_path):
                    # print(f"{pred_path} is not exist")
                    lack_files.append(pred_path)
                    continue

                vertices, faces = load_obj(pred_path)

                gt_pc = sample_pc(gt_path, args.npoints)
                pred_pc = sample_pc(pred_path, args.npoints)
                input_pts = normalize_points(load_xyz(pts_path)).astype(np.float64)

                hd = pcu.hausdorff_distance(pred_pc, gt_pc) * args.multiple
                cd_inp = pcu.chamfer_distance(input_pts, pred_pc) * args.multiple
                cd_rec = pcu.chamfer_distance(pred_pc, gt_pc) * args.multiple
                rmse_inp = compute_rmse_open3d(input_pts, pred_pc) * args.multiple
                emd = emd_distance(gt_pc, pred_pc) * args.multiple

                fscore, pd, rd = pc_fscore(gt_pc, pred_pc, threshold=args.threshold)

                result_dict["HD"] += hd
                result_dict["CD(inp->rec)"] += cd_inp
                result_dict["CD(rec->ref)"] += cd_rec
                result_dict["RMSE(inp->rec)"] += rmse_inp
                result_dict["EMD"] += emd
                result_dict["F1-Score"] += fscore
                result_dict["Precision"] += pd
                result_dict["Recall"] += rd

                complexity_dict["Average Vertices"] += len(vertices)
                complexity_dict["Average Faces"] += len(faces)

                writer.writerow(
                    {
                        "Mesh_Id": mesh_id,
                        "HD": round(hd, 4),
                        "CD(inp->rec)": round(cd_inp, 4),
                        "CD(rec->ref)": round(cd_rec, 4),
                        "RMSE(inp->rec)": round(rmse_inp, 4),
                        "EMD": round(emd, 4),
                        "F1-Score": round(fscore, 4),
                        "Precision": round(pd, 4),
                        "Recall": round(rd, 4),
                    }
                )
            except Exception as e:
                lack_files.append(pred_path)
                print(f"Error Compute Metrics for {mesh_id}: {e}\n")
                continue

    print(f"mesh_list = {len(mesh_list)}")
    print(f"lack_files = {len(lack_files)}")
    metrics_mesh_count = len(mesh_list) - len(lack_files)
    for key in result_dict.keys():
        result_dict[key] /= metrics_mesh_count
        print(f"{key} = {result_dict[key]}")

    for key in complexity_dict.keys():
        if key != "Failure Number":
            complexity_dict[key] /= metrics_mesh_count
            print(f"{key} = {complexity_dict[key]}")
    complexity_dict["Failure Number"] = len(lack_files)
    print(f"Failure Number = {complexity_dict['Failure Number']}")

    distance_filename = f"distance_result/{args.flag}/{args.city}-{args.npoints}-{args.threshold}-{args.multiple}-{now}.csv"
    complexity_filename = f"complexity_result/{args.flag}/{args.city}-{now}.csv"
    os.makedirs(dirname(distance_filename), exist_ok=True)
    os.makedirs(dirname(complexity_filename), exist_ok=True)
    save_results_to_csv(result_dict, distance_filename, accelerated_cd=False)
    save_results_to_csv(complexity_dict, complexity_filename, accelerated_cd=False)
