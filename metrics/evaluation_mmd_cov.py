import torch
from tqdm.auto import tqdm
from eval_utils import *
import os
from os.path import join, dirname, splitext
from tqdm import tqdm
import argparse
from extensions.chamfer3D.dist_chamfer_3D import chamfer_3DDist_nograd
from time import time
from pathlib import Path


def chamfer_distance_l2(x, y):
    # expect B.2048.3 and B.2048.3
    cdl2_fun = chamfer_3DDist_nograd()
    dist1, dist2, _, _ = cdl2_fun(x, y)
    return dist1, dist2


def _pairwise_CD_(sample_pcs, ref_pcs, batch_size):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    for sample_b_start in tqdm(range(N_sample), desc="Pairwise CD"):
        sample_batch = sample_pcs[sample_b_start]
        cd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]
            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1
            )
            sample_batch_exp = sample_batch_exp.contiguous()
            dl, dr = chamfer_distance_l2(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))
        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)
    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    return all_cd


def mmd_cov(all_dist, multiple):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "mmd": mmd * multiple,
        "cov": cov,
        "mmd_smp": mmd_smp * multiple,
    }


def compute_all_metrics(sample_pcs, ref_pcs, batch_size, multiple):
    results = {}
    M_rs_cd = _pairwise_CD_(ref_pcs, sample_pcs, batch_size)

    ## CD
    res_cd = mmd_cov(M_rs_cd.t(), multiple)
    results.update({"%s-CD" % k: v for k, v in res_cd.items()})

    for k, v in results.items():
        print("[%s] %.8f" % (k, v.item()))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation NNA")
    parser.add_argument("--flag", type=str, default="Test")
    parser.add_argument("--city", type=str, default="Zurich")
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--multiple", type=int, default=1000)
    args = parser.parse_args()

    now = int(time())
    batch_size = args.batch_size

    result_dir = join(
        Path(__file__).resolve().parents[1], "results", args.flag, args.city
    )

    city = args.city
    pred_dir = f"/home/kemove/devdata1/zyx/results/City3D_Result/{city}/success/meshes"
    gt_dir = f"/home/kemove/devdata1/zyx/datasets/{city}_Filtered/meshes"

    # pred_dir = join(result_dir, "meshes")
    # gt_dir = join(result_dir, "meshes_GT")

    with open(join(dirname(gt_dir), "val.txt"), "r") as vt:
        mesh_list = [mesh.strip() for mesh in vt.readlines()]
    # mesh_list = [name for name in os.listdir(gt_dir) if name.endswith(".obj")]

    predict = []
    truth = []

    error_files = []
    error_files_txt = join(dirname(pred_dir), "error_files.txt")

    for mesh_name in tqdm(mesh_list, desc="Load Points"):
        try:
            mesh_id = splitext(mesh_name)[0]
            pred_path = join(pred_dir, mesh_id + ".obj")
            gt_path = join(gt_dir, mesh_id + ".obj")

            if not os.path.exists(pred_path):
                continue

            pred_pc = sample_pc(pred_path, args.npoints)
            gt_pc = sample_pc(gt_path, args.npoints)

            predict.append(get_pc_tensor(pred_pc))
            truth.append(get_pc_tensor(gt_pc))

        except Exception as e:
            error_files.append(mesh_name)
            print(f"Error generating mesh for {mesh_name}: {e}\n")

    with open(error_files_txt, "w") as eft:
        for error_file in error_files:
            eft.write(error_file + "\n")

    predict = torch.stack(predict, dim=0)
    truth = torch.stack(truth, dim=0)

    results = compute_all_metrics(predict, truth, batch_size, args.multiple)

    filename = f"cov_mmd_result/{args.flag}/{args.city}-{args.npoints}-{args.batch_size}-{args.multiple}-{now}.csv"
    os.makedirs(dirname(filename), exist_ok=True)
    save_results_to_csv(results, filename)
