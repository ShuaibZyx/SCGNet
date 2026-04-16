from os.path import join, exists, dirname
import os
from tqdm import tqdm

dataset_path = "/root/autodl-tmp"
city = "Tallinn"
result_path = f"/root/autodl-fs/point2building/results/{city}/meshes"

lack_list = []
val_txt = join(dataset_path, f"{city}_Filtered", "val.txt")
with open(val_txt, "r") as vt:
    gt_mesh_list = [vt.strip() for vt in vt.readlines()]


for mesh_name in tqdm(gt_mesh_list, desc="detecting lack files"):
    pred_mesh_path = join(result_path, f"{mesh_name}.obj")
    if not exists(pred_mesh_path):
        lack_list.append(mesh_name)

lack_save_path = join(dirname(result_path), "lack_files.txt")

with open(lack_save_path, "w") as lsp:
    for lack_mesh in lack_list:
        lsp.write(lack_mesh + "\n")
print(f"Lack files are saved at {lack_save_path}")
print(f"Total lack files: {len(lack_list)}")
