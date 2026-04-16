import sys

sys.path.append("..")
import os
from os.path import join, basename, splitext
import numpy as np
from glob import glob
from tqdm import tqdm
import json
import src.utils.data_utils as data_utils
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration Parameters")
    parser.add_argument(
        "--exp_tag",
        type=str,
        required=True,
        help="exp_tag",
    )
    parser.add_argument(
        "--city", type=str, default="Zurich", help="Choose Whitch City."
    )
    args = parser.parse_args()

    mesh_files = sorted(glob(join("../results", args.exp_tag, args.city, "meshes", "*.obj")))
    with open(join("../datasets", args.city, "testset/info.json")) as f:
        data_info = json.load(f)

    material_list = []
    vs_list = []
    fs_list = []

    for i, mesh_file in enumerate(tqdm(mesh_files)):
        vertices, faces = data_utils.load_obj(mesh_file)
        one_data_info = data_info[splitext(basename(mesh_file))[0]]
        vertices = vertices * one_data_info["scale"] + one_data_info["center"]
        material_list.append(i)
        vs_list.append(vertices)
        fs_list.append(faces)

    offset_f = np.hstack([1, np.cumsum([len(item) for item in vs_list])[:-1] + 1])
    new_fs_list = []
    for i, fs in enumerate(fs_list):
        new_fs_list.append([[x + offset_f[i] for x in row] for row in fs])
    vs_list = np.vstack(vs_list)

    with open(join("../results", args.exp_tag, args.city, f"{args.city}_testset_pred.obj"), "w") as wf:
        wf.write("mtllib cad.mtl\n")
        for v in vs_list:
            wf.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for j, new_fs in enumerate(new_fs_list):
            wf.write("usemtl Material{}\n".format(material_list[j]))
            for new_f in new_fs:
                face_indices = " ".join(str(idx) for idx in new_f)
                wf.write(f"f {face_indices}\n")
    # TODO
    print("vis city done")
