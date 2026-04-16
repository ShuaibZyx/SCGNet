import os
from os.path import join, exists, dirname
import subprocess
import signal
import argparse
from tqdm import tqdm  # 导入 tqdm 用于进度条

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Rendering Obj Files With Blender.")
parser.add_argument(
    "--exp_tag",
    type=str,
    required=True,
    help="exp_tag",
)
parser.add_argument("--city", type=str, default="Zurich", help="Choose Whitch City.")
parser.add_argument(
    "--resolution", type=int, default=640, help="Resolution for processing."
)
parser.add_argument(
    "--use_material",
    type=bool,
    default=False,
    help="Whether to use material for processing.",
)
parser.add_argument(
    "--transpose",
    type=bool,
    default=True,
    help="Whether to transpose mesh.",
)

# 解析命令行参数
args = parser.parse_args()

input_dir = join("../results", args.exp_tag, args.city, "meshes")
output_dir = join("../results", args.exp_tag, args.city, "render")


# 定义一个处理中断的函数
def handle_interrupt(signum, frame):
    print("Processing interrupted by user")
    exit(1)


# 注册信号处理函数
signal.signal(signal.SIGINT, handle_interrupt)

if not exists(output_dir):
    os.makedirs(output_dir)

# 获取 object_path 目录下的所有 .obj 文件
obj_files = [f for f in os.listdir(input_dir) if f.endswith(".obj")]

# 定义日志文件路径
log_file_path = join(dirname(output_dir), "blender_output.log")

# 使用 tqdm 创建进度条
for obj_file in tqdm(obj_files, desc="Render Meshes"):
    obj_file_path = join(input_dir, obj_file)
    # 构建命令行参数
    command = [
        "blender",
        "-b",
        "-noaudio",
        "-P",
        "blender_script.py",
        "--",
        "--object_path",
        obj_file_path,
        "--output_dir",
        output_dir,
        "--resolution",
        str(args.resolution),
        "--use_material",
        str(args.use_material).lower(),
        "--transpose",
        str(args.transpose).lower(),
    ]
    # 执行命令，并将输出重定向到日志文件
    with open(log_file_path, "a") as log_file:  # 'a' 模式追加内容到日志文件
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
