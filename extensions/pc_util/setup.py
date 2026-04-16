from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

sources = glob.glob("src/*.cpp") + glob.glob("src/*.cu")
headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

setup(
    name="pc_util",
    version="1.0",
    ext_modules=[
        CUDAExtension(
            name="pc_util",
            sources=sources,
            extra_compile_args={"cxx": ["-O2", headers], "nvcc": ["-O2", headers]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
