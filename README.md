<p align="center">
<h1 align="center"><strong>Integrating spatial context with generative model for building reconstruction from airborne LiDAR point clouds</strong></h1>
  <p align="center">
    <a href='https://github.com/ShuaibZyx' target='_blank'>Yuxiang Zhao</a>&emsp;
    <a href='https://orcid.org/0000-0002-1829-4006' target='_blank'>Shengwen Li</a>&emsp;
    <a href='https://orcid.org/0000-0001-8969-8879' target='_blank'>Fang Fang</a>&emsp;
    <a href='' target='_blank'>Nan Min</a>&emsp;
    <a href='' target='_blank'>Sishi Gong</a>&emsp;
    <a href='' target='_blank'>Yu Wang</a>&emsp;
    <a href='' target='_blank'>Shunping Zhou</a>&emsp;
    <br>
    China University of Geosciences, Wuhan
    <h2 align="center">IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2026</h2>
  </p>
</p>


<div align="center">
  <a href="https://ieeexplore.ieee.org/abstract/document/11419187"><img src="https://img.shields.io/badge/Paper-📖-blue?"></a> &nbsp;&nbsp;
</div>

## 📌 Code Availability
- [x] Inference - (`inference/generate_meshes.py`)
- [x] Training - (`train_face.py and train_vertex.py`)
- [x] Evalution - (`metrics/evaluation_distance.py and metrics/evaluation_mmd_cov.py`)
- [x] Dataset
- [x] Checkpoint


## 🚀 Getting Started

### 1. Clone the repository and create conda environment

```
git clone https://github.com/ShuaibZyx/SCGNet.git
cd SCGNet

conda create -n scgnet python=3.10
conda activate scgnet
```
### 2. Install PyTorch≥2.1.0 with CUDA support

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

### 3. Install additional dependencies

```
pip install -r requirements.txt (remove the pytorch dependencies)
```
>  Install other dependencies based on the `error messages`.

### 4.Download pre-trained models

Download the pre-trained models using the following links:
- **[Zurich](https://pan.quark.cn/s/988b098a4753)**
- **[Tallinn](https://pan.quark.cn/s/7ea7b2d686b8)**

> **⚠️ Note**
> You need to adjust `face_max_count` and `face_vertex_max_count` in `config/face_model.yaml` based on your dataset. It is recommended that you retrain the model. (I didn’t think that far ahead at the time.) The checkpoints should be located in the root folder under 'runs/flag/checkpoints/***.ckpt'.


### 5.Download dataset

Download the datasets using the following links:

- **[Zurich](https://pan.quark.cn/s/61aa5668beed)**
- **[Tallinn](https://pan.quark.cn/s/8d2c6c86dedc)**
- **[AHN3](https://pan.quark.cn/s/421815e05885)**<br>
> The dataset should be placed in the `data` folder in the root directory, and the path should be updated in the configuration file.


### 5.Evalution
`You need to install the dependencies for metric calculations in the 'extensions' folder.`
```
# MinkowskiEngine
cd metrics/MinkowskiEngine
conda install openblas-devel -c anaconda python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

# TorchSparse
cd metrics/torchsparse
sudo apt-get install libsparsehash-dev python setup.py install

# chamfer3D
cd metrics/chamfer3D
python setup.py install

# pc_util
cd metrics/pc_util
python setup.py install
```
`Next, modify the configuration settings to match your file path, then run 'inference/generate_meshes.py'`


### Acknowledgement

- [Point2Building](https://github.com/prs-eth/point2building)
- [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- [TorchSparse](https://github.com/mit-han-lab/torchsparse)
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
- [City3D](https://github.com/tudelft3d/City3D)


## Citation

```
@ARTICLE{11419187,
  author={Zhao, Yuxiang and Li, Shengwen and Fang, Fang and Min, Nan and Gong, Sishi and Wang, Yu and Zhou, Shunping},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Integrating Spatial Context With Generative Model for Building Reconstruction From Airborne LiDAR Point Clouds}, 
  year={2026},
  volume={64},
  number={},
  pages={1-13}
}
```
