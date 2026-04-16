1. MinkowskiEngine
conda install openblas-devel -c anaconda
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
2. TorchSparse
sudo apt-get install libsparsehash-dev
python setup.py install
3. chamfer3D
python setup.py install
4. emd
python setup.py install
5. pc_util
python setup.py install