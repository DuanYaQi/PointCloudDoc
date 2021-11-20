# Shape As Point

NIPS2021_Shape As Points: A Differentiable Poisson Solver



## conda

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes


conda env create -f environment.yaml
conda activate sap
```







## code

package



Trimesh 是一个纯 Python (2.7-3.4+) 库，用于加载和使用[三角形网格](https://en.wikipedia.org/wiki/Triangle_mesh)，重点是 watertight  曲面。

```shell
pip install trimesh
pip install scikit-image
```



*scikit-image*是图像处理算法的集合。



PyTorch3D通过PyTorch为3D计算机视觉研究提供高效，可重复使用的组件。

```shell
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu102_pyt190/download.html
```



PyTorch Scatter 优化分散 scatter





## Error

ERROR: After October 2020 you may experience errors when installing or updating packages. This is because pip will change the way that it resolves dependency conflicts.

We recommend you use --use-feature=2020-resolver to test your packages with the new resolver before it becomes the default.

imageio 2.11.0 requires pillow>=8.3.2, but you'll have pillow 8.0.0 which is incompatible.