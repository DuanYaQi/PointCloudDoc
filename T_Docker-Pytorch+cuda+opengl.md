# Docker-Pytorch+cuda+opengl

```bash
# 添加超链接
# duan @ duan-Lenovo-ideapad-Y700-15ISK in ~/windows/Flow-based/Doc [13:29:27] C:127
ln -s ../../PointCloud/Doc/wechat_zhihu.md wechat_zhihu.md

ln -s /home/data/MSN/data/ data

ln -s /home/data/latent_3d_points_Pytorch/shape_net_core_uniform_samples_2048/ shape_net_core_uniform_samples_2048

ln -s /home/data/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/ shapenetcore_partanno_segmentation_benchmark_v0

ln -s /home/data/MSN-Point-Cloud-Completion/data data
```



## 运行

````bash
docker run --runtime=nvidia --rm -it -w /home -v /home/duan/windows/udata:/home/data/ -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE nvidia/cudagl:10.2-devel-ubuntu18.04

docker commit -p 43d4f8e0674a nvidia/cudagl:10.2-devel-ubuntu18.04

tensorboard --logdir ./runs/lightning_logs
# PU-Flow
````

```bash
docker run --runtime=nvidia --rm -it -w /home -v /home/duan/windows/udata:/home/data/ -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE nvidia/cudagl:duan

docker commit -p 578cdbcbaca7 nvidia/cudagl:duan
# latent_3d_points_Pytorch  
# pointnet.pytorch
# NICE
```



## cuda+opengl

```bash
# https://hub.docker.com/r/nvidia/cudagl
docker pull nvidia/cudagl:10.2-devel-ubuntu18.04
```



## 安装

```bash
# 更换软件源
rm -r /etc/apt/sources.list.d
mv /etc/apt/sources.list /etc/apt/sources.list.bak

echo "deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse" > /etc/apt/sources.list

cat /etc/apt/sources.list
apt clean
apt update
```



```bash
# 安装sudo vi git
apt install -y sudo vim git xarclock wget cmake zip


# optional 本地 xhost+
xarclock

apt install -y mesa-utils
glxinfo |grep rendering #提示：direct rendering: Yes 表明启动正常；
glxgears #里面有3个转动的齿轮，并且终端每5秒显示出转动多少栅；
```



```bash
# 安装python
apt install -y python python3 python3-pip python3-tk
ln -sf /usr/bin/python3 /usr/bin/python
ln -sf /usr/bin/pip3 /usr/bin/pip
rm -r /var/lib/apt/lists/*
```



```bash
# 安装pip
pip install --upgrade pip

# 换源1
mkdir ~/.pip
cd ~/.pip/
vi pip.conf

[global]
index-url = http://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=mirrors.aliyun.com

# 换源2
pip install --no-cache-dir --user pqi
pqi use aliyun 
```



## pip库

```bash
# pytorch torchvision
# https://pytorch.org/get-started/locally/
# torch 1.7.0 & CUDA 10.2
pip install --no-cache-dir torch torchvision pytorch_lightning

pip install --no-cache-dir open3d && apt install -y libgl1-mesa-glx
pip install --no-cache-dir omegaconf h5py

pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
wget -P /usr/bin https://github.com/unlimblue/KNN_CUDA/raw/master/ninja && chmod +x /usr/bin/ninja

pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

cd metric/ && git clone https://github.com/daerduoCarey/PyTorchEMD
cd PyTorchEMD/
vi cuda/emd_kernel.cu

AT_CHECK>>>>>>>>>>>>>>>>>>TORCH_CHECK
line16/17

python setup.py install
cp build/lib.linux-x86_64-3.6/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
```





---

