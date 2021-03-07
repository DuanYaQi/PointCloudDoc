# Colab

修改-笔记本设置-GPU



## 挂载云盘

```python
from google.colab import drive
drive.mount('/content/drive')
```



```python
import os
os.chdir("/content/drive/My Drive")
!ls

!git clone https://github.com/DuanYaQi/latent_3d_points_Pytorch.git

cd latent_3d_points_Pytorch

!sh download_data.sh
```





## 查看配置

```python
! /opt/bin/nvidia-smi
```



## 同步

```python
drive.flush_and_unmount()
```

