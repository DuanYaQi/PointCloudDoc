# PU-GAN复现



## start

```
docker run --runtime=nvidia --rm -it tensorflow/pu-gan:latest


docker run --runtime=nvidia --rm -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE tensorflow/pu-gan:latest
```



```
docker cp /home/duan/Downloads/4577251581f7b1fe1dea6f6320002e46ba1348b4-PatchSRFlow-resort4-100epoch.ckpt 6e90cd0b4f1e:/home/unknownue/PU-Flow/
```



```
docker ps -a
docker commit -a "duan" -m "xxxx" -p 6e90cd0b4f1e tensorflow/pu-gan:latest
```



```bash
ssh-keygen -t rsa -C "837738300@qq.com" 
cat ~/.ssh/id_rsa.pub
git remote set-url origin git@xxx.com:xxx/xxx.git
```





## 问题

1. 安装tensorflow后报错ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory

```
版本不匹配，cuda和
```



2. 安装tensorflow1.11.0后出现

```
pytorch-lightning 0.9.0 requires tensorboard==2.2.0, but you'll have tensorboard 1.11.0 which is incompatible.
```

​	导致这两个库无法使用，需要重新安装



3. 使用后./evaluation: error while loading shared libraries: libCGAL.so.13: cannot open shared object file: No such file or directory

```
先安装cgal
https://www.cgal.org/download/linux.html

cmake-gui 修改CGAL_DIR的路径
/usr/local/lib/cmake/CGAL

cmake时需要Realease模式
cmake -DCMAKE_BUILD_TYPE=Release .
```

