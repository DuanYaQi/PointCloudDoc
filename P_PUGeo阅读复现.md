# PUGeo



```shell
docker run --runtime=nvidia --rm -it -w /home -v /home/duan/windows/udata:/home/data/ -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE tensorflow/tensorflow:1.13.2-gpu

docker commit -p 5550435f968f tensorflow/tensorflow:1.13.2-gpu
```





```
python main.py --phase test --up_ratio 4 --pretrained PUGeo_x4/model/model-final --eval_xyz input-thing10k

python main.py --phase test --up_ratio 4 --pretrained PUGeo_x4/model/model-final --eval_xyz input-thing10k/xyz-fps

```





## error

**cannot find -ltensorflow_framework**

```
In my cae, the tensorflow_framework library file's name was not libtensorflow_framework.so but libtensorflow_framework.so.1. The tensorflow version is 1.14.0.

I just made symbolic link for it and the problem has gone.

$ cd /usr/local/lib/python2.7/dist-packages/tensorflow/
$ ln -s libtensorflow_framework.so.1 libtensorflow_framework.so

```



**编译后，无法load库**

```error
_ZN10tensorflow12OpDefBuilder5InputENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```



将原始sh文件

```sh
#/bin/bash
CUDA=/usr/local/cuda-10.0
TF=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

$CUDA/bin/nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TF/include -I $CUDA/include -I $TF/include/external/nsync/public -lcudart -L $CUDA/lib64/ -L $TF -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
```

替换为下列sh文件

```sh
#!/usr/bin/env bash
CUDA=/usr/local/cuda-10.0
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

$CUDA/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.13
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TF_INC -I $CUDA/include -I $TF_INC/external/nsync/public -lcudart -L $CUDA/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
```





将原始sh文件

```sh
#!/bin/bash

CUDA=/usr/local/cuda-10.0
TF=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')


$CUDA/bin/nvcc -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $TF/include -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $TF/include -lcudart -L $CUDA/lib64 -O2 -I $TF/include/external/nsync/public -L $TF -ltensorflow_framework
```

替换为下列sh文件

```sh
#!/usr/bin/env bash
CUDA=/usr/local/cuda-10.0
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

$CUDA/bin/nvcc -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $TF_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

# TF1.13
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $TF_INC -I $CUDA/include -I $TF_INC/external/nsync/public -lcudart -L $CUDA/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
```



如果还不行

查看tensorflow版本 1.14.0不支持1.13.2支持

