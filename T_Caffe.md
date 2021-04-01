## Caffe安装  （ENet）

#### 官方教程

 http://caffe.berkeleyvision.org/installation.html



cmake方法编译配置    lists.txt

https://github.com/BVLC/caffe/pull/1667



cmakelist.txt                CPU_ONLY = ON

Makefile.config            WITH_PYTHON_LAYER := 1 

export PYTHONPATH=/home/Workspace/ENet/caffe-enet/python:$PYTHONPATH



```shell
python test_segmentation.py --model ~/Workspace/ENet/prototxts/enet_deploy_final.prototxt --weights ~/Workspace/ENet/enet_weights_zoo/cityscapes_weights.caffemodel --colours ~/Workspace/ENet/scripts/cityscapes19.png --input_image ~/Workspace/ENet/example_image/munich_000000_000019_leftImg8bit.png --out_dir ~/Workspace/ENet/example_image/
```





#### error-1 This file requires compiler and library support for the ISO C++ 2011

修改CMakeLists.txt中，SET(CMAKE_CXX_FLAGS " ")为SET(CMAKE_CXX_FLAGS "-std=c++0x")



#### error-2 ‘CV_LOAD_IMAGE_COLOR’ was not declared in this scope

 找到报错前的文件，如：

> /home/user/caffe/src/caffe/util/io.cpp:76:34: error: ‘CV_LOAD_IMAGE_COLOR’ was not declared in this scope

就编辑/home/user/caffe/src/caffe/util/io.cpp这个文件，

> 将CV_LOAD_IMAGE_COLOR   改成    cv::IMREAD_COLOR
> 将CV_LOAD_IMAGE_GRAYSCALE   改成   cv::IMREAD_GRAYSCALE

https://blog.csdn.net/qq_28660035/article/details/80772071



#### error-3: ‘CV_RGB2GRAY’ was not declared in this scope 

在头文件里添加

> #include <opencv2/imgproc/types_c.h>

https://www.cnblogs.com/lonelypinky/p/11586388.html



#### error-4 import caffe失败 No module named caffe

A.把环境变量路径放到 ~/.bashrc文件中,打开文件

> sudo vim ~/.bashrc  

在文件下方写入

> export PYTHONPATH=~/caffe/python:$PYTHONPATH 

上述语句中 “～” 号表示caffe 所在的根目录。



B.关闭文件，在终端写入下面语句，使环境变量生效

> source ~/.bashrc 



还有另一种方法，直接在运行的程序前加上如下代码：

> import sys
>
> sys.path.append("/home/用户名/caffe/python")
>
> import os,sys,caffe



#### error-5 ImportError：No module named skimage.io

> sudo apt-get install python-skimage



#### error-6 ImportError: No module named google.protobuf.internal

> sudo apt-get install python-protobuf



#### error-7 TypeError: __init__() got an unexpected keyword argument 'syntax

https://blog.csdn.net/e01528/article/details/81282343



#### error-8 ubuntu下运行python提示: no module named pip

https://blog.csdn.net/qq_36269513/article/details/80450421?utm_source=blogxgwz0







ubuntu16.04编译caffe(CPU版)



```shell
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
```

```shell
$0 类似于python中的sys.argv[0]等。 
$0 指的是Shell本身的文件名。类似的有如果运行脚本的时候带参数，那么$1 就是第一个参数，依此类推。

dirname 用于取指定路径所在的目录 ，如 dirname /home/ikidou 结果为 /home。
$ 返回该命令的结果
pwd -P 如果目录是链接时，格式：pwd -P 显示出实际路径，而非使用连接（link）路径。
```