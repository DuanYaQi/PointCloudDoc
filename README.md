# PointCloudDoc

My own research notes. ( e.g. Point Cloud、Deep Learning、Generative Network、Pytorch、Cuda)



---
## PointCloud Paper
> UP 代表 UpSamping，Gen代表Generation，Com 代表 Completion，GNN 代表图神经网络，AE 代表auto-endcoder

### PointCloud Upsampling

| 题目 | 描述 |
| :--: | :-- |
| [SampleNet](P_SampleNet阅读.md) |CVPR2020 B |
| [3PU](P_3PU阅读复现.md)  | CVPR2019 UP     |
| [PU-NET](P_PU-NET阅读复现.md) |CVPR2018 B UP|
| [PU-GAN](P_PU-GAN阅读复现.md) | ICCV2019 UP GAN|
| [PU-Geo](P_PUGeo阅读复现.md) | ECCV2020 UP CG|
| [PCUandNE](P_PCUandNE.md) |VISIGRAPP2021 UP Normal|
| [Meta-PU](P_Meta-PU阅读.md) |arXiv2021 US |
| [MAPU-Net](P_MAPU-Net.md)    | arXiv2020 UP    |
| [SPU-Net](P_SPU-Net阅读.md) |arXiv2020 UP|
| [PU-GCN](P_PU-GCN阅读.md) | arXiv2019 UP GCN |
|      |      |


### PointCloud Completion
| 题目 | 描述 |
| :--: | :-- |
| [Cycle4Completion](P_G_Cycle4Completion.md)     | CVPR2021 Com     |
| [SA-NET](P_C_SA-Net阅读.md) | CVPR2020 Com|
| [EC-NET](P_C_EC-NET阅读复现.md)    | ECCV2018 Com    |
| [MSN](P_C_MSN阅读复现.md) | AAAI2020 Com |
|      |      |


### PointCloud Generation
| 题目 | 描述 |
| :--: | :-- |
| [Diffusion Probabilistic Models for 3D Point Cloud Generation](P_C_diffusionPC阅读.md)    | CVPR2021 Gen     |
| [FoldingNet](P_G_FoldingNet阅读.md)    |CVPR2018 Gen     |
| [Latent_3D_Points](P_G_latent_3d_points阅读复现改写.md)  |ICLR-W2018  AE  |
|      |      |


### PointCloud Learning Representation
| 题目 | 描述 |
| :--: | :-- |
|  [PointNet](P_LR_PointNet阅读复现.md)    | CVPR2017 B |
| [DGCNN](P_LR_DGCNN.md)     |  TOG2019 B、GNN  |
| [PointCNN](P_LR_PointCNN阅读复现.md) | NIPS2018 B|
| [PointNet++](P_LR_PointNet++阅读.md)     |NIPS2017  B    |
|      |      |

---

## Generative Paper

| 题目 | 描述 |
| :--: | :--: |
| [Implicit Normalizing Flows](G_INF.md)     | ICLR2021_Spot INF |
|[GrapAF](G_GraphAF.md)      | ICLR2020 GraphAF     |
| [Glow](G_Glow阅读.md)     | NIPS2018 GLOW     |
| [RealNVP](G_RealNVP阅读.md)     | ICLR2017  RealNVP    |
| [NICE](G_NICE阅读.md)     | ICLR2015 NICE     |
| [VAE](G_VAE.md)     | ICLR2014 VAE     |
| [生成模式概述](G_生成模型概述.md)      |   生成模型概述   |
|      |      |
|      |      |


---
## DeepLearning Paper
| 题目 | 描述 |
| :--: | :--: |
| [Stable architectures for deep neural networks](Pre_StableArch4DNN.md)| 2018Inverse Problems, Stable architectures for deep neural networks |
| [ResNet](Pre_ResNet.md)     |CVPR2016 BestPaper ResNet 残差+恒等连接      |
| [DenseNet](Pre_DenseNet.md)     |CVPR2017 BestPaper DenseNet 密集连接     |
| [Inception](Pre_Inception.md)     |      |


---
## Notes

| 题目 | 描述 |
| :--: | :--: |
| [PU-FLOW](P_PU-FLOW.md)     |   PU-FLOW开发文档   |
| [PU-FLOW-SurfaceReconstruction](P_PU-FLOW-SurfaceReconstruction.md)     |  PU-FLOW曲面重建开发文档    |
|      |      |
|      |      |
|      |      |

---
## Experimental Doc

| 题目 | 描述 |
| :--: | :--: |
| [NICE](E_NICE.md)     |  NICE实验文档    |
| [C++深度学习框架](E_Cpp4DL.md)     | C++实现深度学习框架     |
|      |      |

---
## Tools

| 题目 | 描述 |
| :--: | :--: |
|  [Pytorch](T_Pytorch.md)    |  Pytorch学习文档    |
|  [PytorchAPI](T_PytorchAPI.md)   | Pytorch常用API总结    |
|  [PytorchDebug](T_Pytorch_Debug.md)   | Pytorch常见Debug解决方案    |
|  [Docker-Pytorch+cuda+opengl](T_Docker-Pytorch+cuda+opengl.md)    |  Pytorch+cuda+opengl环境Docker搭建教程    |
|  [Docker](T_Docker.md)    |  Docker学习教程(GUI+VSCode+CPU)    |
|  [Git](T_Git.md)    |  Git学习文档(Typora)    |
|  [colab](T_Colab.md)    |   Colab使用教程   |
|  [Fish](T_Fish.md)    |  摸鱼    |
|  [GPU](T_GPU.md)    | GPU架构学习文档     |
|  [PCList](T_PCList.md)   |点云入门文章      |
|  [pypoisson](T_pypoisson.md)    | 泊松重建     |
|  [Ubuntu/Windows双系统](T_Ubuntu双系统安装.md)    | Ubuntu/Windows     |
| [Caffe](T_Caffe.md) | Caffe debug文档|
| [CMake](T_CMake.md)     |CMake 学习文档      |
| [深度学习模型部署](T_DLModel部署.md)    | 深度学习模型部署教程     |
| [Linux](T_Linux.md)     | Linux命令 + Vim使用教程 + 双系统安装  | [OS](T_OS.md)     | 操作系统学习文档     |
| [PaperVocabulary](T_PaperVocabulary.md)    |  计算机领域词汇表    |
|  [wechat_zhihu](T_wechat_zhihu.md)    | 微信知乎文章     |
|      |      |
|      |      |
|      |      |
|      |      |

---
## Class Notes

| 题目 | 描述 |
| :--: | :--: |
| [吴恩达-神经网络和深度学习-I](C_DL_W1.md)    | 神经网络和深度学习 专题笔记   |
| [吴恩达-改善深层神经网络-II](C_DL_W2.md)     | 改善深层神经网络 专题笔记     |
| [吴恩达-改善深层神经网络-III](C_DL_W3.md)     | 结构化机器学习项目 专题笔记      |
| [吴恩达-卷积神经网络-IV](C_DL_W4.md)     |  卷积神经网络 专题笔记    |
| [吴恩达-序列模型-V](C_DL_W4.md)     |  序列模型 专题笔记    |
| [深度学习全栈训练营](C_DLFullStack.md)     |  全栈深度学习训练营    |
| [李宏毅Flow-based Model](C_Flow.md)     | 李宏毅深度学习   |
| [深蓝学院概率图模型](C_PGM.md)     | 概率图模型    |
| [从0搭建深度学习框架](C_3DL.md)     | 从0搭建深度学习框架     |
| [深蓝学院概率论](C_ProbabilityTheory.md)     | 概率图模型    |
|      |      |


---
## Book Notes

| 题目 | 描述 |
| :--: | :--: |
| [C++ Primer Plus](B_CppPrimerPlus.md)     |   C++ Primer Plus 读书笔记   |
| [CUDA C编程权威指南](B_CUDA_C.md)     |  CUDA C编程权威指南 (高性能计算技术丛书)读书笔记    |
| [C++元模板编程:深度学习框架](B_Cpp4DL.md)     |C++元模板编程:深度学习框架      |
|      |      |
