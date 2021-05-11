# Learning Continuous Image Representation with Local Implicit Image Function

CVPR 2020 Oral https://yinboc.github.io/liif/



## Abstract

How to represent an image? While the visual world is presented in a continuous manner, machines store and see the images in a discrete way with 2D arrays of pixels. In this paper, we seek to learn a **continuous representation** for images. 如何表示图像？ 在以连续方式呈现视觉世界的同时，机器以 2D 像素矩阵的离散方式存储和查看图像。在本文中，我们试图学习图像的连续表示。

Inspired by the recent progress in 3D reconstruction with **implicit neural representation**, we propose **Local Implicit Image Function** (LIIF), which takes an **image coordinate** and the **2D deep features** around the coordinate as inputs, predicts the **RGB value** at a given coordinate as an output. Since the coordinates are continuous, LIIF can be presented in **arbitrary resolution**. To generate the continuous representation for images, we train an encoder with LIIF representation via a self-supervised task with super-resolution. The learned continuous representation can be presented in arbitrary resolution even extrapolate to ×30 higher resolution, where the training tasks are not provided. 

受到最近以隐式神经表示进行 3D 重建的进展的启发，我们提出了局部隐式图像函数（LIIF），该函数将图像坐标和坐标周围的二维深度特征作为输入，预测给定坐标下的 RGB 值作为输出。由于坐标是连续的，因此可以以任意分辨率呈现 LIIF。为了生成图像的连续表示，我们通过具有超分辨率的自监督任务训练具有 LIIF 表示的编码器。所学习的连续表示可以任意分辨率呈现，甚至可以外推到 30 倍的高分辨率，而无需提供训练任务。

We further show that LIIF representation builds a bridge between discrete and continuous representation in 2D, it naturally supports the learning tasks with size-varied image ground-truths and significantly outperforms the method with resizing the ground-truths. 我们进一步证明，LIIF表示法在2D离散表示法和连续表示法之间架起了一座桥梁，它自然支持大小可变的图像地面实况的学习任务，并且在调整地面实境大小方面明显优于该方法。





