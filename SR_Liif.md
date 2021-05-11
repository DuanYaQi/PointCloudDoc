# Learning Continuous Image Representation with Local Implicit Image Function

CVPR 2020 Oral https://yinboc.github.io/liif/



## Abstract

How to represent an image? While the visual world is presented in a continuous manner, machines store and see the images in a discrete way with 2D arrays of pixels. In this paper, we seek to learn a **continuous representation** for images. 如何表示图像？ 在以连续方式呈现视觉世界的同时，机器以 2D 像素矩阵的离散方式存储和查看图像。在本文中，我们试图学习图像的连续表示。

Inspired by the recent progress in 3D reconstruction with **implicit neural representation**, we propose **Local Implicit Image Function** (LIIF), which takes an **image coordinate** and the **2D deep features** around the coordinate as inputs, predicts the **RGB value** at a given coordinate as an output. Since the coordinates are continuous, LIIF can be presented in **arbitrary resolution**. To generate the continuous representation for images, we train an encoder with LIIF representation via a self-supervised task with super-resolution. The learned continuous representation can be presented in arbitrary resolution even extrapolate to ×30 higher resolution, where the training tasks are not provided. 

受到最近以隐式神经表示进行 3D 重建的进展的启发，我们提出了局部隐式图像函数（LIIF），该函数将图像坐标和坐标周围的二维深度特征作为输入，预测给定坐标下的 RGB 值作为输出。由于坐标是连续的，因此可以以任意分辨率呈现 LIIF。为了生成图像的连续表示，我们通过具有超分辨率的自监督任务训练具有 LIIF 表示的编码器。所学习的连续表示可以任意分辨率呈现，甚至可以外推到 30 倍的高分辨率，而无需提供训练任务。

We further show that LIIF representation builds a bridge between discrete and continuous representation in 2D, it naturally supports the learning tasks with size-varied image ground-truths and significantly outperforms the method with resizing the ground-truths. 我们进一步证明，LIIF表示法在2D离散表示法和连续表示法之间架起了一座桥梁，它自然支持大小可变的图像地面实况的学习任务，并且在调整地面实境大小方面明显优于该方法。



---

## 1. Introduction

Our visual world is continuous. However, when a ma-
chine tries to process a scene, it will usually need to first
store and represent the images as 2D arrays of pixels, where
thetrade-offbetweencomplexityandprecisioniscontrolled
by resolution. While the pixel-based representation has
been successfully applied in various computer vision tasks,
they are also constrained by the resolution. For example, a
dataset is often presented by images with different resolu-
tions. If we want to train a convolutional neural network,
we will usually need to resize the images to the same size,
which may sacrifice fidelity. Instead of representing an im-
age with a fixed resolution, we propose to study a contin-
uous representation for images. By modeling an image as
a function defined in a continuous domain, we can restore
and generate the image in arbitrary resolution if needed.

我们的视觉世界是连续的。 但是，当机器尝试处理场景时，通常需要先将图像存储和表示为2D像素阵列，其中复杂度和精度之间的权衡是由分辨率控制的。 尽管基于像素的表示已成功应用于各种计算机视觉任务中，但它们也受到分辨率的限制。 例如，数据集通常由具有不同分辨率的图像呈现。 如果要训练卷积神经网络，通常需要将图像调整为相同大小，这可能会牺牲保真度。 代替以固定的分辨率表示图像，我们建议研究图像的连续表示。 通过将图像建模为在连续域中定义的函数，我们可以根据需要恢复和生成任意分辨率的图像。



How do we represent an image as a continuous function? 我们如何将图像表示为连续函数？

Our work is inspired by the recent progress in implicit neural representation [34, 27, 6, 38, 18, 41] for 3D shape reconstruction. The key idea of implicit neural representation is to **represent an object** as **a function** that **maps coordinates** to the corresponding **signal** (e.g. **signed distance** to a **3D object surface**, **RGB value** in an image), where the function is parameterized by a deep neural network. 我们的工作受到3D形状重构的隐式神经表示[34，27，6，38，18，41]的最新进展的启发。 隐式神经表示的关键思想是**将对象表示为将坐标映射到相应信号的函数**（例如，到3D对象表面的有符号距离，图像中的RGB值），其中该函数由**深度神经网络进行参数化**。 

To share knowledge across instances instead of fitting individual functions for each object, **encoder-based** methods [27, 6, 41] are proposed to predict latent codes for different objects, then a decoding function is shared by all the objects while it takes the latent code as an additional input to the coordinates.   为了在实例之间共享知识，而不是为每个对象拟合单独的功能，提出了基于编码器的方法[27、6、41]来预测不同对象的潜在代码，然后所有解码器都将一个解码函数共享给潜在的潜在对象。 代码作为坐标的附加输入。

Despite its success in 3D tasks [38, 39], previous encoder-based methods of implicit neural representation only succeeded in representing simple images such as digits [6], but failed to represent natural images with high fidelity [41].  尽管它在3D任务中取得了成功[38，39]，但是以前基于编码器的隐式神经表示方法仅成功地表示了简单的图像，例如数字[6]，但未能表示出具有高保真度的自然图像[41]。



In this paper, we propose the Local Implicit Image Function (LIIF) for representing natural and complex images in a continuous manner. In LIIF, an image is represented as a set of latent codes distributed in spatial dimensions. Given a coordinate, the decoding function takes the coordinate information and queries the local latent codes around the coordinate as inputs, then predicts the RGB value at the given coordinate as an output. Since the coordinates are continuous, LIIF can be presented in arbitrary resolution.

在本文中，我们提出了局部隐式图像函数（LIIF），用于以连续方式表示自然图像和复杂图像。 在LIIF中，图像表示为在空间维度上分布的一组潜在代码。 给定一个坐标，解码功能将获取坐标信息，并查询该坐标周围的局部潜码作为输入，然后预测给定坐标处的RGB值作为输出。 由于坐标是连续的，因此可以以任意分辨率呈现LIIF。



