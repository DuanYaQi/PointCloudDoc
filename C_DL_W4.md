# 深度学习工程师

由 deeplearning.ai 出品，网易引进的正版授权中文版深度学习工程师微专业课程，让你在了解丰富的人工智能应用案例的同时，学会在实践中搭建出最先进的神经网络模型，训练出属于你自己的 AI。



deeplearning.ai

https://www.coursera.org/learn/neural-networks-deep-learning?action=enroll

https://study.163.com/my#/smarts

https://www.bilibili.com/video/av66646276





**note**

https://redstonewill.blog.csdn.net/article/details/78651063

https://www.zhihu.com/column/DeepLearningNotebook

http://www.ai-start.com/dl2017/



**课后作业**

https://blog.csdn.net/u013733326/article/details/79827273

https://www.heywhale.com/mw/project/5e20243e2823a10036b542da





## Question

- [ ] 改善深层神经网络-[1.11 权重初始化](#winit)，





## 卷积神经网络

### 第一周 卷积神经网络

#### 1.1 计算机视觉

首先，计算机视觉的高速发展标志着新型应用产生的可能，这是几年前，人们所不敢想象的。通过学习使用这些工具，你也许能够创造出新的产品和应用。

其次，即使到头来你未能在计算机视觉上有所建树，也可以将**所学的知识应用到其他算法和结构**。



一张 64x64x3 的图片，神经网络输入层的维度为12288。一张 1000x1000x3 的图片，神经网络输入层的维度将达到 3M，使得网络权重 W 非常庞大。这样会造成两个后果，

- 一是神经网络结构复杂，数据量相对不够，容易出现过拟合；
- 二是所需内存、计算量较大。

解决这一问题的方法就是使用卷积神经网络（CNN）。



---

#### 1.2 边缘检测示例

神经网络由浅层到深层，分别可以检测出图片的边缘特征 、局部特征（例如眼睛、鼻子等）、整体面部轮廓。

![这里写图片描述](assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTI4MDkyMTM1NzA5)

最常检测的图片边缘有两类：一是垂直边缘（vertical edges），二是水平边缘（horizontal edges）。

![这里写图片描述](assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTI4MDkyODA1NTUz)

图片的边缘检测可以通过与相应**滤波器**进行卷积来实现。以垂直边缘检测为例，原始图片尺寸为6x6，滤波器filter尺寸为3x3，卷积后的图片尺寸为4x4，得到结果如下：

![这里写图片描述](assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTI4MTAwMzAwMDg1)

卷积过程动态示意图

![img](assets/v2-6428cf505ac1e9e1cf462e1ec8fe9a68_b.webp)



---

#### 1.3 更多边缘检测内容

还有很多其他的滤波器（检测算子）

![image-20210411155016785](assets/image-20210411155016785.png)

随着深度学习的发展，我们学习的其中一件事就是当你真正想去检测出复杂图像的边缘，你不一定要去使用那些研究者们所选择的这九个数字，但你可以从中获益匪浅。把这矩阵中的9个数字当成9个参数，并且在之后你可以学习使用**反向传播**算法，其目标就是去**理解这9个参数**。

![image-20210411155020970](assets/image-20210411155020970.png)

将这9个数字当成参数的思想，已经成为计算机视觉中最为有效的思想之一。



---

#### 1.4 Padding

**本小节步长全部为1**

valid convolution : no padding

输入 * 卷积核 → 输出维度   （n, n） * （f, f） →  （n - f + 1, n - f + 1）



same convolution: padding   一搬填充为0

输入 * 卷积核 → 输出维度   （n + 2p, n + 2p） * （f, f） →  （n  + 2p - f + 1, n  + 2p - f + 1）



如果希望输出维度和原始输入维度一样，则计算得
$$
p = \frac{f-1}{2}
$$
这里也诠释了为什么卷积核一般为奇数尺寸。并且奇数有中心像素点，便于索引滤波器的位置。

odd number 奇数

even number 偶数



---

#### 1.5 卷积步长

给定 padding : p 、strider: s

则有输入 * 卷积核 → 输出维度   （n, n） * （f, f） →  （$\frac{n+2p-f}{s}+1$, $\frac{n+2p-f}{s}+1$）

如果商不为整，向下取整 `floor`，有一部分超出范围就不进行计算。

 

相关系数（cross-correlations）与卷积（convolutions）之间是有区别的。真正的卷积运算（数学/信号处理）会先将filter绕其中心旋转180度，然后再将旋转后的filter在原始图片上进行滑动计算。filter旋转如下所示：

![这里写图片描述](assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTI4MjAwNDU1MjI2)

而在深度学习领域，默认不需要反转，直接求积。严格意义来讲我们**平时使用的方法不叫卷积而叫互相关**。

之所以可以这么等效，是因为**滤波器算子一般是水平或垂直对称的**，180度旋转影响不大；而且最终滤波器算子需要通过CNN网络梯度下降算法计算得到，**旋转部分可以看作是包含在CNN模型算法**中。总的来说，忽略旋转运算可以大大提高CNN网络运算速度，而且不影响模型性能。



---

#### 1.6 卷积为何有效





#### 1.7 单层卷积网络



#### 1.8 简单卷积网络示例





#### 1.9 池化层





#### 1.10 卷积神经网络示例





#### 1.11 为什么使用卷积？