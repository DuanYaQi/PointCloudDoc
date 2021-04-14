# 深度学习工程师

由 deeplearning.ai 出品，网易引进的正版授权中文版深度学习工程师微专业课程，让你在了解丰富的人工智能应用案例的同时，学会在实践中搭建出最先进的神经网络模型，训练出属于你自己的 AI。



**CNN可视化**

https://poloclub.github.io/cnn-explainer/





deeplearning.ai

https://www.coursera.org/learn/neural-networks-deep-learning?action=enroll

https://study.163.com/my#/smarts

https://www.bilibili.com/video/av66646276





**note**

https://redstonewill.blog.csdn.net/article/details/78769236

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

#### 1.6 三维卷积

![这里写图片描述](assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTI4MjAzODUzNTI2)

对于3通道的RGB图片，其对应的滤波器算子同样也是3通道的。例如一个图片是6 x 6 x 3，分别表示图片的高度（height）、宽度（weight）和通道（channel）。

过程是将每个单通道（R，G，B）与对应的filter进行卷积运算求和，然后再将**3通道的和相加**，得到输出图片的一个像素值。

**不同通道的滤波算子可以不相同**。例如**R**通道filter实现**垂直**边缘检测，**G和B**通道**不进行**边缘检测，全部置零，或者将R，G，B三通道filter全部设置为水平边缘检测。



为了进行多个卷积运算，实现更多边缘检测，可以增加更多的滤波器组。例如设置第一个滤波器组实现垂直边缘检测，第二个滤波器组实现水平边缘检测。这样，不同滤波器组卷积得到不同的输出，个数由滤波器组决定。

![这里写图片描述](assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTI4MjMwNzA0Mjk3)



则有输入 * 卷积核 → 输出维度   
$$
（n, n, n_c） * n_k（f, f, n_c） →  （\frac{n+2p-f}{s}+1, \frac{n+2p-f}{s}+1, n_k）
$$
$n$ 为图片尺寸大小，也可用 h 和 w 表示高和宽，例中为 6 ；

$f$ 为卷积核尺寸大小，例中为 3 ；

$n_c$ 为图片通道数目，例中为2；

$n_k$ 为滤波器组个数，例中为2；

padding : p 为填充数，例中为0，无填充；

strider: s 为步长，例中为1。则有：
$$
（6, 6, 3） * 2（3, 3, 3） →  （4, 4, 2）
$$



---

#### 1.7 单层卷积网络

![这里写图片描述](assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTI4MjMzMjQyNDk0)

相比之前的卷积过程，CNN 的单层结构多了激活函数 ReLU 和偏移量b。整个过程与标准的神经网络单层结构非常类似：
$$
Z^{[l]}=W^{[l]} A^{[l-1]}+b\\
A^{[l]}=g^{[l]}\left(Z^{[l]}\right)
$$
输出 = 非线性激活函数（线性函数+偏差），例中 W 为卷积核，A 为输入图像，b 为偏置，g为 `relu` 。相当于有两个个待学习的参数：**卷积核，偏差**



总结CNN单层结构的所有标记符号，设层数为 $l$。

**filter size:** $f^{[l]}$   				 滤波器尺寸

**padding:** $p^{[l]}$      				 填充

**stride:** $s^{[l]}$          				  步长

**number of filters:** $n_{c}^{[l]}$  	滤波器个数



**input:** $n_{H}^{[l-1]} \times n_{\omega}^{[l-1]} \times n_{c}^{[l-1]} $     输入维度，l-1层

**output:**  $n_{H}^{[l]} \times n_{\omega}^{[l]} \times n_{c}^{[l]}$             输出维度，l层

其中  
$$
\left.\begin{array}{l}
n_{H}^{[l]}=\left[\frac{n_{H}^{[l-1]}+2 p^{[l]}-f^{[l]}}{s^{[l]}}+1\right. \\
n_{W}^{[l]}=\left\lfloor\frac{n_{W}^{[l-1]}+2 p^{[l]}-f^{[l]}}{s^{[l]}}+1\right.
\end{array}\right]
$$




**filter:** $f^{[l]} \times f^{[l]} \times n_{c}^{[l-1] }$               			滤波器维度，最后一维与输入channel相同

**weights:** $f^{[l]} \times f^{[l]} \times n_{c}^{[l-1]} \times n_{c}^{[l]}$ 		  权重维度 = 滤波器维度 * 滤波器个数 

**bias:** $1 \times 1 \times 1 \times n_{c}^{[l]}$								偏置维度，只与滤波器个数有关

**activations:** $n_{H}^{[l]} \times n_{W}^{[l]} \times n_{c}^{[l]}$                  激活函数维度，与输出维度完全相同



如果**mini-batch**有m个样本，进行向量化运算，相应的输出维度为 $m \times n_{H}^{[l]} \times n_{\omega}^{[l]} \times n_{c}^{[l]}$



假设你有10个过滤器，神经网络的一层是3×3×3，那么，这一层有多少个参数呢？

我们来计算一下，每一层都是一个3×3×3的矩阵，因此每个过滤器有27个参数，也就是27个数。然后加上一个偏差，用参数表示，现在参数增加到28个现在我们有10个过滤器，加在一起是28×10，也就是280个参数。



---

#### 1.8 简单卷积网络示例

![这里写图片描述](assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTMwMTA1NTQ2MzIx)

CNN模型各层结构如上图所示。需要注意的是，$a^{[3]}$ 的维度是 `7 x 7 x 40`，将 $a^{[3]}$ 排列成 1 列，维度为 `1960 x 1`，然后连接最后一级输出层。输出层可以是一个神经元，即二元分类（logistic）；也可以是多个神经元，即多元分类（softmax）。最后得到预测输出 $\hat y$ 。值得一提的是，随着CNN层数增加，$n_H^{[l]}$  和 $n_W^{[l]}$ 一般逐渐减小，而$n_c^{[l]}$ 一般逐渐增大。

CNN有三种类型的layer：

- Convolution层（CONV）
- Pooling层（POOL）
- Fully connected层（FC）



---

#### 1.9 池化层

缩减模型大小，提高计算速度，提高所提取特征的鲁棒性。

**最大池化**

![这里写图片描述](assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTI5MjAyMzIyMzA2)

这就像是应用了一个规模为2的过滤器，因为我们选用的是2×2区域，步幅是2，这些就是最大池化的超参数。

![image-20210412151328057](assets/image-20210412151328057.png)

![image-20210412151343259](assets/image-20210412151343259.png)

**数字大意味着可能探测到了某些特定的特征**，左上象限具有的特征可能是一个垂直边缘，一只眼睛。显然左上象限中存在这个特征，这个特征可能是一只猫眼探测器。必须承认，使用最大池化的主要原因是此方法在很多实验中效果都很好。计算卷积层输出大小的公式同样适用于最大池化，即 $\frac{n+2p-f}{s}+1$



**平均池化**

选取的不是每个过滤器的最大值，而是平均值。

![这里写图片描述](assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTI5MjAzODQxMzM2)

目前来说，最大池化比平均池化更常用。但也有例外，就是深度很深的神经网络，可以用平均池化来分解规模为7×7×1000的网络的表示层，在整个空间内求平均值，得到1×1×1000。



**参数**

池化的超级参数包括过滤器大小 $f$ 和步幅 $s$ ,常用的参数值为 $f=2, s=2$，效果相当于高度和宽度缩减一半。很少用到超参数 padding。最常用的 padding 值是0，输入为$n_{H} \times n_{W} \times n_{c}$，输出为$ \lfloor\frac{n_{H}-f}{s}+1\rfloor \times\lfloor\frac{n_{\mathrm{w}}-f}{s}+ 1\rfloor \times n_{c}$。

输入与输出**通道数相同**，因为我们对每个通道都做了池化。需要注意的一点是，池化过程中**没有需要学习的参数**。执行反向传播时，反向传播没有参数适用于最大池化。这些设置过的超参数，可能是手动设置的，也可能是通过交叉验证设置的。



---

#### 1.10 卷积神经网络示例

计算网络层数，通常是带有权重和参数的才算一层，想池化层和激活层就不算单独的一层。

![这里写图片描述](assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMTMwMTAyNTMwNTQ0)

图中，`CON` 层后面紧接一个 `POOL` 层，`CONV1` 和 `POOL1` 构成第一层，`CONV2` 和 `POOL2` 构成第二层。特别注意的是 `FC3` 和 `FC4` 为全连接层 `FC`，它跟标准的神经网络结构一致。最后的输出层（softmax）由10个神经元构成。

随着神经网络深度的加深，**高度和宽度都会减小**，从 32×32 到 28×28，到 14×14，到 10×10，再到 5×5；**而通道数量会增加**，从 3 到 6 到 16 不断增加，然后得到一个全连接层。



接下来我们讲讲神经网络的**激活值形状**，**激活值大小**和**参数数量**。有几点要注意，第一，池化层没有参数；第二，**卷积层的参数相对较少**，第三，**全连接层参数较多**，其实许多参数都存在于神经网络的全连接层。观察可发现，**随着神经网络的加深，激活值size会逐渐变小，如果激活值size下降太快，也会影响神经网络性能**。

整个网络各层的尺寸和参数如下表格所示：

![在这里插入图片描述](assets/20200208104004691.png)

`CONV1` 参数$ = 6 \times （5\times5\times3+1）$

`CONV2` 参数$ = 16 \times （5\times5\times6+1）$

`FC2` 参数$ = （400\times120）+120$

`FC3` 参数$ = （120\times84）+84$

`softmax` 参数$ = （84\times10）+10$

上边的conv算法是每个通道滤波器做的事情不一样，如果每个通道滤波器一致，则参数量要除以通道数。



常规做法是，**尽量不要自己设置超参数**，而是**查看文献中别人采用了哪些超参数**，**选一个在别人任务中效果很好的架构**，那么它也有可能适用于你自己的应用程序。



----

#### 1.11 为什么使用卷积？

与全连接层相比，卷积层的两个主要优势在于

**参数共享**：一个特征检测器（例如垂直边缘检测）对图片A的某块区域有用，同时也可能作用在图片A的其它区域。即不用对图像不同区域使用不同的滤波器，所有区域只使用一个滤波器处理即可。（如果不参数共享，滤波器滑一次变一次，参数过多）

**稀疏连接**：因为滤波器算子尺寸限制，每一层的每个输出只与输入**部分区域**内有关。而且其它像素值都不会对输出产生任影响。

**平移不变性**：神经网络的卷积结构使得即使移动几个像素，这张图片依然具有非常相似的特征，应该属于同样的输出标记。



除此之外，由于 CNN 参数数目较小，所需的训练样本就相对较少，从而一定程度上不容易发生过拟合现象。而且，CNN比较擅长捕捉区域位置偏移。也就是说CNN进行物体检测时，不太受物体所处图片位置的影响，增加检测的准确性和系统的健壮性。



---

#### 1.12 interview yann-lecun

lenet

Conditional Random Field 条件随机场

给开源社区做贡献



-----

### 第二周 深度卷积网络：实例探究

#### 2.1 为什么要进行实例探究？

典型的CNN模型包括：

- **LeNet-5**-1998 LeCun
- **AlexNet**-2012年ImageNet竞赛冠军获得者Hinton和他的学生Alex Krizhevsky设计
- **VGG**-2014年ILSVRC竞赛的第二名，第一名是GoogLeNet。

除了这些性能良好的CNN模型之外，Residual Network（ResNet）的特点是可以构建很深很深的神经网络（目前最深的好像有152层）。Inception Neural Network。



---

#### 2.2 经典网络

**LeNet-5**

![这里写图片描述](assets/20171211150034018)

5 层，6W 个参数，平均池化，sigmoid 和 tanh 函数，直接输出类别

> Gradient-Based Learning Applied to Document Recognition



**AlexNet**

![这里写图片描述](assets/20171211155720094)

60M 个参数，最大池化，relu 函数，局部响应归一化层 Local Response Normalization，softmax 输出各类别概率

> ImageNet Classification with Deep Convolutional Neural Networks



**VGG-16**

![这里写图片描述](assets/20171211175203203)

16 层，138M 个参数，只用 3x3 same 卷积，最大池化，专注于构建卷积层的简单网络，简化了神经网络架构

随着网络的加深，**图像缩小的比例和通道数增加的比例是有规律的**。很吸引人。

> very deep convolutional networks for large-scale image recognition
>
> Visual Geometry Group Network - 视觉几何群网络





---

#### 2.3 残差网络

Residual Networks (ResNets)

非常深的网络会出现梯度消失和梯度爆炸问题

skip connection / short cut

> Deep Residual Learning for Image Recognition

**残差块**

![这里写图片描述](assets/20171211204756960)

上图中红色部分就是skip connection，直接建立 $a[l]$ 与 $a[l+2]$ 之间的隔层联系。相应的表达式如下：
$$
\begin{equation}
z^{[l+1]}=W^{[l+1]} a^{[l]}+b^{[l+1]}\\
a^{[l+1]}=g\left(z^{[l+1]}\right)\\
z^{[l+2]}=W^{[l+2]}a^{[l+1]}+b^{[l+2]}\\
a^{[l+2]}=g\left(z^{[l+2]}+a^{[l]}\right)
\end{equation}
$$


$a[l]$ 直接隔层与下一层的线性输出相连，与 $z[l+2]$ 共同通过激活函数（ReLU）输出$a[l+2]$。

![image-20210414155426103](assets/image-20210414155426103.png)

由多个 Residual block 组成的神经网络就是 Residual Network。实验表明，这种模型结构对于训练非常深的神经网络，效果很好。Residual Network的结构如下图所示。

![这里写图片描述](assets/20171211211417392)

另外，为了便于区分，我们把非 Residual Networks 称为 Plain Network。与 Plain Network 相比，Residual Network 能够训练更深层的神经网络，有效避免发生发生梯度消失和梯度爆炸。从下面两张图的对比中可以看出，随着神经网络层数增加，Plain Network 实际性能会变差，training error 甚至会变大。然而，Residual Network 的训练效果却很好，training error 一直呈下降趋势。

![这里写图片描述](assets/20171211213835572)



---

#### 2.4 残差网络为什么有用？

网络在训练集上表现好，才能在验证集和测试集上表现好。

![这里写图片描述](assets/20171211215418919)

如上图所示，输入 $x$ 经过很多层神经网络后输出 $a^{[l]}$ ，$a^{[l]}$ 经过一个 Residual block 输出$a^{[l+2]}$。$a^{[l+2]}$ 的表达式为：

$$
\begin{equation}a^{[l+2]}=g\left(z^{[l+2]}+a^{[l]}\right)=g\left(W^{[l+2]} a^{[l+1]}+b^{[l+2]}+a^{[l]}\right)\end{equation}
$$
输入 $x$ 经过Big NN后，若 $W^{[l+2]}≈0$ ，$b^{[l+2]}≈0$，则有：
$$
\begin{equation}a^{[l+2]}=g\left(a^{[l]}\right)=\operatorname{ReL} U\left(a^{[l]}\right)=a^{[l]} \quad when \quad a^{[l]} \geq 0\end{equation}
$$
可以看出，即使发生了梯度消失，$W^{[l+2]}≈0$ ，$b^{[l+2]}≈0$，也能直接建立 $a^{[l+2]}$ 与 $a^{[l]}$ 的线性关系，且 $a^{[l+2]}=a^{[l]}$，这其实就是 Identity function。$a^{[l]}$ 直接连到 $a^{[l+2]}$，从效果来说，**相当于直接忽略了$a^{[l]}$之后的这两层神经层**。

这样，看似很深的神经网络，其实由于许多 Residual blocks 的存在，**弱化削减了某些神经层之间的联系**，实现**隔层线性传递**，而**不是一味追求非线性关系**，模型本身也就能“容忍”更深层的神经网络了。而且从性能上来说，这两层额外的 Residual blocks 也不会降低 Big NN的性能。当然，如果 Residual blocks **确实能训练得到非线性关系**，那么也**会忽略 short cut**，跟 Plain Network 起到同样的效果。



有一点需要注意的是，ResNet中使用了**same卷积**，使得$a^{[l]}$ 和 $a^{[l+2]}$ 的**维度相同**；但如果 Residual blocks 中 $a^{[l]}$ 和 $a^{[l+2]}$ 的**维度不同**，通常可以引入矩阵$W_s$，与 $a^{[l]}$ 相乘，使得 $W_s*a^{[l]}$ 的维度与 $a^{[l+2]}$ 一致。参数矩阵 $W_s$ 有来两种方法得到：一种是将 $W_s$ **作为学习参数**，通过模型训练得到；另一种是固定 $W_s$ 值（类似单位矩阵），不需要训练，$W_s$ 与 $a^{[l]}$ 的乘积仅仅使得 $a^{[l]}$ **截断或者补零**。这两种方法都可行。



**ResNet 结构**![image-20210414163056395](assets/image-20210414163056395.png)

上图为普通的网络，输入image，多个卷积层，最后输出一个Softmax。只需要添加 skip connection，就转换为 ResNet，如下图：

![这里写图片描述](assets/20171212142205247)

ResNets 同类型层之间，例如 CONV layers(实线)，大多使用 Same 类型，保持维度相同。如果是不同类型层之间的连接，例如 CONV layer 与 POOL layer 之间(虚线)，如果维度不同，则引入矩阵 $W_s$。





> $W^{[l+2]}$, $b^{[l+2]}$ 就相当于一个开关，只要让这两个参数 w 和 b 为 0，就可以达到增加网络深度却不影响网络性能的目的。而**是否把这两个参数置为 0 就要看反向传播**，网络最终能够知道到底要不要skip。
>
> 
>
> 何凯明说过其实 AlexNet 是解决太深导致梯度消失，因为随着层数的增加计算出的梯度  会慢慢变小，可以映射到激活值会慢慢变小，之后他就想加一个 skip connection就可以保证无论中间多少层，最终两端激活值大体上不改变。
>
> 
>
> 当网络不断加深时，就算是选用学习恒等函数的参数都很困难，所以很多层最后的表现不但没有更好，反而更糟。
>
> 
>
> 这保证了深度的增加不会给模型带来负面影响，**至少不会比 Plain Network 差**。**残差块很容易学习恒等映射**



---

#### 2.5 Network in network 以及 1×1 卷积

池化层可以压缩高度和宽度。1x1卷积可以压缩通道数。

1x1卷积，处理多通道的数据。即不同通道的数据加权求和，然后通过非线性函数。



对于单个 filter，1x1 的维度，意味着卷积操作等同于乘积操作。对于多个filters，1x1 Convolutions 的作用实际上**类似全连接层**的神经网络结构。效果等同于 Plain Network 中 $a^{[l]}$ 到 $a^{[l+1]}$ 的过程。

![这里写图片描述](assets/20171212144647936)

1x1 Convolutions 可以用来缩减输入图片的通道数目:

![这里写图片描述](assets/20171212145859683)

> Network in network, lin, 2013.



---

#### 2.6 Inception network motivation

构建卷积层时，你要决定过滤器的大小究竟是1×1，3×3 还是 5×5，或者要不要添加池化层。而Inception网络的作用就是**代替你来决定**，虽然网络架构因此变得更加复杂，但网络表现却非常好。

​	





---

#### 2.7 Inception 网络



---

#### 2.8 Mobilenet





---

#### 2.9 Mobilenet架构



---

#### 2.10 EfficientNet



---

#### 2.11 使用开源的实现方案





---

#### 2.12 迁移学习



---

#### 2.13 数据扩充



---

#### 2.14 计算机视觉现状



---