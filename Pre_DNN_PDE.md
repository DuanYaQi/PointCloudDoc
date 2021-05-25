# Deep Neural Networks Motivated by Partial Differential Equations

Lars Ruthotto and Eldad Haber



## Abstract

偏微分方程（PDE）对于建模许多物理现象必不可少，并且通常用于求解图像处理任务。 在后一领域中，基于PDE的方法将图像数据解释为**多元函数的离散化**，而将图像处理算法的输出解释为某些PDE的解决方案。将图像处理问题摆在无穷大的尺寸设置中，为分析和解决问题提供了强大的工具。 在过去的几十年中，通过PDE镜头对经典图像处理问题的重新解释已经创建了多种著名的方法，这些方法使许多领域受益，包括图像分割，去噪，配准和重建。

在本文中，我们为一类深度卷积神经网络（CNN）建立了新的PDE解释，该网络通常用于从语音，图像和视频数据中学习。我们的解释包括卷积残差神经网络（ResNet），这是用于诸如图像分类等任务的最有前途的方法之一，这些方法改善了在著名基准挑战中的最新性能。尽管最近获得了成功，但深层ResNet仍然面临与其设计相关的一些关键挑战，巨大的计算成本和内存要求以及对推理的理解不足。

在行之有效的PDE理论的指导下，我们得出了三个新的ResNet架构，它们分为两个新类别：抛物线型和双曲线型CNN。我们演示了PDE理论如何为深度学习提供新的见解和算法，并使用数值实验演示了三种新的CNN架构的竞争力。



## 1. Introduction

在过去的三十年中，受偏微分方程（PDE）启发的算法对涉及语音，图像和视频数据的许多处理任务产生了深远的影响。  适应传统上在物理学中用于执行图像处理任务的PDE模型已经做出了开创性的贡献。 开创性工作的不完整清单包括用于运动估计的光流模型[26]，用于图像滤波的非线性扩散模型[38]，用于图像分割的变分方法[36、1、8]和非线性保留边缘的去噪[42]。  。

基于PDE的数据处理的标准步骤是**解释所涉及的数据作为多元函数的离散化**。因此，可以将对数据的许多操作建模为作用于基础功能的PDE运算符的离散化。这种连续的数据模型已经为通过利用PDE和变分演算的丰富结果而获得的经典数据处理任务提供了扎实的数学理论（例如[43]）。连续的观点还使更多独立于实际分辨率的抽象公式成为可能，这些公式已被用来获得有效的多尺度和多层次算法（例如[34]）。

在本文中，我们建立了以语音，图像和视频数据为特征的深度学习任务的新PDE解释。 深度学习是机器学习的一种形式，它使用具有许多隐藏层的神经网络[4，31]。 尽管神经网络至少可以追溯到1950年代[41]，但几年前当深度神经网络（DNN）在语音识别[39]和图像分类[25]方面优于其他机器学习方法时，神经网络的普及率猛增。 深度学习还极大地改善了计算机视觉，例如在图像识别方面超过了人类[25，29，31]。 这些结果点燃了该领域最近的研究热点。 为了获得PDE解释，我们使用图像的连续表示并通过[19，15]扩展最近的工作，这些工作将一般数据类型的深度学习问题与常微分方程（ODE）联系起来。

深度神经网络使用几层来过滤输入特征，这些层的操作由逐个元素的非线性和仿射变换组成。 卷积神经网络（CNN）[30]的主要思想是将仿射变换基于具有紧密支持的滤波器的卷积算子。 监督学习旨在从训练数据中学习过滤器和其他参数（也称为权重）。  CNN被广泛用于解决大规模学习任务，涉及包含代表连续函数离散化的数据的数据，例如语音，图像和视频[29、30、32]。 通过设计，每个CNN层都利用图像信息之间的局部关系，从而简化了计算[39]。

尽管取得了巨大的成功，但深层的CNN仍然面临着严峻的挑战，包括设计对实际学习任务有效的CNN体系结构，这需要许多选择。除了层的数量（也称为网络深度）之外，重要的方面是每层卷积滤波器的数量（也称为层的宽度）以及这些滤波器之间的连接。 最近的趋势是倾向于在广域网中进行深度扩展，以期提高泛化程度（即CNN在在训练期间未使用的新示例上的性能）[31]。 另一个关键挑战是设计层，即选择仿射变换和非线性的组合。 一种实用但昂贵的方法是将架构的深度，宽度和其他属性视为超参数，并通过网络权重共同推断它们[23]。 我们将CNN架构解释为离散的PDE，为指导设计过程提供了新的数学理论。 简而言之，我们通过使用适当的时间积分方法离散化基础PDE来获得体系结构。

除了大量的培训费用外，深层的CNN在其可解释性和鲁棒性方面还面临着根本性的挑战。 特别是，用于关键任务（例如无人驾驶汽车）的CNN面临着“可解释”的挑战。 在非线性PDE理论中运用学习任务，可以使我们更好地理解此类网络的性质。 我们相信，对此处介绍的数学结构进行的进一步研究将使人们对网络有更深入的了解，并将缩小深度学习与依赖非线性PDE（例如流体动力学）的更成熟领域之间的差距。 在研究例如对抗性例子时，可以观察到我们方法的直接影响。 最近的工作[37]表明，由深度网络获得的预测可能对输入图像的扰动非常敏感。
   这些发现促使我们倾向于使用稳定的网络，即其输出对于输入特征的小扰动具有鲁棒性的网络，类似于PDE分析所建议的。

在本文中，我们考虑了残差神经网络（ResNet）[22]，这是一种非常有效的神经网络。 我们表明，**残差CNN**可以解释为**时空微分方程的离散化**。我们使用这个连接来分析网络的**稳定性**，并激发与众所周知的PDE具有相似性的新网络模型。 使用我们的框架，我们提出了三种新的体系结构。首先，我们介绍了**抛物线型CNN**，其将前向传播限制为使图像特征平滑并与各向异性滤波具有相似性的动力学[38，45，12]。 其次，我们提出了受汉密尔顿系统启发的**双曲CNN**，最后提出了第三，**二阶双曲CNN**。 可以预料，这些网络具有不同的属性。 例如，双曲线CNN大致保留了系统中的能量，这使它们与使图像数据变得平滑的抛物线网络区分开来，从而降低了能量。 在计算上，可以利用双曲线正向传播的结构来减轻内存负担，因为可以使双曲线动力学在连续和离散级别上都是可逆的。 这里建议的方法与可逆ResNets [16，9]密切相关。



## 2. Residual Networks and Differential Equations

机器学习的抽象目标是找到一个函数 $ f: \mathbb{R}^{n} \times \mathbb{R}^{p} \rightarrow \mathbb{R}^{m} $，以便 $ f(\cdot, \boldsymbol{\theta}) $ 准确预测观察到的现象的结果（例如，图像的类别，话语等）。 该函数通过使用示例训练的权重向量 $ \boldsymbol{\theta} \in \mathbb{R}^{p} $ 进行参数化。 在监督学习中，一组输入特征 $ \mathbf{y}_{1}, \ldots, \mathbf{y}_{s} \in \mathbb{R}^{n} $ 和输出标签 $ \mathbf{c}_{1}, \ldots, \mathbf{c}_{s} \in \mathbb{R}^{m} $ 可用，并用于训练模型 $ f(\cdot, \boldsymbol{\theta}) $。 输出标签是向量，其分量对应于属于给定类别的特定示例的估计概率。 例如，考虑图1中的图像分类结果，其中使用条形图将预测标签和实际标签可视化。为简洁起见，我们用 $ \mathbf{Y}=\left[\mathbf{y}_{1}, \mathbf{y}_{2}, \ldots, \mathbf{y}_{s}\right] \in \mathbb{R}^{n \times s} $ 和 $ \mathbf{C}=\left[\mathbf{c}_{1}, \mathbf{c}_{2}, \ldots, \mathbf{c}_{s}\right] \in \mathbb{R}^{m \times s} $ 表示训练数据

在深度学习中，函数 $f$ 由称为隐层的非线性函数组成。 每一层都由**仿射线性变换**和**逐点非线性**组成，旨在以一种能够学习的方式过滤输入特征。  作为一个相当笼统的公式，我们考虑[22]中使用的层的扩展版本，该层按如下方式过滤特征 $\mathbf{Y}$ <span id="eq1"></span>
$$
\begin{equation}
 \mathbf{F}(\boldsymbol{\theta}, \mathbf{Y})=\mathbf{K}_{2}\left(\boldsymbol{\theta}^{(3)}\right) \sigma\left(\mathcal{N}\left(\mathbf{K}_{1}\left(\boldsymbol{\theta}^{(1)}\right) \mathbf{Y}, \boldsymbol{\theta}^{(2)}\right)\right) 
\end{equation}\tag{1}
$$
在此，参数向量 $\boldsymbol{\theta}$ 分为三个部分，其中 $\boldsymbol{\theta}^{(1)}$ 和 $\boldsymbol{\theta}^{(3)}$ 参数化线性算子 $ \mathbf{K}_{1}(\cdot) \in \mathbb{R}^{k \times n}$ 并且 $ \mathbf{K}_{2}(\cdot) \in \mathbb{R}^{k_{\text {out }} \times k} $，  $\boldsymbol{\theta}^{(2)}$ 是归一化层 $\mathcal{N}$ 的参数。激活函数 $ \sigma: \mathbb{R} \rightarrow \mathbb{R} $ 逐分量地应用。 常见示例为 $ \sigma(x)=\tanh (x) $ 或定义为 $ \sigma(x)=\max (0, x) $ 的整流线性单位（ReLU）。 深度神经网络可以通过将（1）中给出的许多层进行串联来编写。

当处理图像数据时，通常将特征分组到不同的通道中（例如，对于RGB图像数据，存在三个通道）并将运算符 $ \mathbf{K}_{1} $ 和 $ \mathbf{K}_{2} $ 定义为由**空间卷积组成的块矩阵**。通常，将输出图像的每个通道计算为每个卷积输入通道的加权和。 举一个例子，假设 $ \mathbf{K}_{1} $ 具有三个输入通道和两个输出通道，并用 $ \mathbf{K}_{1}^{(\cdot, \cdot)}(\cdot) $ 是标准卷积算子[21]。 在这种情况下，我们可以将 $ \mathbf{K}_{1} $ 写成
$$
\begin{equation}
 \mathbf{K}_{1}(\boldsymbol{\theta})=\left(\begin{array}{lll}\mathbf{K}_{1}^{(1,1)}\left(\boldsymbol{\theta}^{(1,1)}\right) & \mathbf{K}_{1}^{(1,2)}\left(\boldsymbol{\theta}^{(1,2)}\right) & \mathbf{K}_{1}^{(1,3)}\left(\boldsymbol{\theta}^{(1,3)}\right) \\ \mathbf{K}_{1}^{(2,1)}\left(\boldsymbol{\theta}^{(1,2)}\right) & \mathbf{K}_{1}^{(2,2)}\left(\boldsymbol{\theta}^{(2,2)}\right) & \mathbf{K}_{1}^{(2,3)}\left(\boldsymbol{\theta}^{(2,3)}\right)\end{array}\right) 
\end{equation}\tag{2}
$$
其中 $ \boldsymbol{\theta}^{(i, j)} $ 表示第 $ (i, j) $ 个卷积算子的核参数。<span id="eq2"></span>共六个卷积核。

（1）中 $\mathcal{N}$ 的常见选择是批处理归一化层[27]。该层计算输入图像中每个通道在空间维度和示例上的经验均值和标准差，并使用此信息对输出图像的统计量进行归一化。 尽管不同示例的结合是违反直觉的，但其使用却得到了广泛应用，并且受到经验证据的启发，这些证据表明训练算法的收敛速度更快。 权重 $\boldsymbol{\theta}^{(2)}$ 表示在**归一化之后应用的每个输出通道的缩放因子和偏差**（即应用于该通道中所有像素的恒定位移）

ResNets 最近在几个基准中改进了最新技术，包括关于图像分类的计算机视觉竞赛。给定输入特征 $ \mathbf{Y}_{0}=\mathbf{Y} $，具有 $N$ 层的 ResNet 单元将生成过滤后的版本 $\mathbf{Y}_{N}$，如下所示<span id="eq3"></span>
$$
\begin{equation}
 \mathbf{Y}_{j+1}=\mathbf{Y}_{j}+\mathbf{F}\left(\boldsymbol{\theta}^{(j)}, \mathbf{Y}_{j}\right) \quad \text{for}\quad j=0,1, \ldots, N-1 
\end{equation}\tag{3}
$$
其中 $\boldsymbol{\theta}^{(j)}$ 是第 $j$ 层的权重（卷积模板W和偏差b）。为了强调该过程对权重的依赖性，我们表示为 $ \mathbf{Y}_{N}(\boldsymbol{\theta}) $。

注意，在 ResNets 单元的所有层上，特征向量的维数（即图像分辨率和通道数）是相同的，这在许多实际应用中是有限的。 因此，深层 CNN 的实现包含 ResNet 单元与其他层的连接，这些层可以更改，例如，通道数和图像分辨率。

在图像识别中，目标是使用例如由全连接层建模的线性分类器（即，具有**密集矩阵的仿射变换**）来对（3）的输出 $ \mathbf{Y}_{N}(\boldsymbol{\theta}) $ 进行分类。为了避免与 ResNet 单位混淆，我们将这些转换表示为 $ \mathbf{W} \mathbf{Y}_{N}(\boldsymbol{\theta})+\left(\mathbf{B}_{W} \boldsymbol{\mu}\right) \mathbf{e}_{s}^{\top} $，其中 $\mathbf{B}_{W}$ 的列表示分布偏差，而 $ \mathbf{e}_{s} \in \mathbb{R}^{s} $ 是所有值的向量。网络和分类器的参数未知，必须学习。 因此，学习的目标是通过近似解决优化问题来估计网络参数 $\boldsymbol{\theta}$ 和分类器的权重 $ \mathbf{W}, \boldsymbol{\mu} $。<span id="eq4"></span>
$$
\begin{equation}
 \min _{\boldsymbol{\theta}, \mathbf{W}, \boldsymbol{\mu}} \frac{1}{2} S\left(\mathbf{W} \mathbf{Y}_{N}(\boldsymbol{\theta})+\left(\mathbf{B}_{W} \boldsymbol{\mu}\right) \mathbf{e}_{s}^{\top}, \mathbf{C}\right)+R(\boldsymbol{\theta}, \mathbf{W}, \boldsymbol{\mu}) 
\end{equation}\tag{4}
$$
其中 $S$ 是一个损失函数，在第一个参数中是凸的，而 $R$ 是下面讨论的凸正则化器。 损失函数的典型示例是回归和对数回归中的最小二乘函数或分类中的交叉熵函数[17]。

由于以下几个原因，（4）中的优化问题具有挑战性。首先，这是一个**高维且非凸**的优化问题。 因此，必须满足于**局部最小值**。其次，每个示例的计算成本很高，并且示例数量很大。 第三，非常深的体系结构容易出现诸如梯度消失和爆炸[5]的问题，这些问题可能在离散前向传播不稳定[19]时出现。



---

### 2.1. Residual Networks and ODEs

我们推导出了在[19]中 ResNets 所提供的过滤的连续解释。在[15，10]中也有类似的观察。[Eq. 3](#eq3) 中的网格可以看作是前向的欧拉离散化(固定步长 $ \delta_{t}=1 $ )的初值问题<span id="eq5"></span>
$$
\begin{equation}
 \begin{aligned} \partial_{t} \mathbf{Y}(\boldsymbol{\theta}, t) &=\mathbf{F}(\boldsymbol{\theta}(t), \mathbf{Y}(t)), \text { for } t \in(0, T] \\ \mathbf{Y}(\boldsymbol{\theta}, 0) &=\mathbf{Y}_{0} \end{aligned} 
\end{equation}\tag{5}
$$

在这里，我们引入一个人工时间 $ t \in [0, T ] $。网络的深度与任意最终时间 $T$ 和 [Eq.1](#eq1)中矩阵 $ \mathbf{K}_{1} $ 和 $ \mathbf{K}_{2} $ 的大小有关。这个观测显示了学习问题  [Eq.4](#eq4) 和非线性常微分系统的参数估计方程式之间的关系。请注意，这种解释不假设 $\mathbf{F}$ 层的任何特殊结构。

ResNet 的连续解释可以通过多种方式来利用。一个想法是通过解决优化问题的层次结构来加速训练逐渐引入新权重 $\mathbf{\theta}$ 的新时间离散点。还有，新的基于最优控制理论的数值求解器已被提出在[33]中提出。另一个最近的工作[11]使用更复杂的时间积分器求解前向传播和伴随adjoint问题(在伴随问题，这种情况下通常称为反向传播)，这是计算目标函数的导数所需要的关于网络权重。



---

### 2.2. Convolutional ResNets and PDEs

在下文中，我们考虑学习任务，这些任务涉及语音，图像或视频数据提供的功能。 对于这些问题，输入特征 $ \mathbf{Y} $ 可以看作是连续函数 $ Y(x) $ 的离散化。 我们假设 [Eq.1](#eq1) 中的矩阵 $ \mathbf{K}_{1} \in \mathbb{R}^{\tilde{w} \times w_{\text {in }}} $ 和 $ \mathbf{K}_{2} \in \mathbb{R}^{w_{\text {out }} \times \tilde{w}} $ 表示卷积算子。参数$ w_{\mathrm{in}}, \tilde{w} $, 和 $ w_{\text {out }} $表示该层的宽度，即它们对应于该层的输入，中间和输出特征的数量。

我们现在显示，一类特殊的深度残差 CNN 可以解释为 PDE 的非线性系统。为了便于说明，我们首先考虑一个单通道特征与一个一维卷积，然后概述结果如何扩展到更高的空间尺寸和多通道。

假设向量 $ \mathbf{y} \in \mathbb{R}^{n} $ 表示一维网格函数，该函数通过在具有 $n$ 个元胞且网格尺寸 $h = 1 / n$  ,i.e., for $ i=1,2, \ldots, n $ 的规则网格的元胞中心处离散 $ y:[0,1] \rightarrow \mathbb{R} $ 来获得。 
$$
\begin{equation}
 \mathbf{y}=\left[y\left(x_{1}\right), \ldots, y\left(x_{n}\right)\right]^{\top} \quad \text{with} \quad x_{i}=\left(i-\frac{1}{2}\right) h 
\end{equation}
$$
假定例如 [Eq.1](#eq1) 中的算子  $ \mathbf{K}_{1} \in \mathbb{R}^{n \times n} $ 由模板 $ \boldsymbol{\theta} \in \mathbb{R}^{3} $ 参数化。应用坐标更改，我们看到
$$
\begin{equation}
 \begin{aligned} \mathbf{K}_{1}(\boldsymbol{\theta}) \mathbf{y} &=\left[\boldsymbol{\theta}_{1} \boldsymbol{\theta}_{2} \boldsymbol{\theta}_{3}\right] * \mathbf{y} \\ &=\left(\frac{\boldsymbol{\beta}_{1}}{4}[1\quad 2\quad1]+\frac{\boldsymbol{\beta}_{2}}{2 h}[-1\quad0\quad1]+\frac{\boldsymbol{\beta}_{3}}{h^{2}}[-1\quad2\quad-1]\right) * \mathbf{y} . \end{aligned} 
\end{equation}
$$
这里，权重 $ \boldsymbol{\beta} \in \mathbb{R}^{3} $ 由下式给出
$$
\begin{equation}
 \left(\begin{array}{rrr}\frac{1}{4} & -\frac{1}{2 h} & -\frac{1}{h^{2}} \\ \frac{1}{2} & 0 & \frac{2}{h^{2}} \\ \frac{1}{4} & \frac{1}{2 h} & -\frac{1}{h^{2}}\end{array}\right)\left(\begin{array}{l}\boldsymbol{\beta}_{1} \\ \boldsymbol{\beta}_{2} \\ \boldsymbol{\beta}_{3}\end{array}\right)=\left(\begin{array}{l}\boldsymbol{\theta}_{1} \\ \boldsymbol{\theta}_{2} \\ \boldsymbol{\theta}_{3}\end{array}\right) 
\end{equation}
$$
对于任何 $h> 0$ ，它都是一个非奇异的线性系统。我们用 $\boldsymbol{\beta}(\theta)$ 表示该线性系统的唯一解。 当取极限 $h→0$ ，这一观察促使人们将卷积算子参数化为
$$
\begin{equation}
 \mathbf{K}_{1}(\boldsymbol{\theta})=\boldsymbol{\beta}_{1}(\boldsymbol{\theta})+\boldsymbol{\beta}_{2}(\boldsymbol{\theta}) \partial_{x}+\boldsymbol{\beta}_{3}(\boldsymbol{\theta}) \partial_{x}^{2} 
\end{equation}
$$
变换矩阵中的各个项分别对应于反应 reaction，对流  convection，扩散 diffusion，并且 [Eq.1](#eq1) 中的偏置项分别是源/汇 source/sink 项。 请注意，可以通过乘以不同的卷积运算符或增加模板大小来生成高阶导数。

这种简单的观察揭示了学习权重对图像分辨率的依赖性，这可以在实践中利用，例如，通过多尺度训练策略[20]。 这里的想法是使用图像分辨率的粗糙到精细层次（通常称为图像金字塔）来训练网络序列。由于训练所需的操作数量和内存都与图像大小成正比，因此可以在训练期间立即节省费用，但也可以使已经训练过的网络粗糙化，从而可以进行有效的评估。 除了计算上的好处外，在粗网格上训练时忽略精细特征也可以减少陷入不希望的局部最小值的风险，这在其他图像处理应用程序中也可以观察到。

我们的论点扩展到更高的空间维度。 在2D中，例如，我们可以将由 $ \boldsymbol{\theta} \in \mathbb{R}^{9} $ 设置的 $ 3 \times 3 $ 模板参数与
$$
\begin{equation}
 \begin{aligned} \mathbf{K}_{1}(\boldsymbol{\theta})=& \boldsymbol{\beta}_{1}(\boldsymbol{\theta})+\boldsymbol{\beta}_{2}(\boldsymbol{\theta}) \partial_{x}+\boldsymbol{\beta}_{3}(\boldsymbol{\theta}) \partial_{y} \\ &+\boldsymbol{\beta}_{4}(\boldsymbol{\theta}) \partial_{x}^{2}+\boldsymbol{\beta}_{5}(\boldsymbol{\theta}) \partial_{y}^{2}+\boldsymbol{\beta}_{6}(\boldsymbol{\theta}) \partial_{x} \partial_{y} \\ &+\boldsymbol{\beta}_{7}(\boldsymbol{\theta}) \partial_{x}^{2} \partial_{y}+\boldsymbol{\beta}_{8}(\boldsymbol{\theta}) \partial_{x} \partial_{y}^{2}+\boldsymbol{\beta}_{9}(\boldsymbol{\theta}) \partial_{x}^{2} \partial_{y}^{2} \end{aligned} 
\end{equation}
$$
为了获得 [Eq.1](#eq1) 中层的完全连续模型，我们对 $ \mathbf{K}_{2} $ 进行相同的处理。 鉴于 [Eq.2](#eq2) ，我们注意到当输入和输出通道的数量大于一个时，$ \mathbf{K}_{1}$ 和 $ \mathbf{K}_{2} $ 导致耦合偏微分算子的系统。

给定 CNN 的连续时空解释，我们将优化问题 [Eq.4](#eq4) 视为最佳控制问题，类似地，将学习视为与时间相关的非线性PDE的参数估计问题 [Eq.5](#eq5) 。开发有效的数值方法来解决由最优控制和参数估计引起的PDE约束的优化问题是一项富有成果的研究成果，并导致了科学和工程学的许多进步（有关最新概述，请参见[7，24，6]）。 在机器学习应用中使用最优控制的理论和算法框架直到最近才获得一些关注（例如[15、19、9、33、11]）。



---

## 3. Deep Neural Networks motivated by PDEs

众所周知，不是每个依赖于时间的偏微分方程都是稳定的关于初始条件的扰动。若要求 [Eq.5](#eq5) 中的前向传播是稳定的，则应满足：有独立于 $T$ 的常数 $M > 0$，且：
$$
\begin{equation}
 \|\mathbf{Y}(\boldsymbol{\theta}, T)-\tilde{\mathbf{Y}}(\boldsymbol{\theta}, T)\|_{F} \leq M\|\mathbf{Y}(0)-\tilde{\mathbf{Y}}(0)\|_{F} 
\end{equation}\tag{6}
$$
其中 $\mathbf{Y}$ 和 $\tilde{\mathbf{Y}}$ 是 [Eq.5](#eq5) 对于不同初始值的解且 $ \mid \cdot \|_F $ 是 Frobenius 范数。前向传播的稳定性取决于通过求解 [Eq.4](#eq4) 选择的权重 $θ$ 的值。在学习的环境中，网络的稳定性对于对输入图像的小扰动提供鲁棒性是至关重要的。除了图像噪声之外，对手还可能故意添加干扰来误导网络的预测。最近有一些证据表明，这种微扰的存在可能会误导深层网络，因为人类观察者很难注意到这种微扰(例如[18，37，35])。



### 3.1.



### 3.2.





## 4.



## 5.



## 6.

