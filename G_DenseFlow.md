# Densely connected normalizing flows



arXiv 2021



## Abstract

标准化流是输入和具有完全分解分布的潜在表示之间的双射映射。由于精确的似然评估和有效的采样，它们非常有吸引力。 然而，由于**双射约束**限制了模型宽度，因此它们的有效能力通常不足。我们通过用噪声增量填充中间表示来解决这个问题。

我们根据先前的可逆单元对噪声进行预处理，我们将其描述为跨单元耦合 cross-unit coupling 。我们的可逆 glow-like 模块将 intra-unit 单元内仿射耦合表示为密集连接块和 Nyström 自注意力的融合。

我们将我们的架构称为 DenseFlow，因为跨单元 cross-unit 和单元内 intra-unit 耦合都依赖于密集连接。实验表明，由于提出的贡献，显着改进，并揭示了在中等计算预算下所有生成模型中最先进的密度估计。





---

## 1. Introduction

最重要的方法之一是生成图像、音频波形和自然语言符号。 为了实现预期目标，当前的技术状态使用非线性变换 [1, 2] 的深度组合，称为深度生成模型 [3, 4, 5, 6, 7]。 形式上，深度生成模型估计由一组 i.i.d 给出的未知数据分布 p D。 样本 D = {x 1 ,...,x n } 。 数据分布近似于由模型架构定义的模型分布 p θ 和一组参数 θ 。 虽然架构通常是手工制作的，但参数集 θ 是通过优化训练分布 p D 的似然来获得的：





模型的属性（例如有效采样、评估似然的能力等）直接取决于 p θ (x) 的定义，或避免它的决定。 早期方法考虑非标准化分布 [3]，这通常需要基于 MCMC 的样本生成 [8, 9, 10] 和较长的混合时间。 或者，分布可以被自回归分解 [7, 11]，这允许似然估计和强大但缓慢的样本生成。  VAE [4] 使用潜在表示的分解变分近似，它允许通过优化似然的下限来学习自动编码器。 编码器部分可以通过超参数学习或保持不变 [12、13、14]。
   在这两种情况下，解码器部分都从以潜在变量为条件的分布生成样本。 正交地，GAN [5] 忽略了可能性的因式分解。 相反，生成器网络通过在极小极大游戏中竞争来学习模仿数据集样本。 这允许有效地产生高质量的样本 [15]，但通常不会跨越整个训练分布支持 [16]。 此外，忽略 p θ 的因式分解意味着无法评估可能性。



与之前的方法相反，归一化流 [6,17,18] 使用双射映射到预定义的潜在分布 p(z) 对似然进行建模，通常是多元高斯分布。 鉴于双射f θ ，似然性定义为变量变化公式：



这种方法需要计算雅可比行列式 ( det ∂z ∂x )。 因此，在构建双射变换的过程中，非常重视易处理的行列式计算和高效的逆计算 [18 , 19 ]。 由于这些限制，与标准 NN 构建块 [20] 相比，可逆变换需要更多参数才能实现类似的容量。   尽管如此，使用双射公式对 p θ (x) 建模可以实现精确的似然评估和有效的样本生成，这使得这种方法便于各种下游任务 [21、22、23]。



双射公式（2）意味着输入和潜在表示具有相同的维度。 通常，归一化流方法的卷积单元 [18] 在内部膨胀输入的维度，提取有用的特征，然后将它们压缩回原始维度。 不幸的是，这种转换的能力受到输入维度的限制[24]。 这个问题可以通过将模型表示为一系列双射变换来解决 [18]。 然而，单独增加深度是提高深度模型能力的次优方法 [25]。 最近的工作建议通过增加输入维度来扩大流程 [24, 26]。   我们提出了该想法的有效发展，进一步提高了性能，同时放宽了计算要求。

   我们通过使用高斯噪声对中间潜在表示进行增量增强来增加归一化流的表现力。 所提出的跨单元耦合对噪声应用仿射变换，其中缩放和平移是从一组先前的中间表示中计算出来的。 此外，我们通过提出一种将全局空间上下文与局部相关性融合在一起的转换来改善单元内耦合。 所提出的面向图像的架构提高了表达能力和计算效率。 我们的模型在 ImageNet32、ImageNet64 和 CelebA 的似然评估中设置了最新的最新结果。



---

## 2. Densely connected normalizing flows

我们提出了规范化流的递归视图，并提出了基于潜在表示的增量增强以及与自注意力配对的密集连接耦合模块的改进。 然后使用改进的框架开发面向图像的架构，我们在实验部分对其进行评估。



### 2.1 Normalizing flows with cross-unit coupling

标准化流 (NF) 通过堆叠多个可逆变换来实现其表达能力 [18]。 我们用方案 (3) 说明了这一点，其中每两个连续的潜在变量 z i-1 和 z i 通过专用流单元 f i 连接。 每个流动单元 f i 是一个带有参数 θ i 的双射变换，我们将其省略以保持符号整洁。 变量 z 0 通常是从数据分布 p D (x) 中提取的输入 x。
   z 0 f 1 ←→ z 1 f 2 ←→ z 2 f 3 ←→ … f i−1 ←→ z i f i ←→ … f K ←→ z K , z K ∼ N(0,I)。  (3) 

根据变量公式的变化，可以通过对应变换J f i+1 的雅可比矩阵将连续随机变量zi 和z i+1 的对数似然关联起来[18]： 

lnp(zi ) = lnp(z i  +1 ) + ln|detJ f i+1 |。  (4) 

这种关系可以看作是一个递归。  lnp(z i+1 ) 项可以递归地替换为 (4) 的另一个实例或在潜在分布下进行评估，这标志着终止步骤。
   这种设置是大多数当代建筑的特征 [17, 18, 19, 27]。
   标准 NF 公式可以通过增加噪声变量 e i [24, 26] 的输入来扩展。
   噪声 e i 服从一些已知分布 p ∗ (e i ) ，例如 多元高斯。 我们通过将噪声递增地连接到每个中间潜在表示 z i 来进一步改进这种方法。 可以通过 ei 的蒙特卡罗采样计算似然 p(zi ) 的下界来获得这个想法的易处理公式：

 lnp(zi ) ≥ E ei ∼p ∗ (e) [lnp(zi ,ei ) − lnp  ∗ (ei )]。





学习到的分布 p(zi ,ei ) 近似于目标分布 p ∗ (zi ) 和 p ∗ (ei ) 的乘积，这在附录 D 中有更详细的解释。 我们用逐元素仿射变换来变换引入的噪声 ei  . 此变换的参数由先前表示 z <i = [z 0 ,...,z i−1 ] 的学习非线性变换 g i (z <i ) 计算。 结果层 h i 可以定义为： 

z (aug) i = h i (z i ,e i ,z <i ) = [z i ,σ ?  e i + µ], (µ,σ) = g i (z <i )。  (6) 

方括号[·,·]表示沿特征维度的串联。 为了计算 (z i ,e i ) 的似然，我们需要雅可比

 ∂z (aug) i ∂[z i ,e i ] = ?  I 0 0 diag(σ) ?
   .  (7) 

结果似然为 

lnp(z i ,e i ) = lnp(z (aug) i ) + ln|detdiag(σ)|。  (8)

 我们将等式 (5) 和 (8) 合并为一个步骤：

 lnp(zi ) ≥ E ei ∼p ∗ (ei ) [lnp(z (aug) i ) − lnp ∗ (ei ) + ln|detdiag  (σ)|]。  (9)

 我们将变换 h i 称为跨单元耦合，因为它充当一组先前可逆单元上的仿射耦合层 [17]。 输入张量的潜在部分没有变化地传播，而噪声部分则进行线性变换。 噪声变换可以看作是我们对噪声进行采样的分布的重新参数化 [4]。 通过去除噪声维度可以方便地从 z (aug) i 获得 z i 。

   图 1 比较了标准归一化流 (a) 归一化流与输入增强 [24] (b) 和建议的跨单元耦合增量增强 (c)。 每个流动单元 f 0 i 由几个可逆模块 m i,j 和跨单元耦合 h i 组成。 我们架构的主要新颖之处在于每个流动单元 f 0 i+1 相对于其前身 f 0 i 增加了维度。 跨单元耦合 h i 用噪声 e i 增加潜在变量 z i ，该噪声立即通过仿射变换进行变换。 仿射变换的参数由任意函数 g i 获得，该函数接受多个先前的变量 z <i 。 请注意，反转方向不需要计算 g i ，因为我们只对 z i 的值感兴趣。 为了进一步说明，我们展示了扩展框架的似然计算。







**Example 1(Likelihood computation)**

设m 1 和m 2 分别是从z 0 到z 1 和z (aug) 1 到z 2 的双射映射。 设h 1 是从z 1 到z (aug) 1 的跨单元耦合，z (aug) 1 = [z 1 ,σ?e i +μ] 。
   假设 σ 和 µ 由任何不可逆神经网络 g i 计算。 网络接受 z 0 作为输入。 我们根据以下等式序列计算输入 z 0 的对数似然：[变换、跨单元耦合、变换、终止]。

lnp(z 0 ) = lnp(z 1 ) + ln|detJ f 1 |, (10) 

lnp(z 1 ) ≥ E ei ∼p ∗ (ei ) [lnp(z (aug) 1 ) − lnp(ei )  + ln|detdiag(σ)|], (σ,µ) = g 1 (z 0 ), (11) 

lnp(z (aug) 1 ) = lnp(z 2 ) + ln|detJ f 2 |, (12  ) 

lnp(z 2 ) = lnN(z 2 ;0,I)。  (13) 

我们在训练期间使用单个样本和评估期间的数百个样本使用 MC 采样来近似期望，以减少似然的方差。 但是请注意，我们的架构通过单次通过生成样本，因为逆不需要 MC 采样。
   我们在整个架构中反复应用跨单元耦合 h i 以实现中间潜在表示的增量增强。 因此，数据分布在比输入空间更高维度的潜在空间中建模 [24, 26]。 这使得最终潜在表示与 NF 先验更好地对齐。 我们通过开发一种我们称之为 DenseFlow 的面向图像的架构来实现规范化流框架的拟议扩展





## 2.2. Image-oriented Architecture

我们提出了一种基于跨单元耦合增量增强的面向图像的架构。 每个 DenseFlow 单元 f i 包含几个类似发光的模块 m i,j ，包括激活归一化、1×1 卷积和仿射耦合。
   与最初的辉光设计 [19] 不同，我们提出了基于耦合网络内的密集连接和快速自注意力的高级转换。 所有这些转换都旨在捕获复杂的数据依赖性，同时保持易处理的雅可比矩阵和高效的逆计算。 为了完整起见，我们首先回顾原始发光模块 [19] 的元素。

ActNorm [19] 是批量标准化 [30] 的可逆替代品。 它使用每个通道的尺度和偏置参数执行仿射变换：

y i,j = s ?  x i,j + b。  (14) 

尺度和偏差计算为初始小批量的方差和均值。

可逆 1×1 卷积是通道置换 [19] 的泛化。 具有 1×1 核的卷积在构造上是不可逆的。 相反，正交初始化和损失函数的组合使内核逆数值保持稳定。 标准化流量损失最大化 ln|detJ f | 这相当于最大化 P i ln|λ i |  ，其中 λ i 是雅可比矩阵的特征值。 保持特征值的相对较大的幅度确保了稳定的反演。 这种变换的雅可比可以通过 LU 分解 [19] 有效地计算。

仿射耦合 [18] 将输入张量 x 通道拆分为两半 x 1 和 x 2 。 前半部分无变化地传播，而后半部分进行线性变换 (15)。 线性变换的参数从前半部分开始计算。 最后，将两个结果连接起来，如图 2 所示。



参数 s 和 t 是使用可训练网络计算的，该网络通常作为残差块 [18] 实现。 但是，此设置只能捕获局部相关性。 受判别架构 [29、31、32] 最新进展的启发，我们设计了耦合网络以融合全局上下文和局部相关性，如图 2 所示：首先，我们将输入投影到低维流形中。 接下来，我们将投影张量提供给一个密集连接的块 [29] 和自注意力模块 [31, 33]。 密集连接块捕获局部相关性 [34]，而自注意力模块捕获全局空间上下文。 这两个分支的输出通过 BN-ReLU-Conv 单元连接和混合。 像往常一样，获得的输出参数化耦合层内的仿射变换。

   众所周知，成熟的自注意力层具有非常大的计算复杂度。   在需要许多耦合层和大潜在维度的标准化流的情况下尤其如此。 我们通过使用由 Nyström 方法 [28] 近似的自注意力来缓解这个问题。  Nyström 方法使用低秩矩阵近似而不是实际的键、查询和值矩阵。



最终架构。 我们首先将包含所提出的耦合网络的 N 个类似辉光的模块 (m i = m i,N ◦ · · · ◦ m i,1 ) 连接到一个可逆的 DenseFlow 单元 (f 0 i ) 中，然后是一个跨单元耦合 ( h i )。
   每个 DenseFlow 单元的输入是前一个单元的输出，增加了噪声并以跨单元耦合方式进行了转换。 添加的噪声通道数定义为增长率超参数。 通常，由于潜在表示大小的增加，与初始单元相比，后面的 DenseFlow 单元中的辉光模块数量更大。 我们堆叠 M 个 DenseFlow 单元以形成一个 DenseFlow 块。 块中的最后一个可逆单元没有相应的跨单元耦合。 我们堆叠多个 DenseFlow 块以形成具有足够容量的标准化流。 在两个连续的块之间，我们应用了空间到通道的整形操作并省略了一半的维度 [18]。 我们将开发的架构表示为 DenseFlow - L - k ，其中 L 是模型中发光模块的总数， k 代表使用的增长率。 开发的架构使用两个独立级别的跳跃连接。   第一级（单元内）由每个耦合网络内部的跳跃连接形成，而跨单元跳跃形成第二级。

   图 3 显示了所提出模型的最终架构。 灰色方块代表 DenseFlow 单位。   跨单元耦合用蓝点和虚线跳过连接表示。 最后，连续 DenseFlow 块之间的空间到信道操作由虚线方块表示。 提议的 DenseFlow 设计在更大维度的张量上应用可逆但功能较弱的变换（例如卷积 1×1）。 另一方面，强大的不可逆变换（例如耦合网络）在低维张量上执行大部分操作。 这导致了资源效率更高的架构。


