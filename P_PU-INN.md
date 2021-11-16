# PU-INN

## 论文注意要点

以下是最关键的问题。

- 缺少消融研究，以证明双向映射（即标准化流）是有用的（R3）
- 在pugeo-net中的比较似乎并不公平（R2）
- 缺少对pu-gan数据集的比较(R2)
- 缺乏稳健性，例如对邻域的敏感性，K的不同值会影响（R1）。
- 论文的动机motivation需要改进（R1，R2，R3，R4）
- 缺少引用（R1、R2、R3）（该论文是在CVPR 2021之后~2周提交的--因此，CVPR 2021论文在提交时是同期工作，因此不需要进行比较，但它们仍应作为同期工作被引用）。如果作者决定重新提交，与这些较新的工作进行比较会使论文更有力。



---

### 评论1

这项工作旨在通过使用条件归一化流（CNF），从稀疏的输入中产生一个密集的点集。

CNF 的设计在点云上采样方面似乎很合理。我很欣赏作者将 CNF 引入这一任务并进行了合理的修改。但仍有一些问题需要作者解决。

1) 对我来说，论文的**动机并不明确**。**在没有明确目标的情况下，直接将新技术应用于点云的上采样，这并不令人信服**。另外，即使在 "PUGeo-Net测试数据集中选定的11个模型和PU36数据集中的36个模型 "上，**性能的提高**也是非常**有限**的。
2) 这种方法的**稳健性**很难让人信服。正如在消融研究中所显示的，**网络对超参数相当敏感**，例如，**邻接点的数量**从8个改为6个，这个方法就不如以前的方法了。

3) 如图4和图5所示，与以前的方法相比，也很难说有什么改进。在Kitti上的**视觉结果**似乎也比其他论文中显示的要**差**，例如PU-GAN。

4) 在这次提交之前，有更多的论文[ref-1~2]发表了关于升采样的内容。也请讨论并**与SOTA进行比较**

[ref-1] PU-GCN: Point Cloud Upsampling via Graph Convolutional Network. CVPR 2021
[ref-2] Point Cloud Upsampling via Disentangled Refinement. CVPR 2021



其他问题。
1. 请解释该稿件如何推进该领域的研究和/或对文献的新贡献。你认为该论文最强和最弱的方面是什么？还请评论该论文的新颖性，其潜在的影响，以及结果和方法本身的可复制性。**本文通过使用条件归一化流（CNF）并进行合理的修改，对点云上升采样任务进行了研究。虽然这种新技术似乎是有效的，但其动机并不明确，而且改进也很有限。**
2. 该稿件在技术上是否**合理**？请在下面的公共评论中解释你的答案。**部分**
3. 稿件是否包含足够和适当的参考文献？请在下面的 "公众意见 "中解释并列出缺少的参考文献。**缺少重要的参考文献；需要更多的参考文献**
4. **introduction 是否用鼓励读者继续阅读的措辞说明了稿件的目的**？可以改进
5. 你如何评价**稿件的组织**？是否有**重点**？**长度与主题是否合适**？



---

### 评论2

contribution and experiments 都比较让审稿人满意。但仍有几点值得关注。

1. 由于作者在Tab.1中follows了pugeo-net的实验设置，但现有方法的结果似乎与pugeo-net的结果不同。**最好是直接引用pugeo-net的结果并与之比较，重新训练其他方法的模型似乎不公平。**

2. 建议作者提供在pu-gan（pu-gan[1]和dis-pu[2]）或pu1k（pu-gcn[3]）数据集上的结果，并与dis-pu和pu-gcn进行比较，因为这两个数据集是最先进的方法，而**pu-gan数据集在现有论文中被广泛使用**。

3. 如果能**给出更大的上采样率的定量比较**，比如pugeo-net和dis-pu的**16倍**，会更令人满意。

4. 建议提供与其他方法在真实扫描real scans（图6）和non-uniform非均匀数据（图10）上的**比较**。

5. 除了在Tab.1&2 中的模型尺寸比较，还建议提供**内存消耗的比较**。审稿人可能会担心，有**大量邻居的DGCNN可能会占用更多的内存**。

6. "条件归一化流" 的含义似乎与现有的工作不同，如"SRFlow[4]"和 "C-Flow[5]"。在以前的作品中，**"条件" 是指在从采样的高斯噪声中生成事物时加入图像或其他信息**，而作者的意思是**将输入点中的一组点特征注入到相同输入的另一组特征中**。是不是叫 "**feature fusion 特征融合**" 更好？

[1] Li, Ruihui, et al. "Pu-gan: a point cloud upsampling adversarial network." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
[2] Li, Ruihui, et al. "Point Cloud Upsampling via Disentangled Refinement." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
[3] Qian, Guocheng, et al. "Pu-gcn: Point cloud upsampling using graph convolutional networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
[4] Lugmayr, Andreas, et al. "Srflow: Learning the super-resolution space with normalizing flow." European Conference on Computer Vision. Springer, Cham, 2020.
[5] Pumarola, Albert, et al. "C-flow: Conditional generative flow models for images and 3d point clouds." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.



本文通过使用**归一化流**和**特征插值技术**提出了一种新的点云上采样方法。本文显示了对点云上采样课题的良好影响 The paper shows good impact on point cloud upsampling topic。进行了各种实验来验证该方法的优越性能，但缺少与最先进方法的比较。一个小问题是论文的**新颖性**，因为这些技术看起来像是**不同的现有论文的技术的组合**。



---

### 评论3

作者**遗漏了太多的重要实验，无法充分证明其贡献**。



**优点**

- 本文提出了一种独特的点云上采样技术，包括归一化流和特征插值。因为归一化流学习了**点云从欧几里得空间到潜伏空间的双向映射**，**潜伏空间的插值立即反映了欧几里得空间的变化**。This paper presents a unique technique for point cloud upsampling that includes normalizing flows and feature interpolation. Because the normalizing flow learns a bijective mapping of point clouds from Euclidean space to latent space, interpolating in latent space immediately reflects changes in Euclidean space.
- 这些小雕像很有吸引力，也很有用。

**缺点**

1）**使用归一化流的好处没有得到明确验证**。

- 缺少了**最关键的消融研究**，**无法说明为什么双向映射（归一化流）是重要的**。最基本的，你必须保持流模块，**但不依靠它的反向映射来产生上采样点**，相反，你可以使用解码器（几层MLPs，如PU-Net、MPU或PU-GAN）从潜空间产生上采样点。 
- 改进并不强。如表3所示，**邻域大小对性能的影响比其他建议的模块都要大**，建议的PU-Flow在K=16的情况下只达到了1.1599的CD。**MPU也使用了K=16，他们的性能是1.022 CD，比PU-Flow好**。鉴于**MPU利用了与EdgeConv相同的特征提取器，这可能意味着归一化流没有帮助**，或者说目前提交的材料没有充分展示它。

2）**PU-GCN 和Dis-PU**

3）生成 claim 没有得到完全验证。论文声称 PU-Flow 可以产生均匀分布在底层表面的密集点（Page 1 L13）。在**下表面上产生点的能力是一个强有力的断言**。然而，**不清楚为什么归一化流可能对此有帮助。**

4）在 NF 部分的论文阅读需要改进。作为作者，你最好强调读者想知道的信息。比如说。

- **两个不同维度的空间之间的双向映射是如何进行的**？从一个低维空间到一个高维空间是一个不理想的问题 ill-posed problem不适定问题，因为缺少信息。我必须在稿件的文字之间找出答案，我相信答案就在前一篇论文提出的多尺度归一化流中。**这一部分应该在论文中写清楚，而且应该强调。**
- **双向映射（归一化流）有多大帮助**？这一点应该在文件中适当说明和强调。

5）**小的弱点--有限的新颖性**。归一化流不是一个重要的贡献，因为它已经在许多研究中被用于点云的生成。我同意应该承认利用归一化流进行点云上采样的贡献，但这不足以在旗舰期刊上发表。



---

### 评论4

我喜欢通过插值表示的点上采样方法，该方法编码了通过归一化流学到的**点分布先验**。这篇论文写得很好，很容易理解。对以前工作的参考是全面的，对多个数据集的评估是值得赞赏的。

我确实希望作者能在修改中解决以下问题。
1. 为什么作者选择使用**离散的归一化流**转换而不是PointFlow中使用的**连续归一化流**？离散流的使用似乎对模型结构造成了限制，特别是Sec 4.1中提到的瓶颈问题。这些问题是否可以通过使用一个连续的表述来解决？

2. **插值是在潜空间中进行的**，**但在欧氏空间中定义的邻域上进行**。实际上**在潜空间中寻找邻域不是更有意义**吗？在以前的生成方法[PointFlow]中，潜变量的平滑插值表明，潜空间实际上可能有一个嵌入的**语义相似性测距概念**。这个方案的另一个好处是，人们也许可以**使用简单的插值方法**，如 midpoint injection，**而不是使用学习模块来预测插值权重，这受制于一个固定的上采样率R**。

   In previous generative methods [PointFlow], the smooth interpolation of latent variables suggest that the latent space may actually have a notion of semantic similarity geodesic distance embedded.



论文 PU-Flow 介绍了一种新颖的点集上采样方法。与以前的点上采样网络不同，它的方法是基于生成性建模：NF。该方法本身是非常新颖的，可以启发进一步的研究。同时，与以前的方法的比较也显示出令人信服的优势。



---

# Figure

## 方法图

- [ ] 论文示意图 
- [ ] 论文总架构图
- [ ] 残差块示意图
- [ ] 局部特征提取图
- [ ] 全局特征提取图
- [ ] 特征聚合图
- [ ] 特征插值图



## 实验结果图

- [ ] Fig. 4 点云对比图，可视化P2F误差 5*6 矩阵 5个模型 （5个方法+1个GT）    一个boudingbox放大展示局部信息



- [ ] Fig. 5 曲面对比图，5*6 矩阵 5个模型 （5个方法+1个GT）  一个boudingbox放大展示局部信息



- [ ] Fig. 6 真实场景对比图，2*4矩阵 第一行为稀疏输入，第二行为稠密输出；4个场景，每个场景一个boundingbox放大



- [ ] Fig. 7 不同类别的生成结果，根据最近的点颜色。2*4矩阵 第一行为稀疏输入，第二行为稠密输出；



- [ ] Fig. 8 输入噪声结果对比。两个2行4列，第一行为稀疏输入，第二行为稠密输出； 输入噪声分别为 0 0.5% 1.0% 1.5%

![image-20211114203753145](assets/image-20211114203753145.png)



- [ ] Fig .9 输入点数结果对比。 两个2行4列，第一行为稀疏输入，第二行为稠密输出； 输入点数分别为 512 1024 2048 4196

![image-20211114173630989](assets/image-20211114173630989.png)

![image-20211114174021611](assets/image-20211114174021611.png)



- [ ] Fig. 10 输入不均匀结果对比。3行2列，两行两个物体；第一列为稀疏输入，第二列为稠密输出，第三列为GT







# Global Feature

**Global Context Reasoning for Semantic Segmentation of 3D Point Clouds**

全局上下文依赖性对于 3D 点云的语义分割很重要。 然而，大多数现有方法堆叠特征提取层以扩大感受野，以沿空间维度聚合更多点的上下文信息。 在本文中，我们提出了一个点全局上下文推理（PointGCR）模块来沿通道维度捕获全局上下文信息。 在 PointGCR 中，使用无向图表示（即 ChannelGraph）来学习通道独立性。 具体来说，通道图首先表示为图节点，然后节点之间的独立性表示为图边。  PointGCR 是一个即插即用的端到端可训练模块。 它可以很容易地集成到现有的分段网络中，并实现显着的性能提升。 我们进行了广泛的实验来评估室内和室外数据集上提出的 PointGCR 模块。实验结果表明，我们的 PointGCR 模块有效地捕获了全局上下文依赖关系，并显着提高了几个现有网络的分割性能



**3DContextNet: K-d Tree Guided Hierarchical Learning of Point Clouds Using Local and Global Contextual Cues**

3D 点云的分类和分割是计算机视觉中的重要任务。 由于点云的不规则性，大多数现有方法在将点云用作 ConvNet 的输入之前将其转换为规则的 3D 体素网格。 不幸的是，体素表示对 3D 数据的几何性质高度不敏感。 最近的方法将点云编码为更高维的特征以覆盖全局 3D 空间。 然而，这些模型无法充分捕捉点云的局部结构。
   因此，在本文中，我们提出了一种利用 k-d 树强加的局部和全局上下文线索的方法。 该方法旨在沿着树结构逐步学习表示向量。 在具有挑战性的基准上的实验表明，所提出的模型提供了判别点集特征。 对于 3D 场景语义分割任务，我们的方法明显优于斯坦福大型 3D 室内空间数据集 (S3DIS) 上的最新技术。



**3D Point Capsule Networks**

在本文中，我们提出了 3D 点胶囊网络，这是一种自动编码器，旨在处理稀疏的 3D 点云，同时保留输入数据的空间排列。  3D 胶囊网络是我们新颖的统一 3D 自动编码器公式的直接结果。 他们的动态路由方案 [30] 和我们的方法部署的特殊 2D 潜在空间为几个常见的点云相关任务带来了改进，例如我们的广泛评估证实的对象分类、对象重建和部分分割。 此外，它还支持新的应用，例如零件插补和更换。





**SCANet: Spatial-channel attention network for 3D object detection**

本文旨在实现高精度 3D 对象检测，其中我们提出了一种新颖的空间通道注意网络 (SCANet)，这是一种两阶段检测器，将 LIDAR 点云和 RGB 图像作为输入来生成 3D 对象估计。 第一阶段是 3D 区域提议网络 (RPN)，我们在其中提出了新的空间通道注意 (SCA) 模块和扩展空间上采样 (ESU) 模块。 使用金字塔池化结构和全局平均池化，SCA 模块不仅可以有效地结合多尺度和全局上下文信息，还可以产生空间和通道方面的注意力来选择判别特征。 解码器中的 ESU 模块可以恢复由连续池化算子导致的丢失空间信息，以生成可靠的 3D 区域提议。 在第二阶段，我们设计了一种新的多级融合方案，用于准确分类和 3D 边界框回归。
   实验结果表明，SCANet 在具有挑战性的 KITTI 3D 对象检测基准上实现了最先进的性能。





# Attention

**ACMMM2020_CF-SIS: Semantic-Instance Segmentation of 3D Point Clouds by**
**Context Fusion with Self-Attention**

![image-20211110153842126](assets/image-20211110153842126.png)

**3.2. Inter-task Context Fusion** 

任务间上下文融合由两个相同的子分支组成，即 Sem2Ins 子分支和 Ins2Sem 子分支，只是 Ins2Sem 没有融合门。 不失一般性，以Sem2Ins 子分支为例进行进一步说明。

总的来说，在这个子分支中，我们首先根据语义原始特征 $ F_{\text {sem }}=\left\{s_{i} \mid i=1,2, \ldots, N\right\} $ 通过计算语义注意力来建模点之间的语义相关性。 然后，语义上下文通过语义原始特征的加权平均值聚合，并与实例原始特征 $ F_{i n s}=\left\{\boldsymbol{u}_{i} \mid i=1,2, \ldots, N\right\} $ 使用融合门融合 fusion gate，以进行控制从语义分支到实例分支的信息流。

具体来说，给定一个由语义原始特征 $s_i$ 表示的点，形成其对应语义上下文的 $N-1$ 个候选点是 $ S_{i}=\left\{s_{i_{j}} \mid j \in\{1,2, \ldots, N-1\}\right\} $  . 为了对中心点 $s_i$ 和其他点之间的语义关系进行建模，我们提出了空间注意力来计算中心点和候选点的特征之间的**余弦相似度分数**作为衡量语义关系的分数，给出为 $ q_{i, j}=\frac{s_{i}^{T} s_{i_{j}}}{\left\|s_{i}\right\|_{2}\left\|s_{i_{j}}\right\|_{2}} $ 。**余弦距离可以看作是一种手工制作的特征来表征点之间的语义关系**。我们建议在高维特征空间中**使用余弦距离而不是可学习的注意力机制的原因如下**。 首先，余**弦距离可以迫使网络考虑点之间的空间关系，因为两个相邻点自然会产生相似的特征并产生高余弦相似度**。 其次，**余弦距离可以从其他分支引入更多信息，因为余弦距离是未平滑的激活分数，与可学习注意力中使用的平滑（通过 softmax 激活）注意力分数相比，它具有明显更高的值**。 语义特征上下文 $ F_{\text {sem-c }}=\left\{c_{i} \mid i \in\{1,2, \ldots, n\}\right\} $ 计算为所有候选点的加权平均值，如 $ \boldsymbol{c}_{i}=\sum_{j=1}^{N-1} q_{i, j} \boldsymbol{s}_{i_{j}} $

请注意，由于以下两个原因，任务间上下文融合**考虑了整个输入点集的上下文提取**。 首先，与 3D 空间相比，kNN 在高维特征空间中搜索邻居点的计算成本较高，难以选择性地计算点及其语义相关邻居之间的注意力。其次，收集整个特征空间的语义上下文可以扩展中心点的感知范围，从而迫使网络学习更具有辨别力的相关点选择，并提取更多信息量的语义上下文。



**Fusion gate in inter-task context fusion module**.  由于与 Ins2Sem 子分支相比，Sem2Ins 子分支位于网络的最底层，其中原始语义特征用于上下文提取。因此，**提取的语义上下文是嘈杂的并且可能包含错误的信息**，如果直接融合到实例特征中而不进行预处理，这会损害实例分割的性能，**尤其是在训练的早期阶段**。 因此，**为了滤除语义上下文中的噪声和错误信息，并使训练对模型的初始化和超参数不那么敏感，我们建议使用融合门来控制两者之间的信息流任务。融合门使网络能够自动学习决定何时以及允许多少信息从一项任务流入另一项任务。**Therefore, in order to filter out the noise and the erroneous information in the semantic context, and make the training less sensitive to the initialization and hyper-parameters of the model, we propose to use a fusion gate to control the information flow between the two tasks. The fusion gate enables the network to automatically learn to decide when and how much information is allowed to flow from one task into the other.  具体来说，融合门使用线性变换确定语义特征上下文 $ F_{\text {sem-c }} $ 的质量，并使用 sigmoid 激活对输入特征的每个位置预测 0 到 1 之间的概率。 预测概率表示允许通过的语义上下文的比例。 该过程可以表述为
$$
\begin{equation}
 \beta_{g, i}=\sigma\left(W_{g} c_{i}+b_{g}\right) 
\end{equation}
$$
其中 $\sigma$ 表示 sigmoid 激活，$ \left\{W_{g}, \boldsymbol{b}_{g}\right\} $  是可学习矩阵和线性变换偏差。融合实例特征矩阵  $ F_{i n s-f}=\left\{u_{f, i} \mid i \in\{1,2, \ldots, n\}\right\} $ 门控语义特征上下文矩阵 和实例特征矩阵 $ F_{\text {sem }-c} $ ，给出为
$$
\begin{equation}
 \boldsymbol{u}_{f, i}=\beta_{g, i} \boldsymbol{c}_{i}+\left(1-\beta_{g, i}\right) \boldsymbol{u}_{i} 
\end{equation}
$$
我们注意到融合门也部署在 Sem2Ins 分支中，其结构与上述相同。



---

**3.3 Intra-task Context Fusion**

除了有助于不同任务之间的上下文融合，任务内部注意力的存在还可以帮助模型捕捉语义和实例特征，通过向中心点隐式指出相关区域。 这对于从点云中感知和提取 3D 视觉信息很重要，因为与单个点相比，相关区域包含更多关于语义和实例特征的信息，可以进一步用于区分点背后的语义标签和实例对象， 并产生更具辨别力的分割结果。 在本小节中，我们从之前的基于自我注意的工作 [35] 中找到灵感，通过有效的任务内上下文融合来学习特征空间中的每点上下文。
   该模块由两个相同的子分支组成，即Ins2Ins子分支和Sem2Sem子分支。 在不失一般性的情况下，我们只描述 Ins2Ins 子分支的详细结构。 该模块的输入是融合实例特征𝐺 𝑗𝑜𝑡 - 𝑔 = {𝒗 𝑔,𝑗 |𝑗 ∈{ 1 , 2 ,..,𝑂}} .这些特征首先被转化为三个不同的特征子空间{𝑠,𝑚,ℎ}，通过多层感知器𝑔，参数为𝜾，给出为𝒗𝜐，𝑗∈𝑜，𝑗∈𝑜，𝑗∈𝑜  ,𝑚,ℎ}。  (3) 注意力计算为子空间 𝑠 和 𝑚 中特征向量之间的点积，并使用 softmax 激活进行平滑，可以写成 𝑏 𝑗,𝑘 = exp(𝒗 T 𝑚,𝑗 𝒗 𝑠,𝑘 )  ˝ 𝑂 𝑘=1 exp(𝒗 T 𝑚,𝑗 𝒗 𝑠,𝑘 ) 。  (4) 最后的实例点嵌入 𝐹 𝑗𝑜𝑡 = {𝒇 𝑗𝑜𝑡,𝑗 } 是子空间ℎ中所有特征的加权平均值，具有链接到 𝐺 𝑗𝑜𝑔的子空间 𝑂 ∑︁ 𝑘=1 𝑏 𝑗, 𝑘 𝒗 ℎ, 𝑘 + 𝒗 𝑔 ,𝑗 |𝜾 𝑝 ), (5) 其中𝜇 𝑝 是一个随机学习的初始平衡因子，在训练期间随其他可学习的平衡因子一起更新。 请注意，任务内上下文融合旨在细化实例特征，这些特征将用于预测每个点的实例标签。 为了保留原始实例特征的信息，上下文特征得分相对较小的平滑注意力比未平滑余弦距离更可取。







x**WACV2020_Global Context Reasoning for Semantic Segmentation of 3D Point Clouds**

- 一定要参考它的**Implementation Details**.



**CVPR2021_SOE-Net_A_Self-Attention_and_Orientation_Encoding_Network_for_Point_Cloud**

- Sec 4.2

![image-20211111175727094](assets/image-20211111175727094.png)

图 5 展示了它的架构。  给定局部描述符 $ F_{L} \in \mathbb{R}^{N \times C} $ ，其中 $N$ 是点数，$C$ 是通道数，我们分别将 $ F_{L}$ 馈入两个 MLP 并生成新的特征图 $ X \in \mathbb{R}^{N \times C}, Y \in \mathbb{R}^{N \times C} $ 。 然后计算注意力图 $W$，定义如下： 
$$
\begin{equation}
 W_{j, i}=\frac{\exp \left(Y_{j} \cdot X_{i}^{T}\right)}{\sum_{i, j=1}^{N} \exp \left(Y_{j} \cdot X_{i}^{T}\right)} 
\end{equation}
$$
其中，$W_{j, i}$ 表示第 $i$ 个局部描述符对第 $j$ 个局部描述符的影响，形状为 $N × N$。**这里，它认为是学习逐点局部描述符之间的长程依赖关系的组件。  更重要的局部描述符将对目标全局描述符的表示做出更多贡献**。在另一方面  $ F_{L}$ 被送入另一个 MLP 以输出一个新的特征图 $ Z \in \mathbb{R}^{N \times C} $。 之后，我们将其与 $W$ 的转置相乘以生成结果 $ A^{P} \in \mathbb{R}^{N \times C} $。 最后，我们在其上添加一个尺度参数 $α$ 并加回 $ F_{L}$  ，其定义如下：
$$
\begin{equation}
 F_{L}^{\prime}=\mu A^{p}+F_{L}=\mu W^{T} Z+F_{L} 
\end{equation}
$$
其中 $\mu$ 被初始化为零，并随着学习的进展逐渐分配更多的权重。**与原始局部描述符相比，最终输出具有全局上下文视图。 这通过结合几何和上下文信息来增强特征集成**。







√**CVPR2018_Attentional_ShapeContextNet**

https://github.com/umyta/A-SCN

![image-20211109224458552](assets/image-20211109224458552.png)

- **2.4. Attentional ShapeContextNet**

我们现在介绍一种受自然语言处理（sequence-tosequence）任务研究启发的不同方法。 传统的序列到序列模型通常采用循环神经网络（例如 LSTM[15]）、外部存储器或时间卷积来捕获上下文信息。[35] 中提出的点积自注意力是一种通过轻量级门控机制处理长路径长度上下文建模的模型，其中使用简单的点积生成注意力权重矩阵。值得注意的是，自注意力对于输入顺序也是不变的。与传统的基于注意力的序列到序列模型不同，在自注意力块中，查询向量 $ Q \in \mathcal{R}^{D_{Q}} $、键向量 $ k \in \mathcal{R}^{D_{k}} $（通常 $D_Q = D_K$ ）和值向量 $ V \in \mathcal{R}^{D_{V}} $ 是从相同的输入中学习的。 在监督分类设置中，可以认为 Q、K 和 V 只是由三个独立的 MLP 层学习的三个特征向量。 注意力权重通过 Q 和 K 的点积计算，然后与 V 相乘以获得变换后的表示。

图 2 显示了手动指定的形状上下文内核和可自动学习的自注意力机制之间的异同：它们都旨在捕获相对位置上的分布； 它们在第 2.2 节中统一在相同的表述下；  self-attention 中的选择操作不依赖于手工设计的 bin 分区，因为它可以从数据中学习； 与使用简单的 sumpooling 函数相比，self-attention 通过采用加权和聚合函数具有更好的建模能力。



**Selection and Aggregation**. 我们考虑在大小为 N 的整个点云 P 上计算自注意力。 选择操作产生一个软亲和矩阵，它是大小为 N × N 的自注意力权重矩阵 A，聚合操作是由权重矩阵A和变换值向量 V 点积，
$$
\begin{equation}
\text{Attention}  (Q, V, K)=\operatorname{Softmax}\left(\frac{Q K^{T}}{\sqrt{D_{Q}}}\right) \cdot V 
\end{equation}
$$


**Transformation**. 可以在每次自注意力操作之后添加具有 ReLU 激活函数的 MLP 作为特征转换操作（等式 1）。 为了进一步提高模型的表达能力，我们向 MLP 添加了一个简单的特征门控层，类似于 [8, 24]。



---

√**CVPR2020_SA-Net_PointCloud_Completion_by_Skip-attention_Network_with_HierarchicalFolding**

https://github.com/RaminHasibi/SA_Net





√**TVCG2019_VoxSegNet- Volumetric CNNs for semantic**

https://github.com/screnary/VoxSegNet

- **3.5 Attention Feature Aggregation**

![image-20211109225200508](assets/image-20211109225200508.png)





x**CVPR2019_Modeling Point Clouds with Self-Attention and Gumbel Subset Sampling**

We propose to use attention layers to capture the relations between the points. MHA is successful in modeling relations by introducing a critical multi-head design [10, 40, 28], however we argue that it is voluminous for modeling point clouds. To this regard, we propose a parameter-efficient Group Shuffle Attention (GSA) to replace MHA. There are two improvements over MHA:  

我们建议使用注意力层来捕捉点之间的关系。MHA 通过引入关键的多头设计 [10, 40, 28] 在关系建模方面取得了成功，但是我们认为它对于点云建模非常庞大。对此，我们提出了一种参数高效的 Group Shuffle Attention (GSA) 来代替 MHA。 对 MHA 有两个改进：

Firstly, toget rid ofposition-wise MLPs, we integrate the non-linearity σ into attention modules, named non-linear self-attention,首先，为了摆脱position-wise MLPs，我们将非线性σ整合到注意力模块中，称为非线性自注意力，

Attn σ (Q,X) = S(Q,X) · σ(X), (7)

where we use a Scaled Dot-Product attention [39] for S, i.e., S(Q,X) = softmax(QX T / √ c), and ELU activation [8] for σ. In other words, we use the ”pre-activation” to attend to the ”post-activation”. 其中我们对 S 使用 Scaled Dot-Product attention [39]，即 S(Q,X) = softmax(QX T / √ c)，对 σ 使用 ELU 激活 [8]。 换句话说，我们使用“预激活”来处理“激活后”。



---

# Interpolation

**ICCV2019Oral_Interpolated convolutional networks**

点云是一种重要的 3D 表示类型。然而，由于稀疏、不规则和无序的数据结构，直接在点云上应用卷积具有挑战性。 在本文中，我们提出了一种新颖的插值卷积操作 InterpConv 来解决点云特征学习和理解问题。 关键思想是利用一组离散的内核权重，并通过卷积的插值函数将点特征插入到相邻的内核权重坐标中。 引入归一化项来处理不同稀疏级别的邻域。 我们的 InterpConv 被证明是排列和稀疏不变的，并且可以直接处理不规则输入。



**3.1. Convolutions on Point Sets**

标准的 2D 和 3D 卷积在处理图像和体素网格等规则排列的数据方面取得了巨大成功。 当涉及稀疏和不规则的点集（例如 3D 点云）时，已经提出了多种卷积变体。 在本节中，我们将回顾这些卷积以激发 InterpConv 操作的设计。

考虑标准的 3D 卷积，让 3D 体素网格或特征表示为 F : Z 3 → R c ，卷积核权重 W 是一系列 1 × c 权重向量，其中 c 是通道数。 位置 ˆp 处的标准卷积可以表示为
$$
\begin{equation}
 F * W(\hat{p})=\sum_{p^{\prime} \in \Omega} F\left(\hat{p}+p^{\prime}\right) \cdot W\left(p^{\prime}\right) 
\end{equation}
$$
当涉及到不规则输入时，点不再有规律地排列，点之间的距离变得不规则。 一些方法 [43, 46] 采用**连续权重函数** W(p δ )，它以相邻点 ˆ p+p δ 到中心点 ˆ p 的相对坐标 p δ 作为输入，来预测卷积权重。 连续函数 W(p δ ) 不再是一个 1×c 权重向量，而是一个由 MLP 实现的映射 R 3 → R c 。 那么连续卷积可以表示为
$$
\begin{equation}
 F * W(\hat{p})=\sum_{p_{\delta}} F\left(\hat{p}+p_{\delta}\right) \cdot W\left(p_{\delta}\right) 
\end{equation}
$$
值得注意的是，应用图神经网络 [45, 49, 34] 来处理点云本质上与连续卷积具有相同的想法。用连续函数 W(p δ ) 替换离散内核权重 W(p 0 ) 仍然存在一些问题。 简单地通过 MLP 学习连续函数并不总是在实践中起作用 [49]。 预测的参数可能太多，学习过程效率低下，有时不稳定。 图像中离散内核的巨大成功的知识也不能转移到点云识别任务中。



**3.2. Interpolation Weight**

在我们的方法中，我们采用离散卷积权重的设计，同时保持连续距离的特征，通过将 W(p δ ) 解耦为两部分：**空间离散核权重** W(p 0 ) ∈ R c 和**插值函数** T(  p δ ,p 0 )。 我们注意到空间离散的核权重 W(p 0 ) 是一个 1×c 的向量，可以在训练期间初始化和更新，p 0 是这个核权重向量到核中心的相对坐标。

插值函数T(p δ ,p 0 ) : R 3 × R 3 → R 以核权重向量p 0 和相邻输入点p δ 的坐标为输入，通过一定的插值算法计算权重。 我们的方法考虑了核权重向量附近的每个输入点。 为了使卷积稀疏不变，每个内核权重向量 W(p 0 ) 需要一个密度归一化项 N p 0 ，它对 p 0 邻域中的插值权重或输入点数求和。 最后，我们以位置 ˆp 为中心的 InterpConv 可以表示为
$$
\begin{equation}
 F * W(\hat{p})=\sum_{p^{\prime}} \frac{1}{N_{p^{\prime}}} \sum_{p_{\delta}} T\left(p_{\delta}, p^{\prime}\right) F\left(\hat{p}+p_{\delta}\right) \cdot W\left(p^{\prime}\right) . 
\end{equation}
$$
我们注意到，与标准卷积不同，内核权重有规律地排列，InterpConvs 中的内核权重坐标 p 0 可以灵活设置，甚至可以在训练期间学习。我们提出的 InterpConv 操作有三个关键部分：空间离散的核权重 W、插值函数 T 和归一化项 N。我们首先分别讨论这三个部分，然后介绍完整的算法。

**离散核权重**。在 2D 卷积 [17] 中，一个内核可以表示为一个 n × n × c 张量，其中 n 表示内核大小，c 表示通道数。 在[6, 9]中，一个内核被分成n×n个权重向量，每个权重向量的大小为1×c。 通过这样做，内核权重不再需要有规律地排列，而是可以灵活地放置在 2D 网格上。 在我们的方法中，我们通过为 3D 欧几里德空间中的每个卷积核定义一组核权重向量来进一步改进这个想法。 每个内核权重向量 W(p 0 ) 都有一个 3D 坐标 p 0 来存储其相对于核中心，其权重存储在一个 1×c 向量中，该向量将在训练期间初始化和更新。 矢量坐标 p 0 可以在训练期间固定或更新。 为了简化问题，我们在大多数实验中固定核权重坐标并将它们组织成一个立方体，即如果核权重向量总数为 27，则核权重向量排列在 3 × 3 × 3 3D 规则网格中。我们注意到 这是标准 3 × 3 × 3 离散卷积的类比，而内核权重向量理论上可以放置在 3D 空间中的任意位置。

当我们将核权重向量排列为立方体时，我们定义了两个重要的超参数：核大小 n × n × n 和核长度 l。 空间离散核权重向量的坐标集可以表示为
$$
\begin{equation}
 \Phi=\left\{\left(x^{\prime}, y^{\prime}, z^{\prime}\right) \mid x^{\prime}, y^{\prime}, z^{\prime}=k l,\right.  \left.k \in\left\{-\frac{n-1}{2}, \cdots, \frac{n-1}{2}\right\}\right\} 
\end{equation}
$$
其中 p 0 = (x 0 ,y 0 ,z 0 )。 类似于标准卷积中核大小的定义，核大小 n × n × n ∈ Z 3 表示一个核的每条边上有 n 个核权向量，核权向量的总数为 n 3 。 核长度 l ∈ R 是两个相邻权向量之间的距离。 它决定了欧几里德空间中内核的实际 3D 大小，并被定义为控制感受野，卷积内核可以从中“看到”输入点云。 如果 l 很小，卷积核能够捕获细粒度的局部结构，否则它会编码更多的全局形状信息。





**3.3. Interpolation functions**

在不规则点云上应用离散内核的一个问题是内核权重向量的空间位置通常与输入点不对齐。 天真地将点云光栅化为规则网格 [24, 11] 解决了部分问题，但代价是丢失了局部结构。 在我们的方法中，我们通过采用插值函数来解决这个问题，同时保留所有细粒度结构。 即，我们首先在每个核权重向量附近找到一组输入点，然后将它们的特征插值到核权重向量上进行卷积。 我们提出了两种插值函数：三线性插值和高斯插值。





**ECCV2020_Progressive Point Cloud Deconvolution Generation Network**

在本文中，我们提出了一种有效的点云生成方法，该方法可以从潜在向量生成相同形状的多分辨率点云。 具体来说，我们开发了一种具有**基于学习的双边插值的新型渐进式反卷积网络**。在点云的空间和特征空间中进行基于学习的双边插值，从而可以利用点云的局部几何结构信息。从低分辨率点云开始，通过双边插值和最大池化操作，反卷积网络可以逐步输出高分辨率的局部和全局特征图。 通过连接不同分辨率的局部和全局特征图，我们采用多层感知器作为生成网络来生成多分辨率点云。 为了保持点云不同分辨率的形状一致，我们提出了一种形状保持对抗性损失来训练点云反卷积生成网络。 实验结果证明了我们提出的方法的有效性



**3.1 Progressive deconvolution generation network**

给定一个潜在向量，我们的目标是生成高质量的 3D 点云。   点云生成的一个关键问题是如何利用一维向量生成一组与几何中的 3D 对象一致的 3D 点。   为此，我们为 3D 点云开发了一个特殊的反卷积网络，我们首先使用基于学习的双边插值获得高分辨率特征图，然后应用 MLP 生成局部和全局特征图。 希望生成的局部和全局特征图的融合能够表征高维特征空间中点云的几何结构。

**Learning-based bilateral interpolation** 由于点云的无序和不规则结构，我们无法直接对特征图进行插值操作。 因此，我们需要为特征图上的每个点建立一个邻域来实现插值操作。 在这项工作中，我们简单地使用 k-最近邻 (k-NN) 来构建特征空间中每个点的邻域。 具体来说，给定一个具有 N 个特征向量 $ \boldsymbol{x}_{i} \in \mathbb{R}^{d} $ 的输入，点 i 和 j 之间的相似度定义为：
$$
\begin{equation}
 a_{i, j}=\exp \left(-\beta\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|_{2}^{2}\right) 
\end{equation}
$$
其中 β 在我们的实验中凭经验设置为 β = 1。 如图2（a）和（b）所示，我们可以在特征空间中选择k个具有定义相似度的最近邻点。 并且本文中参数k设置为k=20。

![image-20211110175415785](assets/image-20211110175415785.png)

**Fig.2.** 反卷积网络的过程。 （a）首先，我们定义特征空间中点对之间的相似性。 我们在特征空间中选择 k 个最近邻点 (k-NN)，并在 (b) 中定义相似度。 然后我们在邻域中进行插值以形成（c）中的放大特征图。 最后，我们在 (d) 中应用 MLP 生成新的高维特征图。 请注意，我们可以通过反卷积网络获得双倍的点数。



一旦我们获得了每个点的邻域，我们就可以在其中进行插值。 如图2(c)所示，通过插值，可以将邻域内的k个点生成为特征空间中的2k个点。**线性和双线性插值**等经典插值方法是非学习插值方法，在点云生成过程中无法适应不同类别的3D模型。但**经典插值方法不会同时利用空间和特征空间中每个点的邻域信息**。

为此，我们提出了一种**基于学习的双边插值方法**，该方法利用每个点的邻域的空间坐标和特征来生成高分辨率的特征图。给定点 $ \boldsymbol{p}_{i} \in \mathbb{R}^{3} $ 和其邻域中的 k 个点，我们可以将双边插值公式化为： 
$$
\begin{equation}
 \tilde{\boldsymbol{x}}_{i, l}=\frac{\sum_{j=1}^{k} \theta_{l}\left(\boldsymbol{p}_{i}, \boldsymbol{p}_{j}\right) \psi_{l}\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right) \boldsymbol{x}_{j, l}}{\sum_{j=1}^{k} \theta_{l}\left(\boldsymbol{p}_{i}, \boldsymbol{p}_{j}\right) \psi_{l}\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)} 
\end{equation}\tag{2}
$$
其中 $ \boldsymbol{p}_{i}$ 和 $ \boldsymbol{p}_{j}$ 是 3D 空间坐标，$ \boldsymbol{x}_{i}$ 和 $ \boldsymbol{x}_{j}$ 是 d 维特征向量，$ \theta\left(\boldsymbol{p}_{i}, \boldsymbol{p}_{j}\right) \in \mathbb{R}^{d} $ 和 $ \psi\left(\boldsymbol{p}_{i}, \boldsymbol{p}_{j}\right) \in \mathbb{R}^{d} $  分别是空间和特征空间中的两个嵌入，$ \tilde{\boldsymbol{x}}_{i, l} $ 是插值特征 $ \tilde{\boldsymbol{x}}_{i}, l=1,2, \cdots, d $ 的第 $l$ 个元素。 嵌入 $ \theta\left(\boldsymbol{p}_{i}, \boldsymbol{p}_{j}\right) \in \mathbb{R}^{d} $ 和 $ \psi\left(\boldsymbol{p}_{i}, \boldsymbol{p}_{j}\right) \in \mathbb{R}^{d} $  可以定义为：
$$
\begin{equation}
 \theta\left(\boldsymbol{p}_{i}, \boldsymbol{p}_{j}\right)=\operatorname{ReLU}\left(\boldsymbol{W}_{\theta, j}^{\top}\left(\boldsymbol{p}_{i}-\boldsymbol{p}_{j}\right)\right), \quad \psi\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\operatorname{ReLU}\left(\boldsymbol{W}_{\psi, j}^{\top}\left(\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right)\right) 
\end{equation}
$$
其中 ReLU 是激活函数，$ \boldsymbol{W}_{\theta, j} \in \mathbb{R}^{3 \times d} $ 和 $ \boldsymbol{W}_{\psi, j} \in \mathbb{R}^{d \times d} $ 是要学习的权重。 基于点 $p_i$ 和 $p_j$ 、$p_i − p_j$ 和 $x_i − x_j$ 之间的差异，嵌入 $θ(p_i ,p_j )$ 和 $ψ (x_i ,x_j )$ 可以分别编码点 $p_i$ 在空间和特征空间中的局部结构信息。 值得注意的是，在**Eq. 2**中，采用channel-wise双边插值。 如图 3 所示，新的插值特征 $ \tilde{\boldsymbol{x}}_{i} $ 可以从 $ \boldsymbol{x}_{i} $ 的邻域中获得，具有双边权重。 对于每个点，我们在 k 邻域中执行双边插值以生成新的 k 点。因此，我们可以获得高分辨率的特征图，其中每个点的邻域包含 2k 个点。

![image-20211110210625768](assets/image-20211110210625768.png)

插值后，我们然后在放大的特征图上应用卷积。 对于每个点，我们根据距离将 2k 个点的邻域划分为两个区域。 如图2(c)所示，最近的k个点属于第一个区域，其余的作为第二个区域。 与PointNet [26]类似，我们首先使用多层感知器生成高维特征图，然后使用最大池化操作从两个区域获取两个插值点的局部特征。 如图 2 (d) 所示，我们可以通过反卷积网络将输入的点数加倍以生成高分辨率的局部特征图 X local 。我们还使用最大池化操作来提取全局特征 点云。 通过将全局特征复制 N 次，其中 N 是点数，我们可以获得高分辨率的全局特征图 X global 。 然后我们将局部特征图 X local 和全局特征图 X global 连接起来，得到反卷积网络的输出 X c = [X local ;X global ]。 因此，输出X c 不仅可以表征点云的局部几何结构，还可以在点云生成过程中捕捉点云的全局形状。

![image-20211110201554840](assets/image-20211110201554840.png)

**Fig. 3** 基于学习的双边插值方法示意图。中心点 $x_i$ 附近的点被着色。 我们通过考虑邻域中点的局部几何特征来插入新点。$ \boldsymbol{W}_{\theta, j} $ 和 $ \boldsymbol{W}_{\psi, j}, j=1,2,3,4 $ 是要学习的空间和特征空间中的权重。







x**ICCV2021_Interpolation-Aware Padding for 3D Sparse Convolutional Neural Networks**

这里基于稀疏体素的 CNN 的填充与传统的 2D 图像填充有着根本的不同。我们在空体素周围填充额外的体素作为输入。最初，填充体素的输入特征设置为零，然后所有体素特征都由 CNN 操作动态生成。 对于图像上的 2D 卷积，由于卷积核的形状固定，靠近图像边界的像素需要虚拟填充像素。 并且填充像素的特征直接设置为零（即零填充）或图像边界上反射像素处的特征图（即反射填充）。

我们的目标是在非空体素周围填充额外的体素，以便在对每个点进行三线性插值时所有八个最近的相邻体素都存在。  3D 规则网格 [29] 内的三线性插值可以写为：



其中 i,j,k ∈ {0,1} aretheindicesoftheeightgridcorners, vol ijk 是被查询点和与索引为 ijk 的角对角相对的角所包围的部分体积，f ijk 是角 ijk 上关联的特征向量 , I ijk ∈ {0,1} 表示对应的体素是否存在。 在下文中，我们检查了两种基线填充方案，并提出了我们的插值感知填充方案，以平衡网络性能和运行时内存成本。





**Interpolation-aware padding** 使用eq.1 在非空体素 v 内的 3D 点 p = (x,y,z) 上插入 CNN 特征。 我们将插值中涉及的所有空体素填充到 CNN 计算中。 我们假设指定了输入的 3D 边界框，并且具有最小 x、y、z 坐标的角由 p o = (x o ,y o ,z o ) 表示。 插值涉及的所有八个体素的体素索引计算如下：





----

# Code

## Metric

```shell
git clone https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
git clone https://github.com/daerduoCarey/PyTorchEMD.git
```









## evaluate

### **Point Cloud Utils** (pcu)

- A Python library for common tasks on 3D point clouds. 

依赖: **numpyeigen、Embree、manifold**

```shell
git clone https://github.com/fwilliams/point-cloud-utils.git

python setup.py install
```





### Manifold

将任何三角形网格转换为 Watertight 流行





### Embree

光线跟踪内核库。高性能光线跟踪内核3.13.1 %英特尔公司



### **numpyeigen**

```shell
git clone https://github.com/fwilliams/numpyeigen.git

mkdir build
cd build
cmake ..
make 
make install
```



**numpyEigen** - Fast zero-overhead bindings between NumPy and Eigen

NumpyEigen可以轻松地将NumPy密集数组和SciPy稀疏矩阵透明地转换为Eigen，同时零复制开销，同时利用Eigen的表达式模板系统获得最大性能。

Eigen 是一个C++数值线性代数库。它使用表达式模板为给定的一组输入类型选择最快的数字算法。NumPy和SciPy是在Python中公开快速数值例程的库。 

由于Python中的类型信息只在运行时可用，所以要编写接受多个NumPy或SciPy类型、零复制开销并且可以利用Eigen中最快的数字内核的绑定并不容易。NumpyEigen透明地生成绑定，完成上述所有工作，在编译时向C++代码公开numpy类型信息。



NumpyEigen uses [pybind11](https://github.com/pybind/pybind11) under the hood which is included automatically by the cmake project.

**依赖 pybind11**





### **pybind 11**

```shell
git clone -b numpy-hacks https://github.com/fwilliams/pybind11.git
python setup.py install
```



c++ 11和Python之间的无缝可操作性

pybind11是一个轻量级的仅头库，它在Python中公开C++类型，反之亦然，主要用于创建现有C++代码的Python绑定。

它的目标和语法类似于优秀的Boost。大卫·亚伯拉罕斯的Python库:通过使用编译时内省推断类型信息来最小化传统扩展模块中的样板代码。 



**Boost.Python**

一个C++库，支持C++和Python编程语言之间的无缝互操作性。







---

## train

3 7 8 56    0.25附近   resflow.forward + interp + resflow.backward

67 0.4附近 resflow.forward





## validation_step()

```python
fowward
	compute_loss
    	network.forward
        	_logdetgrad
            	poisson
                	lamb=self.lamb.item()
                not self.exact_trace
                	basic_logdet_estimator
```







## 调试优化

### 每次迭代显存增加2.2Mb

```python
96 * (8, 3, 256) float32

8*3*256 = 6144

6144*96 = 589 824

96 = 8 * 12
96 = 16 * 6

96 = 3 * 32 
   = 3 * 2 * 16 
   = 3 * 2 * 2 * 8
   = 3 * 2 * 2 * 2 * 4
   = 3 * 2 * 2 * 2 * 2 * 2
```





### Profiler 结果分析



```python
        with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
            outputs = self(xyz_sparse, upratio)
        print(prof.table())
```





```python
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            aten::transpose         0.00%      52.270us         0.00%      68.161us      68.161us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%      15.891us         0.00%      15.891us      15.891us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::zeros         0.00%      65.275us         0.00%     143.925us     143.925us      18.250us         0.00%      79.250us      79.250us             1  
                                aten::empty         0.00%      14.920us         0.00%      14.920us      14.920us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::zero_         0.00%      36.012us         0.00%      63.730us      63.730us      35.250us         0.00%      61.000us      61.000us             1  
                                aten::fill_         0.00%      27.718us         0.00%      27.718us      27.718us      25.750us         0.00%      25.750us      25.750us             1  
                                   aten::to         0.00%      42.735us         0.01%     424.923us     424.923us     374.500us         0.00%     425.500us     425.500us             1  
                        aten::empty_strided         0.00%      51.490us         0.00%      51.490us      51.490us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%     330.698us         0.00%     330.698us     330.698us      51.000us         0.00%      51.000us      51.000us             1  
                                aten::zeros         0.00%      28.294us         0.00%      70.929us      70.929us      37.000us         0.00%      69.750us      69.750us             1  
                                aten::empty         0.00%       9.133us         0.00%       9.133us       9.133us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::zero_         0.00%      20.113us         0.00%      33.502us      33.502us      19.250us         0.00%      32.750us      32.750us             1  
                                aten::fill_         0.00%      13.389us         0.00%      13.389us      13.389us      13.500us         0.00%      13.500us      13.500us             1  
                                   aten::to         0.00%      34.819us         0.00%     174.149us     174.149us      62.500us         0.00%     175.500us     175.500us             1  
                        aten::empty_strided         0.00%      27.730us         0.00%      27.730us      27.730us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%     111.600us         0.00%     111.600us     111.600us     113.000us         0.00%     113.000us     113.000us             1  
                                aten::zeros         0.00%      26.785us         0.00%      67.128us      67.128us      35.000us         0.00%      66.750us      66.750us             1  
                                aten::empty         0.00%       8.236us         0.00%       8.236us       8.236us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::zero_         0.00%      19.287us         0.00%      32.107us      32.107us      19.250us         0.00%      31.750us      31.750us             1  
                                aten::fill_         0.00%      12.820us         0.00%      12.820us      12.820us      12.500us         0.00%      12.500us      12.500us             1  
                                   aten::to         0.00%      27.560us         0.00%     119.552us     119.552us      51.250us         0.00%     117.750us     117.750us             1  
                        aten::empty_strided         0.00%      23.261us         0.00%      23.261us      23.261us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      68.731us         0.00%      68.731us      68.731us      66.500us         0.00%      66.500us      66.500us             1  
                                aten::zeros         0.00%      40.482us         0.00%      95.209us      95.209us      47.000us         0.00%      91.250us      91.250us             1  
                                aten::empty         0.00%       9.517us         0.00%       9.517us       9.517us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::zero_         0.00%      20.275us         0.00%      45.210us      45.210us      19.250us         0.00%      44.250us      44.250us             1  
                                aten::fill_         0.00%      24.935us         0.00%      24.935us      24.935us      25.000us         0.00%      25.000us      25.000us             1  
                                   aten::to         0.00%      27.212us         0.00%     119.590us     119.590us      51.250us         0.00%     119.750us     119.750us             1  
                        aten::empty_strided         0.00%      22.941us         0.00%      22.941us      22.941us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      69.437us         0.00%      69.437us      69.437us      68.500us         0.00%      68.500us      68.500us             1  
                                 aten::view         0.00%      25.287us         0.00%      25.287us      25.287us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::is_nonzero         0.00%      38.531us         0.00%     137.551us     137.551us       6.500us         0.00%      79.500us      79.500us             1  
                                 aten::item         0.00%      20.594us         0.00%      99.020us      99.020us       6.500us         0.00%      73.000us      73.000us             1  
                  aten::_local_scalar_dense         0.00%      78.426us         0.00%      78.426us      78.426us      66.500us         0.00%      66.500us      66.500us             1  
                            aten::transpose         0.00%      17.842us         0.00%      24.244us      24.244us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%       6.402us         0.00%       6.402us       6.402us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::contiguous         0.00%      27.235us         0.00%     168.694us     168.694us      53.000us         0.00%     164.500us     164.500us             1  
                           aten::empty_like         0.00%       9.497us         0.00%      29.071us      29.071us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::empty         0.00%      19.574us         0.00%      19.574us      19.574us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%     112.388us         0.00%     112.388us     112.388us     111.500us         0.00%     111.500us     111.500us             1  
                                 aten::view         0.00%      11.307us         0.00%      11.307us      11.307us       0.000us         0.00%       0.000us       0.000us             1  
                                 aten::mean         0.00%     141.131us         0.00%     168.363us     168.363us     356.500us         0.00%     356.500us     356.500us             1  
                                aten::empty         0.00%      16.793us         0.00%      16.793us      16.793us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%      10.439us         0.00%      10.439us      10.439us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::var         0.00%      81.389us         0.00%     108.387us     108.387us      18.500us         0.00%      18.500us      18.500us             1  
                                aten::empty         0.00%       4.695us         0.00%       4.695us       4.695us       0.000us         0.00%       0.000us       0.000us             1  
                              aten::resize_         0.00%      16.301us         0.00%      16.301us      16.301us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%       6.002us         0.00%       6.002us       6.002us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::empty         0.00%       7.860us         0.00%       7.860us       7.860us       0.000us         0.00%       0.000us       0.000us             1  
                                   aten::to         0.00%       7.668us         0.00%       7.668us       7.668us       1.500us         0.00%       1.500us       1.500us             1  
                              aten::detach_         0.00%      10.701us         0.00%      17.971us      17.971us       2.500us         0.00%       3.500us       3.500us             1  
                                    detach_         0.00%       7.270us         0.00%       7.270us       7.270us       1.000us         0.00%       1.000us       1.000us             1  
                                   aten::to         0.00%      17.117us         0.00%      91.064us      91.064us       6.000us         0.00%      51.000us      51.000us             1  
                        aten::empty_strided         0.00%      13.756us         0.00%      13.756us      13.756us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      60.191us         0.00%      60.191us      60.191us      45.000us         0.00%      45.000us      45.000us             1  
                                  aten::max         0.00%      21.698us         0.00%      96.545us      96.545us      21.000us         0.00%      95.500us      95.500us             1  
                              aten::maximum         0.00%      68.552us         0.00%      74.847us      74.847us      74.500us         0.00%      74.500us      74.500us             1  
                                aten::empty         0.00%       6.295us         0.00%       6.295us       6.295us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::neg         0.00%      35.640us         0.00%      92.150us      92.150us      35.500us         0.00%      89.000us      89.000us             1  
                                aten::empty         0.00%       3.760us         0.00%       3.760us       3.760us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::neg         0.00%      46.779us         0.00%      52.750us      52.750us      53.500us         0.00%      53.500us      53.500us             1  
                              aten::resize_         0.00%       5.971us         0.00%       5.971us       5.971us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      45.776us         0.00%      45.776us      45.776us      45.000us         0.00%      45.000us      45.000us             1  
                                  aten::log         0.00%      36.556us         0.00%      80.103us      80.103us      38.500us         0.00%      79.500us      79.500us             1  
                                aten::empty         0.00%       3.610us         0.00%       3.610us       3.610us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::log         0.00%      34.864us         0.00%      39.937us      39.937us      41.000us         0.00%      41.000us      41.000us             1  
                              aten::resize_         0.00%       5.073us         0.00%       5.073us       5.073us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::mul         0.00%      63.171us         0.00%      69.545us      69.545us      68.500us         0.00%      68.500us      68.500us             1  
                                aten::empty         0.00%       6.374us         0.00%       6.374us       6.374us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      41.357us         0.00%      41.357us      41.357us      41.000us         0.00%      41.000us      41.000us             1  
                                aten::fill_         0.00%      47.567us         0.00%      47.567us      47.567us      47.500us         0.00%      47.500us      47.500us             1  
                                 aten::view         0.00%      25.261us         0.00%      25.261us      25.261us       0.000us         0.00%       0.000us       0.000us             1  
                            aten::expand_as         0.00%      12.411us         0.00%      31.113us      31.113us      30.000us         0.00%      30.000us      30.000us             1  
                               aten::expand         0.00%      15.444us         0.00%      18.702us      18.702us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%       3.258us         0.00%       3.258us       3.258us       0.000us         0.00%       0.000us       0.000us             1  
                                 aten::view         0.00%      12.328us         0.00%      12.328us      12.328us       0.000us         0.00%       0.000us       0.000us             1  
                            aten::expand_as         0.00%       9.872us         0.00%      29.046us      29.046us      28.000us         0.00%      28.000us      28.000us             1  
                               aten::expand         0.00%      17.097us         0.00%      19.174us      19.174us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%       2.077us         0.00%       2.077us       2.077us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::add         0.00%      70.829us         0.00%      80.952us      80.952us      78.500us         0.00%      78.500us      78.500us             1  
                        aten::empty_strided         0.00%      10.123us         0.00%      10.123us      10.123us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::exp         0.00%      46.743us         0.00%     103.599us     103.599us      43.500us         0.00%     103.000us     103.000us             1  
                                aten::empty         0.00%       3.347us         0.00%       3.347us       3.347us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::exp         0.00%      47.609us         0.00%      53.509us      53.509us      59.500us         0.00%      59.500us      59.500us             1  
                              aten::resize_         0.00%       5.900us         0.00%       5.900us       5.900us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::mul         0.00%      82.692us         0.00%      91.554us      91.554us      92.000us         0.00%      92.000us      92.000us             1  
                        aten::empty_strided         0.00%       8.862us         0.00%       8.862us       8.862us       0.000us         0.00%       0.000us       0.000us             1  
                                 aten::view         0.00%      15.224us         0.00%      15.224us      15.224us       0.000us         0.00%       0.000us       0.000us             1  
                               aten::expand         0.00%      12.741us         0.00%      15.367us      15.367us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%       2.626us         0.00%       2.626us       2.626us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::contiguous         0.00%      23.662us         0.00%      88.583us      88.583us      37.500us         0.00%      86.000us      86.000us             1  
                           aten::empty_like         0.00%       6.043us         0.00%      16.424us      16.424us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::empty         0.00%      10.381us         0.00%      10.381us      10.381us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      48.497us         0.00%      48.497us      48.497us      48.500us         0.00%      48.500us      48.500us             1  
                                 aten::view         0.00%      12.953us         0.00%      12.953us      12.953us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::sum         0.00%      86.695us         0.00%      93.685us      93.685us      91.500us         0.00%      91.500us      91.500us             1  
                                aten::empty         0.00%       6.990us         0.00%       6.990us       6.990us       0.000us         0.00%       0.000us       0.000us             1  
                                 aten::rsub         0.00%      79.015us         0.00%      85.947us      85.947us       8.500us         0.00%       8.500us       8.500us             1  
                                aten::empty         0.00%       6.932us         0.00%       6.932us       6.932us       0.000us         0.00%       0.000us       0.000us             1  
                    aten::is_floating_point         0.00%       7.082us         0.00%       7.082us       7.082us       1.500us         0.00%       1.500us       1.500us             1  
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 7.867s
CUDA time total: 7.911s
```







```PYTHON
251.7 	648.4 	1045.2	1441.8
		396.7	396.8	396.6
    
251.7	660.8	1057.5	1466.4
		409.1	396.7	408.9

251.7	685.4	1082.1	1491.0
		433.7	396.7	408.9
    
251.7	685.4	1094.4	1540.2
		433.7	409		445.8
    
251.7	710.0	1131.3	1589.4
		458.3	421.3	458.1

```

