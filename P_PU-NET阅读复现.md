# PU-NET复现

## official

```
TF1.3 Python 2.7
```



## mine

```bash
TF1.13 python3.6 

docker pull tensorflow/tensorflow:1.13.2-gpu-py3

docker run --runtime=nvidia --rm -it -v /home/duan/windows/udata:/home/data/ -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE --name samplenetreg tensorflow/tensorflow:1.13.2-gpu-py3


docker commit -m "xxxx" -p 30ad6f281301 tensorflow/tensorflow:1.13.2-gpu-py3
```



## error

1. python2 to python3

```bash
pip install 2to3
2to3 -W main.p
-W 写入输出文件，不需要对该文件进行任何更改
https://docs.python.org/3/library/2to3.html
```

2. `Building wheel for opencv-python (PEP 517) … -` runs forever

```
pip install --upgrade pip
```

3. cv2 ImportError: libGL.so.1: cannot open shared object file: No such file or directory

```
apt install libgl1-mesa-glx
```

4. TypeError: a bytes-like object is required, not 'str'

```
问题出在python3.5和Python2.7在套接字返回值解码上有区别:
list(set([item.split('/')[-1].split('_')[0] for item in name]))

item>>>>>>>>>>>>>>str(item)

list(set([str(item).split('/')[-1].split('_')[0] for item in name]))
```

5. No registered '_CopyFromGpuToHost' OpKernel for CPU devices compatible with node

```
显存不够，减小batch_size
```





# PU-NET

​	Point Cloud Upsampling Network

## Abstart

​	由于数据的稀疏性和不规则性，使用深度网络学习和分析3D点云具有挑战性。 在本文中，我们提出了一种**数据驱动**的点云上采样技术。 

​	The key idea is to learn multi-level features per point and expand the point set via a multi branch convolution unit implicitly in feature space.  核心思想是学习每个点的多层次特征，然后利用不同的卷积分支在特征空间中进行扩充， 

​	 The expanded feature is then split to a multitude of features, which are then reconstructed to an upsampled point set. 扩展的特征会分解为多个特征，然后将其重构为上采样点集。

​	我们的网络是在patch级别实现的，并且使用了联合的损失函数使得上采样后的点在潜在的曲面上分布一致。

​	我们使用合成和扫描数据进行了各种实验，以评估我们的方法，并证明了其相对于某些基准方法和基于优化的方法的优越性。结果表明，我们的上采样点具有更好的均匀性，并且更靠近下层表面。



## 1. Introduction

​	点云是一种基本的3D表示形式，由于各种深度扫描设备的普及而引起了越来越多的关注。 最近，开创性的工作[29，30，18]开始探索通过深度网络来理解点云的可能性，这些网络可以理解几何并识别3D结构。 在这些作品中，深度网络直接从原始3D点坐标中提取特征，而无需使用传统特征，例如法线和曲率。 这些作品为3D对象分类和语义场景分割提供了令人印象深刻的结果。

​	在这项工作中，我们对一个上采样问题感兴趣：给定一组点，通过学习训练数据集的几何来生成更密集的点集来描述基础几何。这个上采样问题在本质上与图像超分辨率问题相似[33，20]。但是，处理3D点而不是2D像素网格带来了新的挑战。

​	首先，与以规则网格表示的图像空间不同，点云没有任何空间顺序和规则结构。其次，生成的点应描述潜在目标对象的基础几何形状，这意味着它们应大致位于目标对象表面上。第三，生成的点应该具备信息的，并且不应杂乱无章。话虽如此，生成的输出点集在目标对象表面上应该更加均匀。因此，输入点之间的简单插值无法产生令人满意的结果。

- [ ] ==针对无序、无拓扑结构== 

- [ ] ==点应该尽量生成在目标表面上（不应在内部）==

- [ ] ==不应杂乱，应该尽量均匀==

​	为了应对上述挑战，我们提出了一种数据驱动的点云上采样网络。我们的网络是在 patch-level，具有联合损失函数，该函数鼓励上采样的点以均匀分布的形式保留在基础表面上。关键思想是学习每个点的多级特征，然后通过隐含在特征空间中的多分支卷积单元扩展点集。然后将扩展的特征拆分为多个特征，然后将其重构为上采样点集。

​	我们的网络PU-Net从3D模型中学习 point-based patches 的几何语义，然后将学到的知识应用于给定点云的上采样。应该注意的是，与先前为3D点集设计的基于网络的方法[29、30、18]不同，我们网络中输入和输出点的数量是不同的。

​	我们制定两个度量标准，即 distribution uniformity 和 distance deviation from underlying surfaces，以定量评估上采样的点集，并使用各种合成的和实际扫描的数据测试我们的方法。我们还评估了该方法的性能，并将其与基准和基于最优化的最新方法进行了比较。结果表明，我们的上采样点具有更好的均匀性，并且更靠近下层表面。

### Related work

#### optimization-based methods

​	Alexa等人[2]通过对局部切线空间中Voronoi图的顶点处的点进行插值来对一个点集进行上采样。 利普曼（Lipman）等人 [24]提出了一种基于L-1中位数的用于点重采样和表面重构的局部最优投影（LOP）算子。即使输入点集包含噪声和异常值，操作员也可以正常工作。黄[14]提出了一种改进的加权LOP来解决点集密度问题。

​	尽管这些工作已显示出良好的结果，但他们坚决假定其下表面是光滑的，从而限制了该方法的范围。 然后，黄[15]通过首先从边缘重采样，然后逐渐接近边缘和拐角，引入了一种边缘感知点集重采样方法。但是，其结果的质量在很大程度上取决于法线在给定点的准确性和仔细的参数调整。值得一提的是吴[35]提出了一种深点表示方法，以在一个连贯的步骤中融合合并和完成。 由于其主要重点是填充大孔，因此并未强制执行全局平滑度，因此该方法对大噪声敏感。 总体而言，上述方法不是数据驱动的，因此严重依赖先验条件。

#### deep-learning-based methods

​	点云中的点没有任何特定的顺序，也没有遵循任何规则的网格结构，因此，只有最近的一些著作采用深度学习模型来直接处理点云。现有的大多数工作都将点云转换为其他3D表示形式，例如体积网格[27、36、31、6]和几何图[3、26]以进行处理。  Qi[29，30]首先介绍了用于点云分类和分割的深度学习网络； 特别是，PointNet ++使用分层特征学习体系结构来捕获本地和全局几何上下文。

​	随后，针对点云的高级分析问题，提出了许多其他网络[18、13、21、34、28]。但是，它们都集中于点云的全局或中级属性。 在另一本书中，Guerrero等人[10]开发了一个网络来估计点云中的局部形状属性，包括法线和曲率。其他相关网络则专注于从2D图像进行3D重建[8、23、9]。 据我们所知，目前尚无专注于点云上采样的工作。



----

## 2. Architecture

![1603181094050](./assets/1603181094050.png)

**Fig. 1 PU-Net的网络架构.** 输入具有 N 个点，而输出具有 rN 个点，其中 r 是上采样率。$C_i$，$\tilde{C}$ 和 $\tilde{C}_{i}$ 表示特征通道数目。我们通过插值为原始的 N 个点恢复不同的级别特征，并通过卷积将所有级别特征缩减为固定维度 $C$ 。点云特征集成（embedding）模块中的红色表示原来的和逐步降采样的点，绿色显示恢复的特征。 我们联合使用重建损失函数和互斥损失函数用来端到端地训练上采样网络。

​	

​	给定一个具有不均匀分布的点坐标的3D点云，我们的网络旨在输出一个更密集的点云，该点云遵循目标对象的基础表面同时分布均匀。

​	PU-Net有四个组件：块提取 Patch Extraction、点特征集成 Point Feature Embedding、特征扩张 Feature Expansion 和坐标重建 Coordinate Reconstruction。



​	首先，我们从给定的一组先验3D模型中提取具有不同尺度和分布的**点块**（第2.1节）。

​	然后，点特征集成组件通过**分层特征学习**和**多级特征聚合**将原始3D坐标映射到特征空间（第2.2节）。

​	之后，我们使用特征扩展组件**扩展特征的数量**（第2.3节）

​	并通过坐标重建组件中的一系列全连接层重建输出点云的3D坐标（第2.4节）。

---

### Patch Extraction

​	我们采集一组3D物体作为训练的先验信息。 这些物体涵盖了各种各样的形状，从光滑的表面到具有**尖锐的边缘**和**角落的**形状。本质上，要使我们的网络对点云进行上采样，它应该从对象中学习局部几何模式。这促使我们采取 patch-based 的方法来训练网络并学习几何语义。 

​	详细地说，我们随机选择这些物体表面上的 $M$ 个点。从每个选定的点开始，我们在物体上生成一个曲面块，该块上的任何点与曲面上的选定点的距离都 **小于** 给定的测地线**距离**（$d$）。 然后，我们使用 Poisson 盘采样在每个块上随机生成 $N$ 个点，作为块上的参考 G.T. 分布。 在我们的上采样任务中，局部和全局上下文都有助于平稳而均匀的输出。 因此，我们用不同的大小设置 $d$ ，以便我们可以以不同的尺度和密度提取先前物体上的点块。



geodesic distance 三维空间中两点的最短路径

---

### Point Feature Embedding

​	神经网络浅层特征一般反映着局部的小尺度特征，为了更好的上采样结果，采用skip-connection来聚集不同层的特征。

​	由于在分层特征提取中逐步对每个小块的输入进行二次下采样，通过PointNet++中的插值方法，首先从下采样的点特征中上采样恢复所有原始点的特征N×Cl ，从而连接每个级别的点特征。具体而言，插值点x在l水平上的特征通过以下方式计算:



​	为了从块中学习局部和全局几何背景，我们考虑以下两种特征学习策略，它们的优势是相辅相成的：

#### Hierarchical feature learning 层次特征学习

​	渐进地捕获越来越多的层次结构特征已被证明是提取局部和全局特征的有效策略。 因此，我们采用PointNet++ [30]中最近提出的分层特征学习机制作为网络中最重要的部分。 

​	为了采用分层特征学习进行点云上采样，我们具体在每个级别使用相对较小的分组半径，因为生成新点通常比[30]中的高级识别任务涉及更多的局部上下文。

#### Multi-level feature aggregation 多级特征聚合

​	网络中的较低层通常对应于较小规模的局部特征，反之亦然。为了获得更好的上采样结果，我们应该最佳地汇总不同级别的要素。以前的一些工作采用残差连接进行级联的多级特征聚合[25、32、30]。 但是，这种自上而下的传播方式对于汇总我们的上采样问题中的特征并不是非常有效。 因此，我们建议直接组合不同级别的特征，并让网络了解每个级别的重要性[11、38、12]。

​	由于在每个块上设置的输入点（请参见图1中的点特征嵌入）在层次特征提取中逐步进行了二次采样，因此我们首先通过PointNet++中的插值方法从下采样后的点特征中恢复所有原始点的特征，从而将每个级别的点特征连接起来。具体来说，级别中的插值点x的特征可通过以下公式计算：
$$
f^{(\ell)}(x)=\frac{\sum_{i=1}^{3} w_{i}(x) f^{(\ell)}\left(x_{i}\right)}{\sum_{i=1}^{3} w_{i}(x)}
$$
​	其中 $w_{i}(x)=1 / d\left(x, x_{i}\right)$ 是距离权重的倒数，而 $x_i，x_2，x_3$是级别 $l$ 中 $x$ 的三个最近邻居。 然后，我们使用1×1卷积将不同级别的插值特征缩减为相同的维数，即 $C$。最后，我们将每个级别的特征连接为嵌入点特征 $f$ 。

​	

---

### Feature Expansion

​	在Point Feature Embedding之后，我们扩展了特征空间中的特征数量，这相当于扩展点的数量，因为点和特征是可以互换的。假设 $f$ 的维数是 $N×\hat{C}$，$N$ 是输入点的数目，$\hat{C}$ 是级联集成特征的特征维数。

​	特征扩展操作将输出维数为 $rN×\hat{C}_2$ 的特征 $f'$，其中 $r$ 是上采样率，$\hat{C}_2$ 是新的特征维数。本质上，这类似于图像相关任务中的特征上采样，这可以通过**反卷积或插值**来完成。然而，由于点的非规则性和无序特性，将这些操作应用于点云并不容易。

​	因此，提出了一种基于子像素卷积层的有效特征扩展操作：
$$
f^{\prime}=\mathcal{R S}\left(\left[\mathcal{C}_{1}^{2}\left(\mathcal{C}_{1}^{1}(f)\right), \ldots, \mathcal{C}_{r}^{2}\left(\mathcal{C}_{r}^{1}(f)\right)\right]\right)
$$
​	其中 $\mathcal{C}_{i}^{1}(\cdot)$ 和 $\mathcal{C}_{i}^{2}(\cdot)$ 是两组分开的 $1\times1$ 卷积。即通过两次卷积操作 $\mathcal{C}_{i}^{1}(\cdot)$ 和 $\mathcal{C}_{i}^{2}(\cdot)$ 将 $N×\hat{C}$ 先后变成 $r$ 个 $N×\hat{C}_1$ 的特征层，再变成 $r$ 个 $N×\hat{C}_2$ 的特征层，最后拼接成 $rN\times \hat{C}_2$ 的特征图。最后通过$\mathcal{R S}(\cdot)$ 即Reshape的操作，将张量尺寸由 $N\times r\hat{C}_2$ 转换为 $rN\times \hat{C}_2$。

​	我们强调 **嵌入空间中的特征** 已经通过 **有效的多级特征聚集** 从邻域封装了 has already encapsulated **相对空间信息**，因此在执行此特征扩展操作时，我们不需要明确考虑空间信息。

​	值得一提的是，从每个集合中的第一卷积  $\mathcal{C}_{i}^{1}(\cdot)$ 生成的 $r$ 个特征集合具有高相关性，这将导致最终重建的3D点彼此过于接近。因此，我们进一步为每个特征集添加另一个卷积(具有单独的权重)。由于我们训练网络学习个 $r$ 特征集的  $r$ 个不同卷积，这些新特征可以包含更多样的信息，从而减少它们的相关性。这种特征扩展操作可以通过对  $r$ 个特征集应用分离的卷积来实现。它也可以通过计算效率更高的分组卷积来实现。

---

### Coordinate Reconstruction

​	在本部分中，我们将从扩展的特征中以 $r N \times \tilde{C}_{2}$ 的大小重建输出点的3D坐标。 具体来说，我们通过每个点的特征上的一系列全连接层对3D坐标进行回归，最终输出上采样点维度为 $r N \times 3$。



---

## 3. End-to-End Network Training

### 3.1. Training Data Generation

​	Point cloud upsampling is an ill-posed problem due to the uncertainty or ambiguity of upsampled point clouds. Given a sparse input point cloud, there are many feasible output point distributions. Therefore, we do not have the notion of “correct pairs” of input and ground truth.  由于上采样点云的不确定性或歧义性，点云上采样是一个不适定的问题。 给定稀疏的输入点云，存在许多可行的输出点分布。 因此，我们没有输入和 G.T. 的“正确对”的概念。

​	To alleviate this problem, we propose an on-the-ﬂy input generation scheme. Speciﬁcally, the referenced ground truth point distribution of a training patch is ﬁxed, whereas the input points are randomly sampled from the ground truth point set with a downsampling rate of r at each training epoch.  为了缓解此问题，我们提出了一种即时输入生成方案。具体而言，固定了训练 patch 的参考 G.T. 点分布，而输入点是在每个训练epoch从 G.T. 点集以 $r$ 的下采样率随机采样的。

​	Intuitively, this scheme is equivalent to simulating many feasible output point distributions for a given sparse input point distribution. Additionally, this scheme can further enlarge the training dataset, allowing us to depend on a relatively small dataset for training.  直观上，此方案等效于针对给定的稀疏输入点分布模拟许多可行的输出点分布。 另外，该方案可以进一步扩大训练数据集，从而允许我们相对较小的数据集进行训练。

----

### 3.2. Joint Loss Function 

​	We propose a novel joint loss function to train the network in an end-to-end fashion. As we mentioned earlier, thefunctionshouldencouragethegeneratedpointstobelocated on the underlying object surfaces in a more uniform distribution. Therefore, we design a joint loss function that combines the **reconstruction loss** and **repulsion loss**. 我们提出了一种新颖的联合损失函数，以端到端的方式训练网络。 如前所述，该功能应鼓励生成的点以更均匀的分布位于基础对象表面上。 因此，我们设计了一个结合重建损失和排斥损失的联合损失函数。

#### Reconstruction loss

​	To put points on the underlying object surfaces, we propose to use the Earth Mover’s distance (EMD) [8] as our reconstruction loss to evaluate the similarity between the predicted point cloud Sp ⊆ R3 and the referenced ground truth point cloud

​	为了将点放置在的物体下表面上，以评估预测点云与参考G.T.点云之间的相似性

​	EMD  Earth Mover’s distance 

​	Actually, Chamfer Distance (CD) is another candidate for evaluating the similarity between two point sets. However,comparedwithCD,EMDcanbettercapturetheshape (see [8] for more details) to encourage the output points to be located close to the underlying object surfaces. Hence, we choose to use EMD in our reconstruction loss. 实际上，倒角距离（CD）是评估两个点集之间相似度的另一个候选方法。 但是，EMD可以更好地捕获形状（有关更多详细信息，请参见[8]），以鼓励将输出点放置在靠近基础对象表面的位置。 因此，我们选择在重建损失中使用EMD。

---

#### Replusion loss

​	尽管使用重建损失进行训练可以在下面的对象表面上生成点，但是生成的点倾向于位于原始点附近。 为了更均匀地分布生成的点，我们设计了斥力损耗



---

## 4. Experiments

### 4.1. Datasets



------

### 4.2. Implementation Details



---

### 4.3. Evaluation Metric 



---

### 4.4. Comparisons with Other Methods





---

### 4.5. Architecture Design Analysis



---

### 4.6. More Experiments 



---

## 5. Conclusion



















