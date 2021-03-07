# FoldingNet

FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation





## Abstract

​	In this work, a novel end-to-end deep auto-encoder is proposed to address unsupervised learning challenges on point clouds. 

​	On the encoder side, a graph-based enhancement is enforced to promote local structures on top of PointNet. 

​	Then, a novel folding-based decoder deforms a canonical 2D grid onto the underlying 3D object surface of a point cloud, achieving low reconstruction errors even for objects with delicate structures. 

​	The proposed decoder only uses about 7% parameters of a decoder with fully-connected neural networks, yet leads to a more discriminative representation that achieves higher linear SVM classiﬁcation accuracy than the benchmark. 

​	In addition, the proposed decoder structure is shown, in theory, to be a generic architecture that is able to reconstruct an arbitrary point cloud from a 2D grid. 

​	在这项工作中，提出了一种新颖的端到端深度自动编码方案，以解决点云上无监督的学习挑战。 

​	在编码器方面，基于图的增强被强制以提升PointNet之上的局部结构。 

​	然后，一种新颖的 folding-based 的解码器将规范的2D网格变形到点云的基础3D对象表面上，即使对于具有精细结构的对象，也实现了较低的重建误差。 

​	所提出的解码器仅使用具有全连接的神经网络的解码器的大约7％的参数，但会导致更具判别性的表示形式，从而实现比基准更高的线性SVM分类准确率。 

​	此外，理论上将提出的解码器结构显示为一种通用体系结构，该体系结构能够从2D网格重建任意点云。 



![1614604444799](assets/1614604444799.png)

**Fig. 1** FoldingNet体系结构。 graph-layers 是第2.1节（2）中提到的graph-based的最大池化层。 第一个和第二个 folding 都通过三层感知器将 codeword 连接到特征向量。每个感知器都独立地应用于[41]中的单个点的特征向量，即应用于m×k矩阵的行。



## 1. Introduction

​	3D point cloud processing and understanding are usually deemed more challenging than 2D images mainly due to a fact that point cloud samples **live on an irregular structure** while 2D image samples (pixels) rely on a 2D grid in the image plane with a regular spacing. Point cloud geometry is typically represented by a set of sparse 3D points. Such a data format makes it difﬁcult to apply traditional deep learning framework. 

​	E.g. for each sample, traditional convolutional neural network (CNN) requires its neighboring samples to appear at some ﬁxed spatial orientations and distances so as to facilitate the convolution. Unfortunately, point cloud samples typically do not follow such constraints. 

​	One way to alleviate the problem is to voxelize a point cloud to **mimic**模仿 the image representation and then to operate on voxels. 

​	The downside is that voxelization has to either **sacriﬁce**牺牲 the representation accuracy or **incurs**招致 huge redundancies, that may pose an unnecessary cost in the subsequent processing, either at a **compromised performance** or an **rapidly increased processing complexity**. Related prior arts will be reviewed in Section 1.1. 

​	通常认为3D点云的处理和理解要比2D图像更具挑战性，这主要是由于以下事实：点云样本生活在不规则的结构上，而2D图像样本（像素）依赖于图像平面中规则间距的2D网格。点云的几何形状通常由一组稀疏的3D点表示。这种数据格式很难应用传统的深度学习框架。

​	例如。 对于每个样本，传统的卷积神经网络（CNN）都要求其邻近样本出现在某些固定的空间方向和距离上，以便于卷积。不幸的是，点云样本通常不遵循此类约束条件。 

​	一种解决问题的方法是对点云进行体素化以模仿图像表示，然后对体素进行操作。

​	缺点是体素化必须牺牲表示的准确性或招致巨大的冗余，这可能会在性能下降或处理复杂性迅速提高的情况下，在后续处理中造成不必要的成本。相关的现有技术将在第1.1节中进行回顾。



​	In this work, we focus on the emerging ﬁeld of unsupervised learning for point clouds. We propose an autoencoder (AE) that is referenced as FoldingNet. The output from the bottleneck layer in the auto-encoder is called a codeword that can be used as a high-dimensional embedding of an input point cloud. We are going to show that a 2D grid structure is not only a sampling structure for imaging, but can indeed be used to construct a point cloud through the proposed folding operation. This is based on the observation that the 3D point clouds of our interest are obtained from object surfaces: either discretized from boundary representations in CAD/computer graphics, or sampled from line-of-sight sensors like LIDAR. 

​	Intuitively, any 3D object surface could be transformed to a 2D plane through certain operations like **cutting, squeezing, and stretching**. The inverse procedure is to **glue** those 2D point samples back onto an object surface via certain folding operations, which are initialized as 2D grid samples. As illustrated in Table 1, to reconstruct a point cloud, successive folding operations are joined to reproduce the surface structure. The points are colorized to show the correspondence between the initial 2D grid samples and the reconstructed 3D point samples. Using the folding-based method, the challenges from the irregular structure of point clouds are well addressed by directly introducing such an implicit 2D grid constraint in the decoder, which avoids the costly 3D voxelization in other works [56]. It will be demonstrated later that the folding operations can build an arbitrary surface provided a proper codeword. Notice that when data are from volumetric format instead of 2D surfaces, a 3D grid may perform better. 

​	在这项工作中，我们专注于针对点云的无监督学习的新兴领域。我们提出了一种称为FoldingNet的自动编码器（AE）。自动编码器中瓶颈层的输出称为codeword，可以用作输入点云的高维嵌入。我们将展示2D网格结构不仅是用于成像的采样结构，而且确实可以通过建议的folding操作用于构建点云。这是基于观察到的，我们感兴趣的3D点云是从物体表面获得的：从CAD /计算机图形中的边界表示离散化，或者从视线传感器（如LIDAR）采样。

​	直观地讲，可以通过某些操作（如切割，挤压和拉伸）将任何3D对象表面转换为2D平面。逆过程是通过某些折叠操作将那些2D点样本胶合回对象表面，并将其初始化为2D网格样本。 如表1所示，为了重建点云，需要合并连续的折叠操作以重现表面结构。点被着色以显示初始2D网格样本和重建的3D点样本之间的对应关系。使用基于折叠的方法，通过在解码器中直接引入这种隐式2D网格约束，可以很好地解决来自点云不规则结构的挑战，从而避免了其他工作中昂贵的3D体素化[56]。稍后将说明，折叠操作可以在提供适当代码字的情况下构建任意表面。请注意，当数据来自体积格式而不是2D曲面时，3D网格的性能可能更好。



​	Despite being strongly expressive in reconstructing point clouds, the folding operation is simple: it is started by augmenting the 2D grid points with the **codeword** obtained from the encoder, which is then processed through a 3-layer perceptron. The proposed decoder is simply a concatenation of two folding operations. This design makes the proposed decoder much smaller in parameter size than the fully-connected decoder proposed recently in [1]. In Section 4.6, we show that the number of parameters of our folding-based decoder is about 7% of the fully connected decoder in [1]. Although the proposed decoder has a simple structure,wetheoreticallyshowinTheorem3.2thatthis folding-based structure is universal in that one folding operation that uses only a 2-layer perceptron can already reproduce arbitrary point-cloud structure. Therefore, it is not surprising that our FoldingNet auto-encoder exploiting two consecutive folding operations can produce elaborate structures. 

​	尽管在重构点云中表现力很强，但折叠操作却很简单：它是通过使用从编码器获得的codeword扩展2D网格点开始的，然后通过3层感知器对其进行处理。 提出的解码器只是两个折叠操作的串联。与[1]中最近提出的全连接解码器相比，这种设计使所提出的解码器在参数大小上要小得多。在第4.6节中，我们显示了基于折叠的解码器的参数数量约为[1]中完全连接的解码器的7％。 尽管所提出的解码器具有简单的结构，但在定理3.2中从理论上表明，这种基于折叠的结构是通用的，因为仅使用2层感知器的折叠操作就可以重现任意点云结构。 因此，利用两个连续折叠操作的FoldingNet自动编码器可以产生复杂的结构也就不足为奇了。



​	To show the efﬁciency of FoldingNet auto-encoder for unsupervised representation learning, we follow the experimental settings in [1] and test the transfer classiﬁcation accuracy from ShapeNet dataset[7] to ModelNet dataset[57]. The FoldingNet auto-encoder is trained using ShapeNet dataset, and tested out by extracting codewords from ModelNet dataset. Then, we train a linear SVM classiﬁer to test the discrimination effectiveness of the extracted codewords. The transfer classiﬁcation accuracy is 88.4% on the ModelNet dataset with 40 shape categories. This classiﬁcation accuracy is even close to the state-of-the-art supervised training result [41]. To achieve the best classiﬁcation performance and least reconstruction loss, we use a graph-based encoder structure that is different from [41]. This graph-based encoder is based on the idea of local feature pooling operations and is able to retrieve and propagate local structural information along the graph structure. 

​	为了显示FoldingNet自动编码器在无监督表示学习中的效率，我们遵循[1]中的实验设置，并测试了从ShapeNet数据集[7]到ModelNet数据集[57]的传输分类准确性。FoldingNet自动编码器使用ShapeNet数据集进行训练，并通过从ModelNet数据集中提取代码字进行测试。 然后，我们训练线性SVM分类器以测试提取的码字的鉴别效果。 在具有40个形状类别的ModelNet数据集上，转移分类的准确性为88.4％。这种分类精度甚至接近最新的监督训练结果[41]。 为了获得最佳的分类性能和最小的重建损失，我们使用不同于[41]的基于图的编码器结构。 这种基于图的编码器基于局部特征池化操作的思想，并且能够沿图结构检索和传播局部结构信息。



​	To intuitively interpret our network design: we want to impose a “virtual force” to **deform/cut/stretch** a 2D grid lattice onto a 3D object surface, and such a deformation force should be inﬂuenced or regulated by interconnections induced by the lattice neighborhood. Since the intermediate folding steps in the decoder and the training process can be illustrated by reconstructed points, the gradual change of the folding forces can be visualized. Now we summarize our contributions in this work:

• We train an end-to-end deep auto-encoder that consumes unordered point clouds directly. 

• We propose a new decoding operation called **folding** and theoretically show it is universal in point cloud reconstruction, while providing orders to reconstructed points as a unique byproduct than other methods. 

• We show by experiments on major datasets that folding can achieve higher classiﬁcation accuracy than other unsupervised methods. 

​	为了直观地解释我们的网络设计：我们想施加“虚拟力”以将2D网格点阵变形/切割/拉伸到3D对象表面上，并且这种变形力应受到由点阵邻域引起的互连的影响或调节。由于解码器中的中间折叠步骤和训练过程可以通过重构点来说明，因此可以看到折叠力的逐渐变化。现在，我们总结一下我们在这项工作中所做的贡献：

•我们训练了一种直接消耗无序点云的端到端深度自动编码器。
•我们提出了一种新的解码操作，称为折叠，从理论上讲它在点云重构中是通用的，同时为重构点提供命令是唯一的副产品，而不是其他方法。
•通过对主要数据集的实验表明，与其他无监督方法相比，折叠可以实现更高的分类精度。



---

### 1.1. Related works

​	Applications of learning on point clouds include shape completion and recognition [57], unmanned autonomous vehicles [36], 3D object detection, recognition and classiﬁcation [9,33,40,41,48,49,53], contour detection [21], layout inference [18], scene labeling [31], category discovery [60], point classiﬁcation, dense labeling and segmentation [3,10,13,22,25,27,37,41,54,55,58], 

​	点云上学习的应用包括形状完成和识别[57]，无人驾驶自动驾驶汽车[36]，3D对象检测，识别和分类[9,33,40,41,48,49,53]，轮廓检测[21]  ，布局推断[18]，场景标记[31]，类别发现[60]，点分类，密集标记和分割[3,10,13,22,25,27,37,41,54,55,58]，



​	Most deep neural networks designed for 3D point clouds are based on the idea of partitioning the 3D space into regular voxels and extending 2D CNNs to voxels, such as [4,11,37], including the the work on 3D generative adversarial network [56]. The main problem of voxel-based networks is the fast growth of neural-network size with the increasing spatial resolution. Some other options include octree-based [44] and kd-tree-based [29] neural networks. Recently, it is shown that neural networks based on purely 3D point representations [1,41–43] work quite efﬁciently for point clouds. The point-based neural networks can reduce the overhead of converting point clouds into other data formats (such as octrees and voxels), and in the meantime avoid the information loss due to the conversion. 

​	专门为3D点云设计的大多数深度神经网络都是基于将3D空间划分为规则体素并将2D CNN扩展到体素的想法，例如[4,11,37]，包括有关3D生成对抗网络的工作[56]。 基于体素的网络的主要问题是随着空间分辨率的提高，神经网络的大小快速增长。其他一些选择包括基于八叉树的[44]和基于kd树的[29]神经网络。最近，研究表明，基于纯3D点表示的神经网络[1,41–43]对于点云非常有效。基于点的神经网络可以减少将点云转换为其他数据格式（例如八叉树和体素）的开销，同时避免由于转换而造成的信息丢失。



​	The only work that we are aware of on end-to-end deep auto-encoder that directly handles point clouds is [1]. The AE designed in [1] is for the purpose of extracting features for generative networks. To encode, it sorts the 3D points using the **lexicographic order** and applies a **1D CNN** on the point sequence. To decode, it applies a **three-layer fully connected** network. This simple structure turns out to outperform all existing unsupervised works on representation extraction of point clouds in terms of the transfer classiﬁcation accuracy from the ShapeNet dataset to the ModelNet dataset [1]. Our method, which has a **graph-based encoder** and a **folding-based decoder**, outperforms this method in transfer classiﬁcation accuracy on the ModelNet40 dataset [1]. 

​	Moreover, compared to [1], our AE design is more **interpretable** 可解释性: the encoder learns the local shape information and combines information by max-pooling on a nearest neighbor graph, and the decoder learns a “force” to fold a two-dimensional grid twice in order to warp the grid into the shape of the point cloud, using the information obtained by the encoder. 

​	An other closely related work reconstructs a pointset from a 2D image[17]. Although the deconvolution network in [17] requires a 2D image as side information, we ﬁnd it useful as another implementation of our folding operation. We compare FoldingNet with the deconvolution based folding and show that FoldingNet performs slightly better in reconstruction error with fewer parameters (see Supplementary Section 9).

​	在直接处理点云的端到端深度自动编码器上，我们唯一了解的工作是[1]。 [1]中设计的AE是为生成网络提取特征的目的。为了进行编码，它使用字典顺序对3D点进行排序，并在点序列上应用1D CNN。 为了进行解码，它应用了三层完全连接的网络。 就从ShapeNet数据集到ModelNet数据集的传输分类精度而言，这种简单的结构胜过所有现有的无监督的点云表示提取工作。 我们的方法具有基于图的编码器和基于折叠的解码器，在ModelNet40数据集上的传输分类精度方面优于该方法[1]。 

​	此外，与[1]相比，我们的AE设计更具可解释性：编码器学习局部形状信息，并通过在最近邻图上进行最大池合并来组合信息，解码器学习“力”以折叠二维网格 两次，以使用编码器获得的信息将网格扭曲为点云的形状。 

​	另一项密切相关的工作是从2D图像重建点云[17]。 尽管[17]中的反卷积网络需要2D图像作为辅助信息，但我们发现它可作为我们折叠操作的另一种实现方式。 我们将FoldingNet与基于反卷积的折叠进行了比较，结果表明FoldingNet在重构误差较小且参数较少的情况下表现更好（请参阅补充部分9）。



​	It is hard for purely point-based neural networks to extract local neighborhood structure around points, i.e., features of neighboring points instead of individual ones. Some attempts for this are made in [1,42]. In this work, we exploit local neighborhood features using a graph-based framework. Deep learning on graph-structured data is not a new idea. There are tremendous amount of works on applying deep learning onto irregular data such as graphs and pointsets[2,5,6,12,14,15,23,24,28,32,35,38,39,43,47,52, 59]. Although using graphs as a processing framework for deep learning on point clouds is a natural idea, only several seminal works made attempts in this direction [5,38,47]. These works try to generalize the convolution operations from 2D images to graphs. However, since it is hard to deﬁne convolution operations on graphs, we use a simple graph-based neural network layer that is different from previous works: we construct the K-nearest neighbor graph(KNNG) and repeatedly conduct the max-pooling operations in each node’s neighborhood. It generalizes the global max-pooling operation proposed in [41] in that the max-pooling is only applied to each local neighborhood to generate local data signatures. Compared to the above graph based convolution networks, our design is simpler and computationally efﬁcient as in [41]. K-NNGs are also used in other applications of point clouds without the deep learning framework such as surface detection, 3D object recognition, 3D object segmentation and compression [20,50,51].

​	纯粹基于点的神经网络很难提取点周围的局部邻域结构，即相邻点的特征而不是单个点的特征。 在[1,42]中对此进行了一些尝试。在这项工作中，我们使用基于图的框架来开发局部邻域特征。在图结构化数据上进行深度学习并不是一个新主意。 将深度学习应用于图形和点集之类的不规则数据方面有大量工作[2,5,6,12,14,15,23,24,28,32,35,38,39,43,47,52,59]。 尽管使用图作为点云上深度学习的处理框架是很自然的想法，但只有几项开创性的工作在这个方向上进行了尝试[5,38,47]。 这些工作试图概括从2D图像到图形的卷积运算。 但是，由于很难在图上定义卷积运算，因此我们使用了一个简单的基于图的神经网络层，该层不同于以前的工作：我们构造了K最近邻图（KNNG），并在其中重复执行最大池化操作。 每个节点的邻居。 它概括了[41]中提出的全局最大池化操作，因为最大池化仅应用于每个本地邻域以生成本地数据签名。 与上述基于图的卷积网络相比，我们的设计更简单，计算效率更高，如[41]中所示。 在没有深度学习框架的情况下，K-NNG也可用于点云的其他应用中，例如表面检测，3D对象识别，3D对象分割和压缩[20,50,51]。



​	The folding operation that reconstructs a surface from a 2D grid essentially establishes a mapping from a 2D regular domain to a 3D point cloud. A natural question to ask is whether we can parameterize 3D points with compatible meshes that are not necessarily regular grids, such as cross-parametrization [30]. From Table 2, it seems that FoldingNet can learn to generate “cuts” on the 2D grid and generate surfaces that are not even topologically equivalent to a 2D grid, and hence make the 2D grid representation universal to some extent. Nonetheless, the reconstructed points may still have genus-wise distortions when the original surface is too complex. For example, in Table 2, see the missing winglets on the reconstructed plane and the missing holes on the back of the reconstructed chair. To recover those ﬁner details might require more input point samples and more complex encoder/decoder networks. Another method to learn the surface embedding is to learn a metric alignment layer as in [16], which may require computationally intensive internal optimization during training.

​	从2D网格重建表面的折叠操作实质上是建立从2D规则域到3D点云的映射。 一个自然的问题是，我们是否可以使用不一定是规则网格的兼容网格来参数化3D点，例如参数交叉化[30]。从表2中可以看出，FoldingNet可以学习在2D网格上生成“切口”并生成在拓扑上甚至不等同于2D网格的表面，因此可以使2D网格表示在某种程度上具有普遍性。 但是，当原始曲面过于复杂时，重构点可能仍会沿属类变形。例如，在表2中，可以看到重建平面上缺少的小翼，以及重建椅子背面上的缺少的孔。要恢复这些细节，可能需要更多的输入点样本和更复杂的编码器/解码器网络。 学习表面嵌入的另一种方法是学习度量对齐层，如[16]中所述，这可能需要在训练过程中进行大量计算上的内部优化。



### 1.2. Preliminaries and Notation 初步声明和符号

​	We will often denote the point set by S. We use bold lower-case letters to represent vectors, such as x, and use bold upper-case letters to represent matrices, such as A. The codeword is always represented by θ. We call a matrix m-by-n or m×n if it has m rows and n columns. 

​	我们通常会表示由S设置的点。我们使用粗体小写字母表示向量（例如x），并使用粗体大写字母表示矩阵（例如A）。codeword始终由θ表示。 如果矩阵有m行n列，我们称矩阵为m×n或m×n。



---

## 2. FoldingNet Auto-encoder on Point Clouds 

​	Now we propose the FoldingNet deep auto-encoder. The structure of the auto-encoder is shown in Figure 1. The input to the encoder is an n-by-3 matrix. Each row of the matrix is composed of the 3D position (x,y,z). The output is an m-by-3 matrix, representing the reconstructed point positions. The number of reconstructed points m is not necessarily the same as n. 

​	Suppose the input contains the point set S and the reconstructed point set is the set b S. Then, the reconstruction error for b S is computed using a layer deﬁned as the (extended) Chamfer distance, 

​	现在，我们提出了FoldingNet深度自编码器。自编码器的结构如图1所示。编码器的输入是一个n×3矩阵。 矩阵的每一行都由3D位置（x，y，z）组成。输出是一个m×3矩阵，表示重建的点位置。重建点的数量不一定与n相同。 

​	假设输入包含点集 $S$ ，而重构的点集为 $\hat{S}$。然后，使用定义为（扩展）倒角距离的层来计算 $\hat{S}$ 的重构误差，
$$
\begin{aligned}
d_{C H}(S, \widehat{S})=\max \left\{\frac{1}{|S|} \sum_{\mathbf{x} \in S} \min _{\widehat{\mathbf{x}} \in \widehat{S}}\|\mathbf{x}-\widehat{\mathbf{x}}\|_{2}\right.\\
\left.\frac{1}{|\widehat{S}|} \sum_{\widehat{\mathbf{x}} \in \widehat{S}} \min _{\mathbf{x} \in S}\|\widehat{\mathbf{x}}-\mathbf{x}\|_{2}\right\}
\end{aligned}
$$
​	其中 $\min _{\hat{x} \in \widehat{S}}\|x-\widehat{x}\|_{2}$ 强制原始点云中的任何3D点 $x$ 在重构点云中都具有匹配项 $\hat{x}$。反之亦然。其中 $\min _{x \in S}\|\widehat{x}-x\|_{2}$ 强制重构点云中的任何3D点  $\hat{x}$ 在原始点云中都具有匹配项 $x$ 。 max 操作要求从$S$ 到 $\hat{S}$ 的距离和 $\hat{S}$ 到 $S$ 的距离同时小。编码器计算每个输入点云的表示（codeword），然后解码器使用该codeword重建点云。 在我们的实验中，代码字长度设置为512与[1]符合。



### 2.1. Graph-based Encoder Architecture  基于图的编码架构

focuses on supervised learning using **point cloud neighborhood graphs**. The encoder is a **concatenation** of multi-layer perceptrons (MLP) and graph-based max-pooling layers. The graph is the K-NNG constructed from the 3D positions of the nodes in the input point set. In experiments, we choose K = 16. First, for every single point v, we compute its local covariance matrix of size 3-by-3 and vectorize it to size 1-by-9. The local covariance of v is computed using the 3D positions of the points that are **one-hop** neighbors of v (including v) in the K-NNG. We **concatenate** the matrix of point positions with size n-by-3 and the local covariances for all points of size n-by-9 into a matrix of size n-by-12 and input them to a 3-layer perceptron. 

​	The perceptron is applied in parallel to each row of the input matrix of size n-by-12. It can be viewed as a per-point function on each 3D point. The output of the perceptron is fed to two consecutive graph layers, where each layer applies max-pooling to the neighborhood of each node. More speciﬁcally, suppose the KNN graph has adjacency matrix A and the input matrix to the graph layer is X. Then, the output matrix is

​	基于图的编码器遵循类似的设计[46]，该设计着重于使用**点云邻域图**进行监督学习。 编码器是多层感知器（MLP）和基于图的最大池化层的串联。该图是根据输入点集中节点的3D位置构造的K-NNG。 在实验中，我们选择K =16。首先，对于每个单点v，我们计算其局部协方差矩阵，其大小为 3×3，并将其矢量化为 1×9 的大小。使用K-NNG中v（包括v）的一跳邻居的点的3D位置来计算v的局部协方差。我们将 n×3 大小的点位置矩阵和 n×9 大小的所有点的局部协方差连接到 n×12 大小的矩阵中，并将它们输入到3层感知器中。

​	感知器并行处理 nx12 大小的输入矩阵以的每行。可以查看每个3D点上的点函数。感知器的输出被馈送到两个连续的图形层，其中每个层将max-pooling应用于每个节点的邻域。更具体地说，假设K-NN图具有邻接矩阵A，并且图层的输入矩阵是X。然后，输出矩阵是
$$
\mathbf{Y}=\mathbf{A}_{\max }(\mathbf{X}) \mathbf{K}
$$
​	其中 K 是特征映射矩阵，矩阵$\mathbf{A}_{\max }(\mathbf{X})$的第$（i，j）$项是

$$
\left(\mathbf{A}_{\max }(\mathbf{X})\right)_{i j}=\operatorname{ReLU}\left(\max _{k \in \mathcal{N}(i)} x_{k j}\right)
$$
​	（3）中的局部最大池化操作 $\max _{k \in \mathcal{N}(i)}$ 本质上是基于图结构计算局部特征的。 该签名可以表示局部邻居的（汇总）拓扑信息。 通过基于图的最大池层的串联，网络将拓扑信息传播到更大的区域。





![1614604444799](assets/1614604444799.png)

**Fig. 1** FoldingNet体系结构。 graph-layers 是第2.1节（2）中提到的graph-based的最大池化层。 第一个和第二个 folding 都通过三层感知器将 codeword 连接到特征向量。每个感知器都独立地应用于[41]中的单个点的特征向量，即应用于m×k矩阵的行。

---

### 2.2. Folding-based Decoder Architecture 基于折叠的解码器架构

​	提出的的解码器使用两个连续的3层感知器将固定的2D网格扭曲为输入点云的形状。输入的 codeword 是从基于图的编码器获得的。在将codeword输入解码器之前，我们将其复制m次，并将 mx512 矩阵与 mx2 矩阵连接起来，该矩阵在以原点为中心的正方形上包含 m 个网格点。 串联的结果是大小为m×514 的矩阵。矩阵由3层感知器逐行处理，输出矩阵大小为 m-by-3(**intermediate point cloud**)。 之后，我们将复制的codeword重新连接到m-by-3 输出，并将其馈入3层感知器。此输出是重构的点云(reconstructed point cloud)。参数n是根据输入点云大小设置的，例如在我们的实验中，与[1]相同n = 2048。我们在一个正方形中选择m个网格点，因此将m选择为2025，**这是最接近2048的平方数**。 

> Deﬁnition 1. 我们将复制codeword的连接称为低维**网格点**，然后将**point-wise MLP** 每个点都经过MLP称为**folding**操作。 

​	**折叠操作实质上形成了通用的2D到3D映射**。为了直观地了解为什么此折叠操作是通用的2D到3D映射，请通过矩阵 $\mathbf{U}$ 表示输入的2D网格点。$\mathbf{U}$ 的每一行都是一个二维网格点。 用 $\mathbf{u}_i$ 表示 $\mathbf{U}$ 的第 $i$ 行，用 $θ$ 表示编码器输出的codeword。然后，在级联之后，输入矩阵的第 $i$ 行到 MLP为 $[\mathbf{u}_{i}, \boldsymbol{\theta}]$。 由于 MLP 并行处理输入矩阵的每一行，因此**输出**矩阵的第 $i$ 行可以写为 $f\left(\left[\mathbf{u}_{i}, \boldsymbol{\theta}\right]\right)$，其中 $f$ 表示由MLP进行的函数。

​	可以将此函数视为参数化的高维函数，其中codeword $θ$ 为参数以指导**函数的结构**（**folding** 操作）。 由于MLP擅长逼近非线性函数，因此它们可以在2D网格上执行复杂的折叠操作。 高维codeword本质上**存储**进行折叠所需的**力**，这使折叠操作更加**多样化**。 

​	提出的解码器具有**两个连续的folding**操作。**第一个将2D网格折叠到3D空间，第二个在3D空间内部折叠**。我们在表1中显示了这两个折叠操作之后的输出。从表1的C列和D列中，我们可以看到每个折叠操作都进行了相对简单的操作，并且两次折叠操作的组成可以产生非常精细的表面形状。 尽管第一个折叠看起来比第二个折叠简单，但它们一起导致最终产出的实质性变化。如果需要更精细的表面形状，则可以应用更成功的折叠操作。 可以在补充部分第8节中找到解码器的更多变化形式，包括网格尺寸的变化和折叠操作的数量。



---

## 3. 理论分析

定理3.1。 所提出的编码器结构是置换不变的，即，如果置换了输入点云矩阵，则codeword保持不变。 

证明。 见补充部分6。然后，我们陈述一个关于所提出的基于折叠的解码器的普遍性的定理。 它显示了基于折叠的解码器的存在，因此通过更改代码字 $θ$ ，输出可以是任意点云。 



定理3.2。 有一个2层感知器，可以使用折叠操作从2维网格中重建任意点云。

​	更具体地，假设输入的矩阵 $\mathbf{U}$ 的大小为 m×2，使得 $\mathbf{U}$ 的每一行都是点在大小为 m 的二维网格上的2D位置。 然后，存在一个2层感知器（带有手工系数）的显式构造，使得对于任意大小为 m×3 的任意3D点云矩阵 $S$（其中 $S$ 的每一行都是$（x，y，z）$ 点云中一个点的位置），则存在一个 codeword 向量 $θ$，这样，如果我们将 $θ$ 连接到 $\mathbf{U}$ 的每一行，并在连接后对每个矩阵行并行应用2层感知器，则可以从感知器的输出中获取点云矩阵 $S$. 



完全证明是补充性的，第7节。在证明中，我们通过显式构造一个满足所陈述特性的2层感知器来显示该存在。 主要思想是表明，在最坏的情况下，2D网格中的点用作选择性逻辑门，以将2D网格中的2D点映射到点云中的相应3D点。请注意，以上证明是基于抗干扰性的，以表明我们的解码器结构是通用的。 它并不表示在折叠式网络自动编码器中是否存在现实状态。 理论上构造的解码器需要3m的隐藏单元，而实际上，我们使用的解码器的大小要小得多。 此外，定理3.2中的构造可实现对点云的无损重构，而FoldingNet自动编码器仅可实现有损重构。 但是，上面的定理可以保证提议的解码操作（即将码字连接到二维网格点并使用感知器处理每一行）是合理的，因为在最坏的情况下，存在基于折叠的神经网络，该网络具有手工制作的边缘权重，可以重建任意点云。 实际上，通过适当的训练对所提出的解码器进行良好的参数设置会导致更好的性能。















































