# MAPU-Net

Deep Magniﬁcation-Arbitrary Upsampling over 3D Point Clouds 

在3D点云上进行深度放大-任意向上采样





## 3. PROBLEM FORMULATION 







## 4. PROPOSED METHOD

A. 

Note that in contrast to existing deep learning based 3D point cloud upsampling methods that support only a predeﬁned and ﬁxed upsampling factor, making them unpractical for real-world applications, the proposed framework is able to achieve magniﬁcation-arbitrary upsampling, i.e., it can handle an arbitrary factor after one-time training. Such a ﬂexibility is credited to the unique principle of our framework, which allows us to learn uniﬁed and sorted interpolation weights. That is, the network is initialized with the maximum factor Rmax, and the interpolation with a random R (R ≤ Rmax) is performed in each iteration during training, i.e., the top-R groups of estimated weights are selected for the R× upsampling, such that the learned groups of interpolation weights are naturally sorted. Therefore, during inference the top-R groups of estimated interpolation weights could be selected for a speciﬁc factor.

请注意，与现有的基于深度学习的3D点云升采样方法仅支持预定义和固定的升采样因子相比，使它们在实际应用中不可行，而提出的框架能够实现放大任意的升采样，即，它可以处理 一次性训练后的任意因素。 这样的灵活性归功于我们框架的独特原理，该原理使我们能够学习统一和排序的插值权重。 也就是说，使用最大因子Rmax初始化网络，并在训练期间的每次迭代中使用随机R（R≤Rmax）进行插值，即，为Rx上采样选择估计权重的前R个组 ，以便对学习到的内插权重组进行自然排序。 因此，在推理过程中，可以为特定因子选择估计插值权重的前R个组。



### B. Geometry-aware Local Feature Embedding 

​	在此阶段，X的每个3D点xi都投影到由ci∈RD表示的高维特征空间上。 特别是，我们采用动态图CNN（DGCNN）[41]来实现这一过程。 与先前针对点云[24]，[34]，[34]的深层特征表示方法不同，这些方法适用于单个点或使用坐标之间的距离构造的固定图，而DGCNN根据在点云中获得的特征之间的距离来定义局部邻域。 上一层。 具体来说，用E⊂X×X表示由k个近邻计算出的边，然后根据特征距离将初始有向图G =（X，E）从一层动态更新为另一层。 此外，它涉及密集的连接以聚合多个级别的功能。 尽管在特征空间中使用局部邻域，但是学习到的特征表示ci编码了局部和非局部信息，同时仍保持了置换不变性。

​	此外，我们采用距离编码器[14]明确嵌入点之间的相对位置。 这样的显式嵌入会增强相应的点要素，以了解其邻域信息。令SK i = {xk i} K k = 1为在欧几里得距离的意义上xi个K个最近邻点的集合，因此DGCNN获得的K个点的相关高维特征由{ck i}表示 K k = 1。 距离编码器采用MLP来获取每个相邻点的高维特征rk i∈RD，即

rk i = MLPxi ⊕xk i ⊕(xi −xk i )⊕kxi −xk ik2, (8)

其中⊕是连接运算符，k·k2是向量的2范数，MLP（·）表示MLP过程。通过 DGCNN 将编码后的相对距离特征进一步与特征ck i相连，以形成e ck i∈R2D：

e ck i = ck i ⊕rk i . (9) 

通过局部坐标信息的显式编码，高维特征可以捕获局部几何图案







