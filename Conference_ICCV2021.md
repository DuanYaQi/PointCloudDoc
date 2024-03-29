# ICCV 2019

去噪

### Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning [denoising; Tensorflow]

完全降噪 3D点云清洗的无监督学习

Ulm University

University College London

​	仅从嘈杂的3D点云数据中就可以无监督地学习3D点云的去噪。 这是通过将最新思想从**无监督图像降噪器**的学习扩展到**无结构3D点云**来实现的。无监督图像降噪器的工作条件是，噪声像素观察是围绕干净像素值的分布的随机实现，这允许对该分布进行适当的学习，最终收敛到正确的值。 遗憾的是，该假设对**非结构化点无效**：3D点云会受到总噪声的影响，即所有坐标均存在偏差，并且**没有可靠的像素网格**。

​	因此，可以观察到干净的3D点的整个流形的实现，这使得将无监督图像降噪器简单地扩展到3D点云是不切实际的。 为了克服这个问题，我们引入了一个空间先验项 spatial prior term，它使转向收敛到流形上许多可能模式中唯一最接近的一种。 

​	我们的结果表明，在给定足够的训练示例的情况下，无监督降噪性能类似于具有干净数据的有监督学习的降噪性能。因此，我们不需要任何对噪声和干净的培训数据。





### 3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions [generation; PyTorch]

生成









STD: Sparse-to-Dense 3D Object Detector for Point Cloud [det]



USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds [keypoints, registration; PyTorch]



LPD-Net: 3D Point Cloud Learning for Large-Scale Place Recognition and Environment Analysis [place recognition]



Unsupervised Multi-Task Feature Learning on Point Clouds [cls, seg]



Multi-Angle Point Cloud-VAE: Unsupervised Feature Learning for 3D Point Clouds from Multiple Angles by Joint Self-Reconstruction and Half-to-Half Prediction [unsupervised, cls, generation, seg, completion]



SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences [dataset]



MeteorNet: Deep Learning on Dynamic 3D Point Cloud Sequences [cls, seg, flow estimation; Tensorflow]



DeepGCNs: Can GCNs Go as Deep as CNNs? [seg; Tensorflow]



VV-NET: Voxel VAE Net with Group Convolutions for Point Cloud Segmentation [seg; Github]



Interpolated Convolutional Networks for 3D Point Cloud Understanding [cls, seg]



Dynamic Points Agglomeration for Hierarchical Point Sets Learning [cls, seg]



ShellNet: Efficient Point Cloud Convolutional Neural Networks using Concentric Shells Statistics [cls, seg; Tensorflow]



Fast Point R-CNN [det]



Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data [dataset; cls; Tensorflow]



KPConv: Flexible and Deformable Convolution for Point Clouds [cls, seg; code]



Fully Convolutional Geometric Features [match; PyTorch]



Deep Closest Point: Learning Representations for Point Cloud Registration [registration; PyTorch]



DeepICP: An End-to-End Deep Neural Network for 3D Point Cloud Registration [registration]



Efficient and Robust Registration on the 3D Special Euclidean Group [registration]



Hierarchical Point-Edge Interaction Network for Point Cloud Semantic Segmentation [seg]



DensePoint: Learning Densely Contextual Representation for Efficient Point Cloud Processing [cls, retrieval, seg, normal estimation; PyTorch]



DUP-Net: Denoiser and Upsampler Network for 3D Adversarial Point Clouds Defense [cls]



Efficient Learning on Point Clouds with Basis Point Sets [cls, registration; PyTorch]



PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows [generation, reconstruction; Pytorch]



PU-GAN: a Point Cloud Upsampling Adversarial Network [upsampling, reconstruction; Project]



3D Point Cloud Learning for Large-scale Environment Analysis and Place Recognition [retrieval, place recognition]



Deep Hough Voting for 3D Object Detection in Point Clouds [det; PyTorch]



Exploring the Limitations of Behavior Cloning for Autonomous Driving [autonomous driving; Pytorch]