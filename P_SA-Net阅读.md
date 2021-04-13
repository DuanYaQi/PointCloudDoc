# SA-Net: Point Cloud Completion by Skip-attention Network with Hierarchical Folding

## Abstract

​	Point cloud completion aims to infer the complete geometries for missing regions of 3D objects from incomplete ones. Previous methods usually predict the complete point cloud based on the global shape representation extracted from the incomplete input. However, the global representation often suffers from the information loss of structure details on local regions of incomplete point cloud. To address this problem, we propose Skip-Attention Network(SA-Net) for 3D point cloud completion. 点云补全的目的是从不完整的对象中推断出3D对象缺失区域的完整对象。 先前的方法通常基于从不完整输入中提取的全局形状表示来预测完整点云。 但是，全局表示通常会遭受不完整点云局部区域上结构细节的信息丢失。 为了解决这个问题，我们提出了跳过注意力网络（SA-Net）来完成3D点云。



​	Our main contributions lie in the following two-folds. 

​	First, we propose a skip-attention mechanism to effectively exploit the local structure details of incomplete point clouds during the inference of missing parts. The skip-attention mechanism selectively conveys geometric information from the local regions of incomplete point clouds for the generation of complete ones at different resolutions, where the skip-attention reveals the completion process in an interpretable way. Second, in order to fully utilize the selected geometric information encoded by skip-attention mechanism at different resolutions, we propose a novel structure-preserving decoder with hierarchical folding for complete shape generation. The hierarchical folding preserves the structure of complete point cloud generated in upper layer by progressively detailing the local regions,using the skip-attentioned geometry at the same resolution. We conduct comprehensive experiments on ShapeNet and KITTI datasets, which demonstrate that the proposed SA-Net outperforms the state-of-the-art point cloud completion methods.




$$
\begin{equation}
 a_{j, k}=\frac{\exp \left(\mathrm{M}\left(\boldsymbol{p}_{j}^{i} \mid \theta_{h}\right)^{\mathrm{T}} \cdot \mathrm{M}\left(\boldsymbol{p}_{k}^{i} \mid \theta_{l}\right)\right)}{\sum_{n=1}^{N_{i}} \exp \left(\mathrm{M}\left(\boldsymbol{p}^{i} \mid \theta_{h}\right)^{\mathrm{T}} \cdot \mathrm{M}\left(\boldsymbol{p}^{i} \mid \theta_{1}\right)\right)} 
\end{equation}
$$

$$
\begin{equation}
 \boldsymbol{p}_{j}^{i} \leftarrow \boldsymbol{p}_{j}^{i}+\sum_{k=1}^{N_{i}^{D}} a_{j, k} \cdot \mathrm{M}\left(\boldsymbol{p}_{k}^{i} \mid \theta_{g}\right) 
\end{equation}
$$



$$
\begin{equation}
 a_{j, k}^{\mathrm{C}}=\frac{\left(\boldsymbol{r}_{k}^{i}\right)^{\mathrm{T}} \boldsymbol{p}_{j}^{i}}{\left\|\boldsymbol{r}_{k}^{i}\right\|_{2}\left\|\boldsymbol{p}_{j}^{i}\right\|_{2}} 
\end{equation}
$$






