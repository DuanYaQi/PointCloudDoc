# SA-Net: Point Cloud Completion by Skip-attention Network with Hierarchical Folding

CVPR2020



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







# 复现



```python
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        #               ,BN(channels[i]))
        for i in range(1, len(channels))
    ])


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch, num_samples=32):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=num_samples)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
    
class SkipAttention(Attention):

    def __init__(self, NN_h, NN_l, NN_g, NN_f=None):
        super(SkipAttention, self).__init__(NN_h, NN_l, NN_g, NN_f)

    def forward(self, p, r):
        h = self.M_h(p).expand(-1, -1, r.size(2), -1).unsqueeze(-2)
        l = self.M_l(r).expand(-1, h.size(1), -1, -1).unsqueeze(-1)
        g = self.M_g(r).squeeze()
        mm = torch.matmul(h, l).squeeze()
        attn_weights = F.softmax(mm, dim=-1)
        atten_appllied = torch.bmm(attn_weights, g)
        if self.M_f is not None:
            return self.M_f(p.squeeze() + atten_appllied)
        else:
            return p.squeeze() + atten_appllied
        
class SaNet(torch.nn.Module):
    meshgrid = [[-0.3, 0.3, 46], [-0.3, 0.3, 46]]
    x = np.linspace(*meshgrid[0])
    y = np.linspace(*meshgrid[1])

    points = torch.tensor(np.meshgrid(x, y), dtype=torch.float32)

    def __init__(self):
        super(SaNet, self).__init__()

        self.sa1_module = SAModule(0.25, 0.2, MLP([3 + 3, 64, 64, 128])) # ratio, r, nn
        self.sa2_module = SAModule(0.5, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 512]))#nn

        self.skip_attn1 = SkipAttention(MLP([512 + 2, 128]), MLP([256, 128]), MLP([256, 512 + 2]), MLP([512 + 2, 512])) # NN_h, NN_l, NN_g, NN_f
        self.skip_attn2 = SkipAttention(MLP([256, 64]), MLP([128, 64]), MLP([128, 256]), MLP([256, 256])) # NN_h, NN_l, NN_g

        self.folding1 = FoldingBlock(64, 256, [MLP([512 + 512, 256]), MLP([512 + 512, 256]), MLP([512 + 512, 512 + 512]),
                                               MLP([512 + 512, 512, 256])], [512 + 2, 512], [1024, 512])

        self.folding2 = FoldingBlock(256, 512, [MLP([256 + 256, 64]), MLP([256 + 256, 64]), MLP([256 + 256, 256 + 256]),
                                                MLP([256 + 256, 256, 128])], [256 + 2, 256], [256, 256])
        self.folding3 = FoldingBlock(512, 2048, [MLP([128 + 128, 64]), MLP([128 + 128, 64]), MLP([128 + 128, 128 + 128]),
                                                 MLP([128 + 128, 128])], [128 + 2, 128], [512, 256, 128])

        self.lin = Seq(Lin(128, 64), ReLU(), Lin(64, 3))
    @staticmethod
    def sample_2D(m, n):
        indeces_x = np.round(np.linspace(0, 45, m)).astype(int)
        indeces_y = np.round(np.linspace(0, 45, n)).astype(int)
        x, y = np.meshgrid(indeces_x, indeces_y)
        p = SaNet.points[:, x.ravel(), y.ravel()].T.contiguous()
        return p

    def Encode(self, data):
        sa1_out = self.sa1_module(data.pos, data.pos, data.batch)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        return sa1_out, sa2_out, sa3_out

    def Decode(self, encoded):
        p = SaNet.sample_2D(8, 8)
        out = encoded[2][0].contiguous()
        out = out.view(out.size(0), 1, 1, out.size(-1)).repeat(1, 64, 1, 1)
        out = torch.cat((out, p.view(1, p.size(0), 1, p.size(-1)).repeat(out.size(0), 1, 1, 1)), -1)
        out = self.skip_attn1(out, encoded[1][0].view(out.size(0), 1, 256, encoded[1][0].size(-1)))
        out = self.folding1(out, 16, 16)
        out = out.unsqueeze(-2)
        out = self.skip_attn2(out, encoded[0][0].view(out.size(0), 1, 512, encoded[0][0].size(-1)))
        out = self.folding2(out, 16, 32)
        out = self.folding3(out, 64, 32)

        return self.lin(out)

    def forward(self, data):
        encoded = self.Encode(data)

        decoded = self.Decode(encoded)

        return decoded, encoded
```






