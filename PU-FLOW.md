# PU-Flow

## start

```
docker run --runtime=nvidia --rm -it unknownue/pu-flow:latest /bin/zsh


docker run --runtime=nvidia --rm -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE unknownue/pu-flow:latest /bin/zsh
```



```
docker cp /home/duan/Downloads/4577251581f7b1fe1dea6f6320002e46ba1348b4-PatchSRFlow-resort4-100epoch.ckpt 6e90cd0b4f1e:/home/unknownue/PU-Flow/
```



```
docker ps -a
docker commit -a "duan" -m "xxxx" -p 6e90cd0b4f1e unknownue/pu-flow:latest
```



```bash
ssh-keygen -t rsa -C "837738300@qq.com" 
cat ~/.ssh/id_rsa.pub
git remote set-url origin git@xxx.com:xxx/xxx.git
```







## Problem

### PyTorchEMD

```
CUDA-10.1

Replaced the deprecated AT_CHECK with TORCH_CHECK
IN cuda/emd_kernel.cu
```

https://github.com/daerduoCarey/PyTorchEMD



### Pointnet2_PyTorch

```
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

https://github.com/erikwijmans/Pointnet2_PyTorch



### PU-GAN

```
Evaluation code

先安装cgal
https://www.cgal.org/download/linux.html

cmake时需要Realease模式
cmake  -DCMAKE_BUILD_TYPE=Release .
```



5f091ac2bfa8e60a6edd6c58c3e21e3d7b728304-PatchSRFlow-resort4-100epoch.ckpt

![image-20201019214842047](/home/duan/.config/Typora/typora-user-images/image-20201019214842047.png)



a230667647096224cc06c3fecd6eaeea14788538-PatchSRFlow-resort4-100epoch.ckpt

![image-20201019220337548](/home/duan/.config/Typora/typora-user-images/image-20201019220337548.png)



4577251581f7b1fe1dea6f6320002e46ba1348b4-PatchSRFlow-resort4-100epoch.ckpt

![image-20201020125341382](/home/duan/windows/PointCloud/pic/image-20201020125341382.png)



```
checkpoint_path = './runs/PUGAN/ckpt/5f091ac2bfa8e60a6edd6c58c3e21e3d7b728304-PatchSRFlow-resort4-100epoch.ckpt'
checkpoint_path = './runs/PUGAN/ckpt/a230667647096224cc06c3fecd6eaeea14788538-PatchSRFlow-resort4-100epoch.ckpt'
checkpoint_path = './runs/PUGAN/ckpt/4577251581f7b1fe1dea6f6320002e46ba1348b4-PatchSRFlow-resort4-100epoch.ckpt'

cat train_pflow.py | tail -10
```







## SRFlow







---

## Code流程

### train_pflow.py

#### def main

```python
设置一个检查点以用来保存文件
checkpoint_path = 'runs/PUGAN/best/4a8c742-PatchSRFlow-resort4fix-100epoch.ckpt'

train('Train', checkpoint_path)
```



#### def train

```python
###################################################### def train
args 提取默认配置
	pc_channel = 3        通道数
    levels = 2            上采样层数
######################################################
dataset_cfg 数据集配置
	seed：				1085											种子数1085
    dataset 
    	root:			./data/PU-GAN/Resort4fix_PUGAN_256_1024.h5   	路径
        batch_size: 	28												块大小
        num_worker: 	16												进程数
        is_normalize: 	True											是否归一化
######################################################    
trainer_config
	default_root_dir 	'./runs/' 						训练结果保存路径
    gpus				1								gpu数
    fast_dev_run		False							
    max_epochs			100								最大前后传播中所有批次的单次训练迭代
    deterministic		False							确定性函数
    num_sanity_val_steps 0								pytorch-lighting参数
    callbacks			[TimeTrainingCallback()]		时间回调函数/计时函数
    
######################################################
datamoudle = VisionAirPUGANDataModule(dataset_cfg)		pytorchlightning数据集datamodule
	visionair点云数据集（pu-gan）的lighting-datamodule 见dataset/pugan.py解析
network = PatchSRFlowNet(args) 							pytorchlightning网络架构
	
```



#### class PatchSRFlowNet

```python
# ---------------------------------------------------------------------------------------------
class PatchSRFlowNet(pl.LightningModule):

    def __init__(self, hparams): 									#初始化
        super(PatchSRFlowNet, self).__init__()
        self.hparams = hparams

        if isinstance(hparams, dict): 								#是否是dict类型
            hparams = OmegaConf.create(hparams)

        self.network = PointSRFlow(hparams.pc_channel, hparams.levels) 					#见modules/sr/uflow.py的类 PointSRFlow

        self.valid_metrics = UpsampingMetrics(['EMD', 'Repulsion'])						#验证集衡量指标
        self.test_metrics  = UpsampingMetrics(['CD', 'EMD', 'Uniform', 'Repulsion']) 	#测试集衡量指标

        self.min_EMD = 100.0
# ---------------------------------------------------------------------------------------------
    @staticmethod 加上静态装饰的类，无论是类还是对象都可以调用到该方法
    def data_preprocessing(x: Tensor, u: Tensor, is_permute=False):

        # x = x.transpose(1, 2).contiguous()

        # plot_pointcloud(x.transpose(1, 2)[1], path='runs/PCN/inputs/x.png', is_show=False, is_save=True)
        # plot_pointcloud(u[1], path='runs/PCN/inputs/u.png', is_show=False, is_save=True)
        # exit()

        # plot_pointclouds_open3d([x[0].detach().cpu()])

        if is_permute:
            permute_x = torch.randperm(x.shape[1])
            permute_u = torch.randperm(u.shape[1])
            x = x[:, permute_x]
            u = u[:, permute_u]
        return x, u
# ---------------------------------------------------------------------------------------------
    def forward(self, x: Tensor, u: Tensor):
        x, u = PatchSRFlowNet.data_preprocessing(x, u, is_permute=False)
        return self.network(x, u)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):

        pt_gt, pt_input, _ = batch
        loss = self(pt_gt, pt_input)

        return loss

    def validation_step(self, batch, batch_idx):

        pt_gt, pt_input, radius = batch
        loss = self(pt_gt, pt_input)
        val_loss = loss.detach().cpu()
        predict_x = self.network.sample(pt_input)

        metrics = self.valid_metrics(predict_x, pt_gt, radius)

        return {
            'vloss'    : val_loss,
            'EMD'      : metrics['EMD'],
            'Repulsion': metrics['Repulsion'],
        }

    def validation_epoch_end(self, batch):

        log_dict = {
            'vloss'    : torch.tensor([x['vloss']     for x in batch]).sum().item(),
            'EMD'      : torch.tensor([x['EMD']       for x in batch]).sum().item(),
            'repulsion': torch.tensor([x['Repulsion'] for x in batch]).sum().item(),
        }

        print()
        self.log_dict(log_dict, prog_bar=True, logger=False)
        
        if log_dict['EMD'] < self.min_EMD:
            save_path = './runs/PUGAN/ckpt/PatchSRFlow-resort4fix-EMD_best-100epoch.ckpt'
            torch.save(self.network, save_path)
            print(f'Best model in EMD metric has been save to {save_path}')
            self.min_EMD = log_dict['EMD']

    def test_step(self, batch, batch_idx):
        pt_gt, pt_input, radius = batch
        predict_x = self.network.sample(pt_input)

        metrics = self.test_metrics(predict_x, pt_gt, radius)
        return {
            'CD'       : metrics['CD'],
            'EMD'      : metrics['EMD'],
            'Uniform'  : metrics['Uniform'],
            'Repulsion': metrics['Repulsion'],
        }

    def test_epoch_end(self, batch):
        log_dict = {
            'CD'       : torch.tensor([x['CD']        for x in batch]).sum().item(),
            'EMD'      : torch.tensor([x['EMD']       for x in batch]).sum().item(),
            'Uniform'  : torch.tensor([x['Uniform']   for x in batch]).sum().item(),
            'Repulsion': torch.tensor([x['Repulsion'] for x in batch]).sum().item(),
        }
        print(log_dict)
```









----

### modules/sr

####  uflow.py

##### class PointSRFlow

```python
# ----------------------------------------------传统写法PointSRFlow
class PointSRFlow(nn.Module):

    def __init__(self, pc_channel: int, levels: int):				# input: 2 3
        super(PointSRFlow, self).__init__()

        self.is_transition = False  # Whether to use Transition Step 				是否采用transition步骤

        in_channel = pc_channel # the number of raw channels of point cloud 		点云常规通道
        hidden_channel = 256    # the number of channels of convolutional layer		卷积层通道数
        self.n_flows = levels   # the count of 2x upsampling 			 			上采样级别为2		
        self.n_steps = 5        # the number of flow steps in each scale level 		每个尺度流动的步骤
        permutation = 'random'  # the strategy for shuffling channels       		划分通道的策略 为random
        ch = in_channel * 2 														

        self.duplicate_strategy = DuplicateStrategy.DiffAll  # strategy to duplicate channels from 从u复制通道策略
        if self.duplicate_strategy == DuplicateStrategy.DiffU:
            self.var_duplicate = 10  # the number of channels variations introduced to duplicate u  引入复制u的通道变化数量
        if self.duplicate_strategy == DuplicateStrategy.DiffAll:
            self.var_duplicate = 2

        self.dist = GaussianDistribution(pc_channel, mu=0.0, vars=1.0, temperature=0.8)   #分布为高斯分布 参考modules/utils/prob.py

        self.squeezer = PtsSqueeze()								#可以增加通道轴的维数，但依然保留局部相关性 参考modules/sr/squeeze.py
        
        self.point_encoder = DynamicGraphCNN(in_channel, k=20, emb_dim=128) #动态图卷积 DGCNN  参考modules/sr/encoder.py
        udim = self.point_encoder.out_dim  # the number of channels for encoded low resolution point cloud 编码的低分辨率点云的通道数

        self.actnorm_layers    = nn.ModuleList()	#
        self.permutate_layers  = nn.ModuleList()	#
        self.injector_layers   = nn.ModuleList()	#
        self.coupling_layers   = nn.ModuleList()	# 耦合层
        self.spline_layres     = nn.ModuleList()	# 
        self.split_layers      = nn.ModuleList()	# 
        self.transition_layers = nn.ModuleList()	# 

        for _ in range(self.n_flows):				# 共两层
            transition = TransitionStep(ch, is_actnorm=True) #ch:4
            self.transition_layers.append(transition)

            for _ in range(self.n_steps):
                # actnorm = ActNorm(ch)
                # self.actnorm_layers.append(actnorm)

                permutate = Permutation(permutation, n_channel=ch)
                self.permutate_layers.append(permutate)

                injector = AffineInjectorLayer(ch, hidden_channel, inject_channel=(udim + self.var_duplicate))
                self.injector_layers.append(injector)

                affine = AffineCouplingLayer('affine-condition', ch, hidden_channel, u_channel=(udim + self.var_duplicate))
                self.coupling_layers.append(affine)

                # spline = SplineCouplingLayer('linear-rational', ch, hidden_channel, u_channel=(udim + self.var_duplicate))
                # self.spline_layres.append(spline)

            split = SplitPts(dist=self.dist, channel=ch)
            self.split_layers.append(split)

    def forward(self, x: Tensor, u: Tensor):
        return self.log_prob(x, u)

    def g(self, x: Tensor, u: Tensor):
        u = self.point_encoder(u)  # [B, udim, N]

        for i in reversed(range(self.n_flows)):
            # [B, udim + self.var_duplicate, rN]
            repeat_u = PointSRFlow.duplicate_u(self.duplicate_strategy, u, i, self.n_flows, self.var_duplicate)

            x = self.split_layers[i].inverse(x)

            for j in reversed(range(self.n_steps)):
                step_idx = i * self.n_steps + j

                # x = self.spline_layres[step_idx].inverse(x, repeat_u)
                x = self.coupling_layers[step_idx].inverse(x, repeat_u)
                x = self.injector_layers[step_idx].inverse(x, repeat_u)
                x = self.permutate_layers[step_idx].inverse(x)
                # x = self.actnorm_layers[step_idx].inverse(x)

            if self.is_transition:
                x = self.transition_layers[i].inverse(x)

            x = self.squeezer.inverse(x)

        return x

    def f(self, x: Tensor, u: Tensor):
        """
        z: High Resolution Point Cloud, in [B, N_high, C]
        u: Low  Resolution Point Cloud, in [B, N_low,  C]
        """
        B, _, _ = u.shape
        log_det_J = torch.zeros((B,), device=x.device)
        log_z     = torch.zeros((B,), device=x.device)

        # Encode low resolution point cloud
        u = self.point_encoder(u)           # [B, udim, N_low]
        z = x.transpose(1, 2).contiguous()  # [B, C, N_ligh]

        for i in range(self.n_flows):
            # TODO: Try different repeat methods
            repeat_u = PointSRFlow.duplicate_u(self.duplicate_strategy, u, i, self.n_flows, self.var_duplicate)

            z = self.squeezer(z)

            if self.is_transition:
                z, log_det_t = self.transition_layers[i](z)
                log_det_J -= log_det_t

            # Conditional flow steps
            for j in range(self.n_steps):
                step_idx = i * self.n_steps + j

                # z, logdet0 = self.actnorm_layers[step_idx](z)
                z, logdet1 = self.permutate_layers[step_idx](z)
                z, logdet2 = self.injector_layers[step_idx](z, repeat_u)
                z, logdet3 = self.coupling_layers[step_idx](z, repeat_u)
                # z, logdet3 = self.spline_layres[step_idx](z, repeat_u)
                # log_det_J -= (logdet0 + logdet1 + logdet2 + logdet3)
                log_det_J -= (logdet1 + logdet2 + logdet3)

            # Split layer
            z, _log_z = self.split_layers[i](z)
            log_z += _log_z

        return z, log_z, log_det_J

    def log_prob(self, x: Tensor, u: Tensor):
        """
        See also https://stackoverflow.com/questions/54635355/what-does-log-prob-do
        """
        z, log_z, log_det_J = self.f(x, u)  # [B, N, C], [B,]
        logp_z = self.dist.standard_logp(z.transpose(1, 2)).to(log_det_J.device) + log_z  # [B,]
        return -torch.mean(logp_z + log_det_J)

    def sample(self, u: Tensor, **kwargs):
        z = self.dist.standard_sample(u.shape, u.device, **kwargs)
        z = z.transpose(1, 2)
        x = self.g(z, u)
        x = x.transpose(1, 2)
        return x

    @staticmethod
    def duplicate_u(strategy: DuplicateStrategy, u: Tensor, flow_i: int, n_flows: int, var_duplicate: int):
        B, _, N = u.shape
        duplicate_count = pow(2, n_flows - flow_i - 1)
        repeat_u = u.unsqueeze(-1).repeat(1, 1, 1, duplicate_count)\
            .view(B, -1, N * duplicate_count)  # [B, C, N * duplicate_count]

        if strategy == DuplicateStrategy.DiffU:
            variations = torch.arange(duplicate_count, dtype=torch.float32, device=u.device)\
                .repeat(B, var_duplicate, N)
        elif strategy == DuplicateStrategy.DiffAll:
            variations = PointSRFlow.gen_grid(N * duplicate_count, B, bound=0.3)  # [B, 2, N]
            variations = variations.transpose(1, 2).contiguous().to(u.device)
        else:
            raise NotImplementedError()
        return torch.cat([repeat_u, variations], dim=1)

    @staticmethod
    def gen_grid(N, B, bound=0.2):
        sqrted = int(math.sqrt(N)) + 1
        num_x, num_y = None, None
        for i in reversed(range(1, sqrted + 1)):
            if (N % i) == 0:
                num_x = i
                num_y = N // i
                break
        grid_x = torch.linspace(-bound, bound, num_x)
        grid_y = torch.linspace(-bound, bound, num_y)

        x, y = torch.meshgrid(grid_x, grid_y)
        grid = torch.stack([x, y], dim=-1).reshape(-1, 2)  # [N, 2] [2, 2, 2] -> [4, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1)  # [B, N, 2]
        return grid
```



##### class TransitionStep

```python
class TransitionStep(nn.Module):  

    def __init__(self, channels: int, is_actnorm=True):   #channels:4
        super(TransitionStep, self).__init__()

        self.is_actnorm = is_actnorm
        self.actnorm = ActNorm(channels)
        self.permute = Permutation('random', channels)
    
    def forward(self, x: Tensor):

        log_det_Arc = torch.tensor([0.0], device=x.device)
        if self.is_actnorm:
            x, log_det_Arc = self.actnorm(x)
        x, log_det_Per = self.permute(x)
        return x, log_det_Arc + log_det_Per
    
    def inverse(self, x: Tensor):

        x = self.permute.inverse(x)
        if self.is_actnorm:
            x = self.actnorm.inverse(x)
        return x
```





---

####  squeeze.py

##### class PtsSqueeze

```python
# -------------------------------------------------------------------------realnvp引入的squeeze操作，增加通道数，提高局部相关性
class PtsSqueeze(nn.Module):

    def forward(self, x: Tensor):
        """
        Increase channel, decrease resolution   增加通道，减少处理
        x: [B, C, N]
        z: [B, C * 2, N / 2]
        """
        x1, x2 = x[:, :, 0::2], x[:, :, 1::2]
        z = torch.cat([x1, x2], dim=1)  #  两个张量(tensor)拼接在一起,
        return z

    def inverse(self, z: Tensor):
        """
        Decrease channel, increase resolution   增加处理，减少通道，反向操作
        z: [B, C * 2, N / 2]
        x: [B, C, N]
        """ 
        B, _, N = z.shape
        x = z.view(B, 2, -1, N).permute(0, 2, 3, 1).flatten(start_dim=2)
        return x
```



----

####  encoder.py

##### class DynamicGraphCNN

```python
class DynamicGraphCNN(nn.Module):
    """
    Dynamic Graph CNN for Learning on Point Clouds.
    Code from https://github.com/AnTao97/dgcnn.pytorch.
    """
    
    def __init__(self, in_channel: int, k: int, emb_dim: int, output='pointwise'):   # 初始化
        super(DynamicGraphCNN, self).__init__()

        self.k = k
        self.output = output  # 'pointwise' or 'global'
        self.out_dim = emb_dim * 2 + 3 + 64 + 64 + 128 + 256

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, xyz: Tensor):
        xyz = xyz.transpose(1, 2).contiguous()
        B, _, N = xyz.shape                     					# batchsize, ndataset, dimension = xyz.shape  
        #[B,N,3], B代表Batchsize,即有多少样本， N代表每个样本的总点数，3代表点云的x,y,z坐标；

        x = DynamicGraphCNN.get_graph_feature(xyz, k=self.k)      	# (B, 3, N) 			-> 	(B, 3 * 2, N, k)
        x = self.conv1(x)                       					# (B, 3 * 2, N, k) 		-> 	(B, 64, N, k)
        x1, _ = x.max(dim=-1, keepdim=False)    					# (B, 64, N, k) 		-> 	(B, 64, N)

        x = DynamicGraphCNN.get_graph_feature(x1, k=self.k)     	# (B, 64, N) 			-> 	(B, 64 * 2, N, k)
        x = self.conv2(x)                       					# (B, 64 * 2, N, k) 	-> 	(B, 64, N, k)
        x2, _ = x.max(dim=-1, keepdim=False)    					# (B, 64, N, k) 		-> 	(B, 64, N)

        x = DynamicGraphCNN.get_graph_feature(x2, k=self.k)     	# (B, 64, N) 			-> 	(B, 64 * 2, N, k)
        x = self.conv3(x)                       					# (B, 64 * 2, N, k) 	-> 	(B, 128, N, k)
        x3, _ = x.max(dim=-1, keepdim=False)    					# (B, 128, N, k) 		-> 	(B, 128, N)

        x = DynamicGraphCNN.get_graph_feature(x3, k=self.k)     	# (B, 128, N) 			-> 	(B, 128 * 2, N, k)
        x = self.conv4(x)                       					# (B, 128 * 2, N, k) 	-> 	(B, 256, N, k)
        x4, _ = x.max(dim=-1, keepdim=False)    					# (B, 256, N, k) 		-> 	(B, 256, N)

        x_f = torch.cat([x1, x2, x3, x4], dim=1)  					# (B, 64 + 64 + 128 + 256, N)

        _x = self.conv5(x_f)   										# (B, 64 + 64 + 128 + 256, N) 	-> 	(B, emb_dim, N)
        x1 = F.adaptive_max_pool1d(_x, 1).view(B, -1)   			# (B, emb_dim, N) 				-> 	(B, emb_dim)

        if self.output == 'pointwise':
            x2 = F.adaptive_avg_pool1d(_x, 1).view(B, -1)   		# (B, emb_dim, N) 		-> 	(B, emb_dim)
            _x = torch.cat([x1, x2], dim=1).unsqueeze(-1).repeat(1, 1, N)   # (B, emb_dim * 2, N)

            x = torch.cat([_x, xyz, x_f], dim=1)  					# (B, emb_dim * 2 + 3 + 64 + 64 + 128 + 256, N)
            return x
        if self.output == 'global':
            return x1

    @staticmethod
    def knn(x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
        return idx

    @staticmethod
    def get_graph_feature(x, k=20, idx=None, dim9=False):
        B = x.size(0)
        N = x.size(2)
        x = x.view(B, -1, N)
        if idx is None:
            if dim9 == False:
                idx = DynamicGraphCNN.knn(x, k=k)   # (B, N, k)
            else:
                idx = DynamicGraphCNN.knn(x[:, 6:], k=k)

        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N

        idx = idx + idx_base

        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()   # (B, N, num_dims)  -> (B*N, num_dims) #   B * N * k + range(0, B*N)
        feature = x.view(B * N, -1)[idx, :]
        feature = feature.view(B, N, k, num_dims) 
        x = x.view(B, N, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
        return feature  # (B, 2 * num_dims, N, k)
```



##### class LinearTransformLayer

```python
class LinearTransformLayer(nn.Module):

    def __init__(self, in_channel: int, hidden_channel: int, out_channel: int):
        super(LinearTransformLayer, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_channel, hidden_channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_channel, out_channel),
            nn.Tanh())  # High performance influence
            # # Change to other activation layer may lead to gradient explosion
            # nn.RReLU(inplace=True))
            # nn.PReLU())
            # nn.Softplus())  # Not work

    def forward(self, h: Tensor):
        h = h.transpose(1, 2)
        h = self.layers(h)
        h = h.transpose(1, 2)
        return h

```









#### normalize.py

##### class Actnorm

```python
class Actnorm(nn.Module):
    def __init__(self, channel):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.
		执行激活归一化的转换。 适用于2D和4D输入。 假设 BxCxHxW 输入格式，则按通道对4D输入（图像）进行归一化。
        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        super(ActnormB, self).__init__()

        self.is_initialized = False
        self.eps = 1e-6
        self.log_scale = nn.Parameter(torch.zeros(channel))
        self.shift     = nn.Parameter(torch.zeros(channel))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def _broadcastable_scale_shift(self):
        return self.scale.view(1, -1, 1), self.shift.view(1, -1, 1)

    def forward(self, inputs):
        if self.training and not self.is_initialized:
            self._initialize(inputs)

        scale, shift = self._broadcastable_scale_shift()
        outputs = scale * inputs + shift

        B, _, N = inputs.shape
        log_det_J = N * torch.sum(self.log_scale) * inputs.new_ones((B,))

        return outputs, log_det_J

    def inverse(self, inputs):
        scale, shift = self._broadcastable_scale_shift()
        outputs = (inputs - shift) / scale

        return outputs

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance.数据相关的初始化 post-actnorm activations的均值和单位方差为零
        """
        num_channels = inputs.shape[1]                     # 0是第1各维度 1第2个维度
        inputs = inputs.permute(0, 2, 1).reshape(-1, num_channels)  # 将tensor的1 2维度换位  
        #reshape返回一个 tensor, 其data和元素数量与 input 一样, 但是改变成指定的形状 输入参数不够即自动填充

        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / (std + self.eps)).mean(dim=0)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu

        self.is_initialized = True
```



#### permuate.py

##### class Permuation

```python
class Permutation(nn.Module):

    def __init__(self, permutation: str, n_channel: int):
        super(Permutation, self).__init__()

        assert permutation in ['reverse', 'random']
        self.permutation = permutation

        if permutation == 'reverse':
            self.direct_idx = np.arange(n_channel - 1, -1, -1).astype(np.long)
            self.inverse_idx = Permutation.get_reverse(self.direct_idx, n_channel)
        if permutation == 'random':
            self.direct_idx = np.arange(n_channel - 1, -1, -1).astype(np.long)
            np.random.shuffle(self.direct_idx)
            self.inverse_idx = Permutation.get_reverse(self.direct_idx, n_channel)
 
    def forward(self, x: Tensor):
        log_det_J = torch.tensor([0.0], device=x.device)
        return x[:, self.direct_idx, :], log_det_J

    def inverse(self, z: Tensor):
        return z[:, self.inverse_idx, :]

    @staticmethod
    def get_reverse(idx, n_channel: int):
        indices_inverse = np.zeros((n_channel,), dtype=np.long)
        for i in range(n_channel):
            indices_inverse[idx[i]] = i
        return indices_inverse
```



#### coupling.py

##### class AffineInjectorlayer

```python
class AffineInjectorLayer(nn.Module):

    def __init__(self, in_channel: int, hidden_channel: int, inject_channel: int):
        super(AffineInjectorLayer, self).__init__()

        self.scaling_layers = LinearTransform(inject_channel, hidden_channel, in_channel) # see modules/sr/encoder.py-class
        self.bias_layers    = LinearTransform(inject_channel, hidden_channel, in_channel)

    def forward(self, h: Tensor, u: Tensor):
        scale = self.scaling_layers(u)
        bias  = self.bias_layers(u)

        h = (h - bias) * torch.exp(-scale)
        log_det_J = torch.sum(scale, dim=[1, 2])
        return h, log_det_J
    
    def inverse(self, h: Tensor, u: Tensor):
        scale = self.scaling_layers(u)
        bias  = self.bias_layers(u)

        h = h * torch.exp(scale) + bias
        return h

```



---

### modules/utils

####  probs.py

##### class GaussianDistribution

```python
class GaussianDistribution(Distribution):
    Log2PI = float(np.log(2 * np.pi))

    def __init__(self, pc_channel: int, mu: float, vars: float, temperature: float=1.0):
        super(GaussianDistribution, self).__init__()

        mu   = torch.ones(pc_channel) * mu
        vars = torch.eye(pc_channel) * vars
        self.prior = MultivariateNormal(mu, vars)                   #先验分布：多元高斯分布
 
        assert temperature >= 0.0 and temperature <= 1.0
        self.temperature = temperature * temperature  # temperature annealing

    def standard_logp(self, z: Tensor):								#计算先验的对数概率
        logp_z = self.prior.log_prob(z.cpu())
        sum_dims = tuple([i for i in range(1, logp_z.dim())])
        logp_z = torch.sum(logp_z, dim=sum_dims) # [B,]
        return logp_z

    def standard_sample(self, shape, device, temperature=None):		#标准取样
        temp = temperature ** 2 if temperature is not None else self.temperature
        return (torch.randn(shape) * temp).to(device)
```



----

### dataset

#### pugan.py

##### class VisionAirPUGANDataModule

```python
## 数据加载模块 dataloadmodule
class VisionAirPUGANDataModule(pl.LightningDataModule):
    
    def __init__(self, hparams: DictConfig):
		#初始化

    def config_dataset(self, split, shuffle=True, is_jitter=False, is_rotate=False, is_scale=False):
		#配置训练集
        dataset = VisionAirPUGAN(self.rootdir, split,
            self.cfg.is_normalize, is_jitter, is_rotate, is_scale,
            seed=self.seed, verbose=True)

        dataloader = DataLoader(
            dataset, self.cfg.batch_size, shuffle=shuffle,
            num_workers=self.cfg.num_worker, pin_memory=(split == 'train'), drop_last=False,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))
        return dataloader
# ---------------------------------------------------------------------------------------------
    #训练-验证-测试
    def train_dataloader(self):
        return self.config_dataset(split='train',
            shuffle=True, is_jitter=False, is_rotate=False, is_scale=False)

    def val_dataloader(self):
        return self.config_dataset(split='valid',
            shuffle=False, is_jitter=False, is_rotate=False, is_scale=False)

    def test_dataloader(self):
        return self.config_dataset(split='test',
            shuffle=False, is_jitter=False, is_rotate=False, is_scale=False)
```



##### class VisionAirPUGAN

```python
## 数据配置类  
class VisionAirPUGAN(Dataset):

    def __init__(self, root, split, is_normalize=True, is_jitter=True, is_rotate=True, is_scale=False, seed=42, verbose=False):
        #初始化
        self.pt_input, self.pt_gt, self.radius = self.read_h5_split(split_ratio, seed, verbose)
        if is_normalize:
            self.pt_input, self.pt_gt = VisionAirPUGAN.normalize(self.pt_input, self.pt_gt, verbose)


    def read_h5_split(self, split_ratio, split_seed, verbose): 	#读取h5点云数据集 并进行随机采样
            return pt_input, pt_gt, pt_radius
    
    def __getitem__(self, index: int):							#获取以加载至缓冲区的点云数据
        return pt_input, pt_gt, pt_radius

    def __len__(self) -> int: 									#获取长度
        return len(self.pt_input)
# ---------------------------------------------------------------------------------------------
    #  归一化 震荡 旋转 尺度变化
    @staticmethod
    def normalize(pt_input, pt_gt, verbose):
        return pt_input, pt_gt

    @staticmethod
    def jitter(pt, sigma=0.005, clip=0.02):
        return pt + jitter_noise.astype(np.float32)

    @staticmethod
    def rotate(pt_input, pt_gt, z_rotated=True):
        return pt_input, pt_gt

    @staticmethod
    def scale(pt_input, pt_gt, scale_low=0.5, scale_high=2):
        return pt_input * scale, pt_gt * scale, scale
```









---

## 扩展补充

### pytorch lightning

#### datamodule

```python
# regular PyTorch  常规的pytorch 加载数据集代码
test_data = MNIST(PATH, train=False, download=True)
train_data = MNIST(PATH, train=True, download=True)
train_data, val_data = random_split(train_data, [55000, 5000])

train_loader = DataLoader(train_data, batch_size=32)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)


可以写成下述的格式
class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = PATH, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):                                   	#API
        self.mnist_test = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):										#API
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):										#API
        return DataLoader(self.mnist_val, batch_size=self.batch_size)
		
    def test_dataloader(self):										#API
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
    
    
mnist = MNISTDataModule(PATH, 32)

```

#### lightning module

```python
最核心的6个方法
init 								Define computations here
forward								Use for inference only (separate from training_step)
training_step						the full training loop
validation_step						the full validation loop
test_step							the full test loop
configure_optimizers				define optimizers and LR schedulers


class LitMNIST(pl.LightningModule):
    
    def __init__(self, data_dir='./', hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
```







## 文件树结构

```
├── data
│   └── PU-GAN			训练数据集
│       ├── PUGAN_poisson_256_poisson_1024.h5
│       ├── ..............................
│       ├── Resort8_PUGAN_poisson_256_poisson_1024.h5
├── dataset
│   ├── mnist.py
│   ├── pcn.py
│   ├── PU-GAN-DATASET
│   │   ├── off
│   │   │   ├── complex
│   │   │   │   ├── AncientTurtl_aligned.off
│   │   │   │   ├── ..............................
│   │   │   │   ├── angel2_aligned.off
│   │   │   ├── medium
│   │   │   │   ├── 10014_dolphin_v2_max2011_it2.off
│   │   │   │   ├── ..............................
│   │   │   │   ├── 11499_Elephant_v2.off
│   │   │   ├── simple
│   │   │   │   ├── armadillo.off
│   │   │   │   ├── ..............................
│   │   │   │   ├── block.off
│   │   │   └── test
│   │   │       ├── 11509_Panda_v4.off
│   │   │       ├── ..............................
│   │   │       ├── camel.off
│   │   ├── xyz-poisson-8192
│   │   │   ├── complex
│   │   │   │   ├── AncientTurtl_aligned.xyz
│   │   │   │   ├── ..............................
│   │   │   │   ├── angel2_aligned.xyz
│   │   │   ├── medium
│   │   │   │   ├── 10014_dolphin_v2_max2011_it2.xyz
│   │   │   │   ├── ..............................
│   │   │   │   ├── 11499_Elephant_v2.xyz
│   │   │   ├── simple
│   │   │   │   ├── armadillo.xyz
│   │   │   │   ├── ..............................
│   │   │   │   ├── block.xyz
│   │   │   └── test
│   │   │       ├── 11509_Panda_v4.xyz
│   │   │       ├── ..............................
│   │   │       ├── camel.xyz
│   │   └── xyz-random-2048
│   │       ├── complex
│   │       │   ├── AncientTurtl_aligned.xyz
│   │       │   ├── ..............................
│   │       │   ├── angel2_aligned.xyz
│   │       ├── medium
│   │       │   ├── 10014_dolphin_v2_max2011_it2.xyz
│   │       │   ├── ..............................
│   │       │   ├── 11499_Elephant_v2.xyz
│   │       ├── simple
│   │       │   ├── armadillo.xyz
│   │       │   ├── ..............................
│   │       │   ├── block.xyz
│   │       └── test
│   │           ├── 11509_Panda_v4.xyz
│   │           ├── ..............................
│   │           ├── camel.xyz
│   │           └── test
│   │               └── output
│   ├── pugan.py
│   └── __pycache__
│       ├── pcn.cpython-36.pyc
│       └── pugan.cpython-36.pyc
├── docker
│   ├── Dockerfile
│   └── README.md
├── evaluate_pflow.py
├── experiments
│   ├── flow.py
│   ├── glow_mnist.py
│   ├── lognormal.py
│   ├── lrs.py
│   ├── mp_processing.py
│   └── realnvp.py
├── history
│   ├── mflow.py
│   └── mpflow.py
├── logs
│   └── Patch-PU-GAN-grids.log
├── metric
│   ├── auction_match
│   │   ├── auction_match_gpu.cpp
│   │   ├── auction_match_gpu.cu
│   │   ├── auction_match.py
│   │   └── __init__.py
│   ├── chamfer_distance
│   │   ├── chamfer_distance.cpp
│   │   ├── chamfer_distance.cu
│   │   ├── chamfer_distance.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── chamfer_distance.cpython-36.pyc
│   │       └── __init__.cpython-36.pyc
│   ├── loss.py
│   ├── pointnet2
│   │   ├── _ext_src
│   │   │   ├── include
│   │   │   │   ├── ball_query.h
│   │   │   │   ├── cuda_utils.h
│   │   │   │   ├── group_points.h
│   │   │   │   ├── interpolate.h
│   │   │   │   ├── sampling.h
│   │   │   │   └── utils.h
│   │   │   └── src
│   │   │       ├── ball_query.cpp
│   │   │       ├── ball_query_gpu.cu
│   │   │       ├── ball_query.h
│   │   │       ├── bindings.cpp
│   │   │       ├── cuda_utils.h
│   │   │       ├── group_points.cpp
│   │   │       ├── group_points_gpu.cu
│   │   │       ├── group_points.h
│   │   │       ├── interpolate.cpp
│   │   │       ├── interpolate_gpu.cu
│   │   │       ├── interpolate.h
│   │   │       ├── sampling.cpp
│   │   │       ├── sampling_gpu.cu
│   │   │       ├── sampling.h
│   │   │       └── utils.h
│   │   ├── pointnet2.egg-info
│   │   │   ├── dependency_links.txt
│   │   │   ├── PKG-INFO
│   │   │   ├── SOURCES.txt
│   │   │   └── top_level.txt
│   │   ├── pointnet2_modules.py
│   │   ├── pointnet2_utils.py
│   │   ├── pytorch_utils.py
│   │   └── setup.py
│   ├── __pycache__
│   │   └── loss.cpython-36.pyc
│   └── PytorchEMD
├── modules
│   ├── cropping.py
│   ├── dpf
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   ├── layer.py
│   │   ├── mflow.py
│   │   └── mpflow.py
│   ├── fps.py
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-36.pyc
│   ├── sr
│   │   ├── coupling.py
│   │   ├── encoder.py
│   │   ├── flow.py
│   │   ├── inv1x1.py
│   │   ├── normalize.py
│   │   ├── permutate.py
│   │   ├── __pycache__
│   │   │   ├── coupling.cpython-36.pyc
│   │   │   ├── encoder.cpython-36.pyc
│   │   │   ├── normalize.cpython-36.pyc
│   │   │   ├── permutate.cpython-36.pyc
│   │   │   ├── split.cpython-36.pyc
│   │   │   ├── squeeze.cpython-36.pyc
│   │   │   └── uflow.cpython-36.pyc
│   │   ├── spline
│   │   │   ├── cubic.py
│   │   │   ├── __init__.py
│   │   │   ├── linear_rational.py
│   │   │   ├── __pycache__
│   │   │   │   ├── cubic.cpython-36.pyc
│   │   │   │   ├── __init__.cpython-36.pyc
│   │   │   │   ├── linear_rational.cpython-36.pyc
│   │   │   │   └── quadratic_rational.cpython-36.pyc
│   │   │   └── quadratic_rational.py
│   │   ├── split.py
│   │   ├── squeeze.py
│   │   └── uflow.py
│   └── utils
│       ├── cropping.py
│       ├── fps.py
│       ├── patch.py
│       ├── permute.py
│       ├── probs.py
│       └── __pycache__
│           ├── fps.cpython-36.pyc
│           ├── patch.cpython-36.pyc
│           └── probs.cpython-36.pyc
├── plotting
│   ├── image.py
│   ├── open3d.py
│   ├── pc_open3d.py
│   └── __pycache__
│       ├── image.cpython-36.pyc
│       ├── open3d.cpython-35.pyc
│       └── open3d.cpython-36.pyc
├── preprocessing
│   ├── evaluate_pugan.py
│   ├── mnist2d.py
│   ├── pcn.py
│   ├── poission_sampling.py
│   └── pugan.py
├── runs					测试结果目录
│   ├── lightning_logs
│   └── PUGAN
│       ├── ckpt			模型文件
│       │   ├── 4577251-PatchSRFlow-resort4-100epoch.ckpt
│       │   ├── 5f091ac-PatchSRFlow-resort4-100epoch.ckpt
│       │   ├── a230667-PatchSRFlow-resort4-100epoch.ckpt
│       │   └── NoisyInjective-PatchSRFlow-resort4-2epoch.ckpt
│       ├── images
│       │   ├── Epoch-0.png
│       │   ├── ..............................
│       │   ├── Epoch-15.png
│       ├── out
│       │   ├── 11509_Panda_v4.xyz
│       │   ├── ..............................
│       │   ├── camel.xyz
│       └── plotting
│           ├── PatchSR-pcn-resort4-0.png
│           ├── ..............................
│           └── PatchSR-resort8.png
├── train_pflow.py
├── train_uflow.py
└── utils
    ├── callback.py
    ├── compare.py
    ├── parallel.py
    ├── __pycache__
    │   ├── callback.cpython-36.pyc
    │   └── time.cpython-36.pyc
    └── time.py

```













