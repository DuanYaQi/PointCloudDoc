# Invertible Residual Networks

ICML2019 



## Abstract

我们证明了标准的 ResNet 体系结构可以使之**可逆**，从而允许将相同的模型用于分类，**密度估计和生成**。通常，实施可逆性需要**划分维度**(partitioning dimensions)或**限制网络体系结构**( restricting network architectures)。相反，我们的方法只需要在**训练**过程中添加一个简单的**标准化步骤**(normalization step)即可，这已经在标准框架中提供了。

Invertible ResNets 定义了一个生成模型，可以通过对未标记数据的**最大似然**进行训练。 为了计算似然，我们对 residual block 的雅可比对数行列式引入了**易于处理的近似**（tractable approximation）。

我们的经验评估表明，Invertible ResNets 在最先进的图像分类器和 flow-based 生成模型上均具有竞争力，这是以前用单一架构无法实现的。



---

## 1.  Introduction

基于神经网络的模型的主要吸引力之一是，单个模型体系结构通常可用于解决各种相关任务。 但是，最近的许多改进都基于针对特定领域（particular domains ）量身定制的专用解决方案。例如，无监督学习中的最新架构正在变得越来越具有领域特定性 （Van Den Oord，2016b; Kingma，2018; Parmar，2018; Karras，2018; Van  Den Oord，2016a）。另一方面，用于判别式学习的最成功的前馈架构之一是深度残差网络（He，2016; Zagoruyko，2016），与同类的生成网络有很大差异。

这种鸿沟使得为**给定任务**选择或设计合适的体系结构变得很复杂。这也使**判别任务**难以从**无监督学习**中受益。我们用在这两个领域都表现出色的新型架构来弥合这一差距。



为实现这一目标，我们专注于可逆网络，这些网络已证明在判别性（Gomez，2017; Jacobsen，2018）和生成性（Dinh，2014; 2017; Kingma，  2018）独立执行任务，尽管使用相同的模型范式（model paradigm）。

它们通常依赖于固定维拆分启发法，但与非体积保留（non-volume conserving ）元素交织的常见拆分受到限制，它们的选择对性能有重大影响（Kingma，2018; Dinh，2017）。这使得建立可逆网络成为一项艰巨的任务。在这项工作中，我们表明，这些具有竞争性的密度估计性能所必需的奇特（exotic）设计会严重损害判别性能。



为了克服这个问题，我们利用 ResNets 作为 ODE 的 Euler 离散化的观点（Haber，2018; Ruthotto，2018; Lu，2017; Ciccone，2018）并且证明了 invertible ResNets (iResNets) 只需更改标准 ResNets 的规范化方案即可构建。

![1621241932669](assets/1621241932669.png)

**Fig. 1:** 标准 ResNet 网络(左)和 iResNet (右)的动力学。两个网络都将区间 $[2，2]$ 映射为: i)半深度处的噪声 **$x^3$-函数**；ii)全深度处的噪声**恒等函数**。Invertible ResNets 描述了一个双射连续动态，而 ResNets 导致了与非双射连续动态相对应的交叉和折叠路径(用白色圈起来)。由于折叠(collapsing)路径，ResNets 不是有效的密度模型。



**Fig. 1** 可视化了标准和可逆ResNet所学到的动力学差异。该方法允许每个 residual block 具有不受约束的体系结构，而只需要**每个 block 的 Lipschitz 常数小于一个常数**。我们证明，在构建图像分类器时，此限制对性能的影响可以忽略不计-在对MNIST，CIFAR10和CIFAR100图像进行分类时，它们的性能与不可逆的性能相当。



然后，我们展示如何将 i-ResNets 训练为未标记数据上的最大似然生成模型。为了计算似然，我们对残差块的雅可比行列式引入了易于处理的（tractable approximation）近似。像FFJORD（Grathwohl，2019）一样，i-ResNet flows 具有不受约束的（free-form）雅可比矩阵，这使他们可以学习**比**使用的三角映射的**其他可逆模型**中**更富有表现力**的变换。我们的经验评估表明，i-ResNets 在最先进的图像分类器和 flow-based 的生成模型上均具有竞争力，使通用体系结构更接近现实。





---

## 2. Enforcing Invertibility in ResNets

在常微分方程初值问题上 ResNet 架构与 Euler 方法有显著的相似性
$$
\begin{equation}
 x_{t+1} \leftarrow x_{t}+g_{\theta_{t}}\left(x_{t}\right) \\ x_{t+1} \leftarrow x_{t}+h f_{\theta_{t}}\left(x_{t}\right) 
\end{equation}
$$
其中 $ x_{t} \in \mathbb{R}^{d} $ 表示激活或状态，$t$ 表示层索引或时间，$h> 0$ 是步长，$g_{θ_t}$ 是残差块。这种联系在深度学习和动力学系统的交叉点吸引了研究（Lu，2017; Haber，2018; Ruthottor，2018; Chen，2018）。 但是，很少有人关注时间倒退(backwards)的动力学。
$$
\begin{equation}
 x_{t} \leftarrow x_{t+1}-g_{\theta_{t}}\left(x_{t}\right) \\ x_{t} \leftarrow x_{t+1}-h f_{\theta_{t}}\left(x_{t}\right) 
\end{equation}
$$
这相当于隐式向后的 Euler 离散化。特别地，及时解决动力学倒退将实现相应的 ResNet 的逆过程。 以下 [Theorem 1](#the1) 指出，一个简单的条件足以使动力学可解，从而使 ResNet 可逆： 



---

### Theorem 1

可逆 ResNets 的充分条件

令 $ F_{\theta}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{d} $ ，其中 $ F_{\theta}=\left(F_{\theta}^{1} \circ \ldots \circ F_{\theta}^{T}\right) $ 定义一个ResNet，该网络内块 $ F_{\theta}^{t}=I+g_{\theta_{t}} $ 。然后如果满足以下条件:
$$
\begin{equation}
 \operatorname{Lip}\left(g_{\theta_{t}}\right)<1 \text{, for all  t=1,} \ldots, T 
\end{equation}
$$
则 ResNet $F_{\theta}$ 可逆，其中 $\operatorname{Lip}\left(g_{\theta_{t}}\right)$ 是 $g_{\theta_{t}}$ 的 Lipschitz 常数。

请注意，**此条件对于可逆性不是必需的**。其他方法（NICE/RealNvp，2014; 2017; i-revnet，2018; Chang，2018; Glow，2018）依赖于**划分维度**或**自回归结构**来创建逆的解析解。



**Algorithm 1.**<span id="algo1"></span>

![1621245202635](assets/1621245202635.png)



当强制令 $\operatorname{Lip}\left(g_{\theta_{t}}\right)<1$ 使 ResNet 可逆，我们没有此逆的解析形式。但是，我们可以通过简单的定点迭代（fixed-point iteration）来获得它，请参见 [Algorithm 1](#algo1)。请注意，定点迭代的起始值可以是任何矢量，因为定点是唯一的。但是，将输出 $y = x + g(x)$ 用作初始化 $x^0:=y$ 是一个很好的起点，因为 $y$ 仅通过**恒等边界扰动**（a bounded perturbation of the identity）从 $x$ 中获得。根据巴纳赫不动点定理（Banach fixed-point theorem），我们有
$$
\begin{equation}
 \left\|x-x^{n}\right\|_{2} \leq \frac{\operatorname{Lip}(g)^{n}}{1-\operatorname{Lip}(g)}\left\|x^{1}-x^{0}\right\|_{2} 
\end{equation}\tag{1}
$$

因此，收敛速度在迭代次数 $n$ 中是指数的，并且较小的 Lipschitz 常数将产生更快的收敛。

除了可逆性之外，压缩的（contractive）残差块还会使残差层变为 bi-Lipschitz（双射Lipschitz）。



---

### Lemma 2<span id="lem2"></span>

（Forward and Inverse 正向和反向的 Lipschitz常数）。令 $ F(x)=x+g(x) $ 且 $ \operatorname{Lip}(g)=L<1 $ 表示残差层。然后，它保持
$$
\begin{equation}
 \operatorname{Lip}(F) \leq 1+L \quad \text{and} \quad \operatorname{Lip}\left(F^{-1}\right) \leq \frac{1}{1-L} 
\end{equation}
$$


因此通过设计，invertible ResNets 为它们的正向和反向映射都提供了稳定性保证。在以下部分中，我们讨论了增强 Lipschitz条件的方法。



---

### 2.1. Satisfying the Lipschitz Constraint

满足Lipschitz约束

我们将残差块实现为**收缩非线性** $ \phi $（例如ReLU，ELU，tanh）和**线性映射的组合**。

例如，在我们的卷积网络中，$ g = W_{3} \phi\left(W_{2} \phi\left(W_{1}\right)\right) $，其中 $W_i$ 是卷积层。因此，
$$
\begin{equation}
 \operatorname{Lip}(g)<1, \quad \text{if}\quad \left\|W_{i}\right\|_{2}<1 
\end{equation}
$$
其中 $ \|\cdot\|_{2} $ 表示谱范数 spectral norm。请注意，对 $g$ 的 Jacobian 谱范数进行正则化（Sokoli, 2017）仅会局部降低它，并不保证上述条件。因此，我们将对每一层强制执行 $ \left\|W_{i}\right\|_{2}<1 $。

如（Miyato, 2018）所述，如果滤波器内核大于 $1 × 1$，那么参数矩阵上的幂迭代仅近似于 $\| W_{i} \|_{2} $ 上的界限，而不是真实的谱范数，有关界限的详细信息，请参见（Tsuzuku,2018）。因此，与（Miyato, 2018）不同。我们按照（Gouk,2018）的建议通过使用 $ W_{i} $ 和 $ W_{i}^{1} $ 进行功率迭代来直接估计 $ W_{i} $ 的谱范数。幂迭代产生了一个低估计值 under-estimate $ \tilde{\sigma}_{i} \leq\left\|W_{i}\right\|_{2} $。 使用此估算值，我们通过下式进行标准化：
$$
\begin{equation}
 \tilde{W}_{i}=\left\{\begin{array}{ll}c W_{i} / \tilde{\sigma}_{i}, & \text { if } c / \tilde{\sigma}_{i}<1 \\ W_{i}, & \text { else }\end{array}\right. 
\end{equation}\tag{2}
$$
其中超参数 $c <1$ 是缩放系数。由于 $ \tilde{\sigma}_{i} $ 是一个低估计值，因此无法保证 $ \left\|W_{i}\right\|_{2} \leq c $ 满足。 但是，经过训练后（Sedghi, 2019）提供了一种在傅立叶变换参数矩阵上使用 SVD 精确检查 $ \left\|W_{i}\right\|_{2} $  的方法，这将使我们能够证明 $ \operatorname{Lip}(g)<1 $ 在所有情况下均成立。





---

## 3. Generative Modelling with i-ResNets

我们可以通过首先对 $ z \sim p_{z}(z) $ 进行采样（其中 $ z \in \mathbb{R}^{d} $）然后为某些函数 $ \Phi: \mathbb{R}^{d} \rightarrow \mathbb{R}^{d} $ 定义 $ x=\Phi(z) $ 来为数据 $ x \in \mathbb{R}^{d} $ 定义一个简单的生成模型。如果 $ \Phi$ 是可逆的，并且我们定义 $ F=\Phi^{-1} $，那么我们可以使用变量变化公式(change of variables formula)来计算该模型下任意 $x$ 的似然<span id="eq3"></span>
$$
\begin{equation}
 \ln p_{x}(x)=\ln p_{z}(z)+\ln \left|\operatorname{det} J_{F}(x)\right| 
\end{equation}\tag{3}
$$
其中 $ J_{F}(x) $ 是 $F$ 在 $x$ 处的雅可比行列式。这种形式的模型称为标准化流（Rezende，2015）。由于引入了功能强大的双射函数逼近器，它们最近已经成为高维数据的流行模型，其雅可比对数行列式可以被有效地计算（Dinh，2014; 2017; Kingma，2018; Chen，2018）或逼近（Grathwohl，2019）。

由于保证了 i-ResNets 是可逆的，因此我们可以使用它们在 [Eq. 3](#eq3) 中对 $F$ 进行参数化。 可以通过首先对 $ z \sim p(z) $ 进行采样，然后使用 [Algorithm 1](#algo1). 计算 $ x=F^{-1}(z) $ 来抽取该模型的样本。在 **Fig. 2** 中，我们与 Glow 相比，显示了使用 i-ResNet 在某些二维数据集上定义生成模型的示例（Kingma，2018）。

![1621944333639](assets/1621944333639.png)

**Fig. 2** Visual comparison of i-ResNet flow and Glow. Details of this experiment can be found in [Appendix C.3](#appC3).



---

### 3.1. Scaling to Higher Dimensions

缩放到更高的尺寸

尽管 i-ResNets 的可逆性使我们可以使用它们来定义归一化流，但我们必须计算 $ \ln \left|\operatorname{det} J_{F}(x)\right| $。 评估模型下的数据密度。计算此数量通常需要 $ \mathcal{O}\left(d^{3}\right) $ 的时间成本，这使得单纯地缩放至高维数据成为不可能。

为了绕过该约束，我们在 [Eq. 3](#eq3) 中给出了对数行列式项的易于处理的近似，它将按比例缩放到高维 $d$。 此前，（Ramesh，2018）将对数行列式估计应用于没有 i-ResNets 特定结构的不可逆深度生成模型。

首先，我们注意到 Lipschitz 约束恒等式的扰动 $x + g(x)$ 产生正行列式，因此
$$
\begin{equation}
 \left|\operatorname{det} J_{F}(x)\right|=\operatorname{det} J_{F}(x) 
\end{equation}
$$
参见 [Lemma 6](#lem6) in Appendix A。将这个结果与非奇异 $ A \in \mathbb{R}^{d \times d} $ 的矩阵恒等式 $ \ln \operatorname{det}(A)=\operatorname{tr}(\ln (A)) $ 结合起来（Withers，2010）
$$
\begin{equation}
 \ln \left|\operatorname{det} J_{F}(x)\right|=\operatorname{tr}\left(\ln J_{F}\right) 
\end{equation}
$$
其中 $\operatorname{tr}$ 表示矩阵的迹，$\ln $表示矩阵取对数。 因此，对于 $ z=F(x)=(I+g)(x) $，它有：
$$
\begin{equation}
 \ln p_{x}(x)=\ln p_{z}(z)+\operatorname{tr}\left(\ln \left(I+J_{g}(x)\right)\right) 
\end{equation}
$$
矩阵对数的迹可以表示为幂列（Hall，2015）<span id="eq4"></span>
$$
\begin{equation}
 \operatorname{tr}\left(\ln \left(I+J_{g}(x)\right)\right)=\sum_{k=1}^{\infty}(-1)^{k+1} \frac{\operatorname{tr}\left(J_{g}^{k}\right)}{k} 
\end{equation}\tag4
$$
当 $ \left\|J_{g}\right\|_{2}<1 $ 时，则列收敛。因此，由于Lipschitz约束，我们可以在保证收敛的情况下通过上述幂级数计算对数行列式。

在给出上述幂级数的随机近似值之前，我们观察到 i-ResNets 的以下属性：由于 $ \operatorname{Lip}\left(g_{t}\right)<1 $ 对于每层 $t$ 的残差块，我们可以在其对数行列式上提供上下限:
$$
\begin{equation}
 d \sum_{t=1}^{T} \ln \left(1-\operatorname{Lip}\left(g_{t}\right)\right) \leq \ln \left|\operatorname{det} J_{F}(x)\right| \\
 d \sum_{t=1}^{T} \ln \left(1+\operatorname{Lip}\left(g_{t}\right)\right) \geq \ln \left|\operatorname{det} J_{F}(x)\right| 
\end{equation}
$$
对于所有 $ x \in \mathbb{R} $ 都满足上式，请参见 [Lemma 7](#lem7) in Appendix A。因此，层数 $T$ 和 Lipschitz 常数都会影响 i-ResNets 的收缩和扩展范围，在设计此类体系结构时必须将其考虑在内。



----

### 3.2. Stochastic Approximation of log-determinant

对数行列式的随机逼近

用 [Eq. 4](#eq4) 中的幂级数来表达对数行列式有三个主要的计算缺陷 drawback：

- 1）计算 $ \operatorname{tr}\left(J_{g}\right) $ 的复杂度为 $ \mathcal{O}\left(d^{2}\right) $，或者大约需要对 $g$ 进行 $d$ 个评估，作为对角线的每个对角线项。Jacobian 需要计算 $g$ 的单独导数（Grathwohl，2019）。  
- 2）需要矩阵幂 $ J_{g}^{k} $，这需要完整的Jacobian知识。  
- 3）级数是无限的。

幸运的是，可以减轻缺陷 1 和 2。首先，向量-雅可比积 $ v^{T} J_{g} $ 可以通过与逆向模式自动微分求值 $g$ 近似相同的成本来计算。其次，$ A \in \mathbb{R}^{d \times d} $ 的矩阵迹的随机逼近
$$
\begin{equation}
 \operatorname{tr}(A)=\mathbb{E}_{p(v)}\left[v^{T} A v\right] 
\end{equation}
$$
称为 Hutchinsons 迹踪估算器，可用于估算 $ \operatorname{tr}\left(J_{g}^{k}\right) $。分布 $ p(v) $ 需要满足 $ \mathbb{E}[v]=0 $ 和 $ \operatorname{cov}(v)=1 $，参见（Hutchinson，1990； Avron，2011）。

虽然这允许对矩阵迹进行无偏估计，但要获得有限的计算成本，[Eq. 4](#eq4) 中的幂级数将在索引 $n$ 处被截断，以解决缺陷 3。 [Algorithm 2](#algo2) 总结了基本步骤。 截断将无偏估计量变为有偏估计量，其中偏置bias取决于截断误差truncation error。幸运的是，这个错误可以得到限制，如下所示。

<span id="algo2"></span>

![1621949113990](assets/1621949113990.png)

为了提高使用此估计器时优化的稳定性，我们建议使用具有连续导数的非线性激活，例如ELU（Clever，2015）或 softplus 的而不是 ReLU（[Appendix C.3](#appc3)）。



---

### 3.3. Error of Power Series Truncation

幂级数截断错误

我们估计ln | det（I + J g）| 与有限幂级数
$$
\tag{5}
$$
在哪里（有些滥用符号）PS（J g，∞）= tr（ln（I + J g））。 我们感兴趣的是将对数行列式的截断误差限制为数据维d，Lipschitz常数Lip（g）和级数n中项数的函数。



### Theorem 3

（损耗的近似误差）。 令g表示残差函数，J g表示雅可比行列式。 然后，在项n处的幂级数被截断的误差定为
$$

$$
尽管上面的结果给出了评估损耗的误差范围，但在训练过程中，损耗梯度的误差引起了人们的极大兴趣。 类似地，我们可以获得以下界限。 证明在附录A中给出。



### Theorem 4

（梯度近似的收敛速度）。   设θ∈R p表示网络F的参数，设g，J g如前。 此外，假设有界输入和具有Lipschitz导数的Lipschitz激活函数。 然后，我们得出收敛速度



其中c：= Lip（g），n为幂级数中使用的项数。

   在实践中，仅需采用5-10项即可获得每个尺寸小于.001位的偏差，通常报告其精度高达0.01精度（请参见附录E）。



## 4. Related Work

### 4.1. Reversible Architectures



---

### 4.2. Ordinary Differential Equations



---

### 4.3. Spectral Sum Approximations





## 5. Experiments



### 5.1. Validating Invertibility and Classification



### 5.2. Comparison with Other Invertible Architectures



### 5.3. Generative Modeling





## 6. Other Applications





## 7. Conclusions





## Acknowledgments





----

## Reference





## A. Additional Lemmas and Proofs





---

### Lemma 6<span id="lem6"></span>

（残差层的雅可比行列式为正行列式）。令 $ F(x)=(I+g(\cdot))(x) $ 表示残差层，且 $ J_{F}(x)=I+J_{g}(x) $ 表示其在 $ x \in \mathbb{R}^{d} $ 的雅可比行列。如果 $ \operatorname{Lip}(g)<1 $，则它对所有 $x$ 都保持 $ J_{F}(x) $ 的 $λ_i$ 为正，因此
$$
\begin{equation}
 \left|\operatorname{det}\left[J_{F}(x)\right]\right|=\operatorname{det}\left[J_{F}(x)\right] 
\end{equation}
$$
其中 $λ_i$ 表示特征值。



**Proof. (Lemma 6)**

首先，由于 $ \operatorname{Lip}(g)<1 $，因此对于所有的 $x$，我们有 $ \lambda_{i}\left(J_{F}\right)=\lambda_{i}\left(J_{g}\right)+1 $ 和 $ \left\|J_{g}(x)\right\|_{2}<1 $。

由于谱半径 $ \rho\left(J_{g}\right) \leq\left\|J_{g}\right\|_{2} $，所以 $ \left|\lambda_{i}\left(J_{g}\right)\right|<1 $ 。

因此有 $ \operatorname{Re}\left(\lambda_{i}\left(J_{F}\right)\right)>0 $，则得到 $ \operatorname{det} J_{F}=\prod_{i}\left(\lambda_{i}\left(J_{g}\right)+1\right)>0 $。



---

### Lemma 7<span id="lem7"></span>

（invertible ResNet 的对数行列式的上下界）。令 $ F_{\theta}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{d} $ 且 $ F_{\theta}= \left(F_{\theta}^{1} \circ \ldots \circ F_{\theta}^{T}\right) $ 表示具有块 $ F_{\theta}^{t}=I+g_{\theta_{t}} $ 的 invertible ResNet。 然后，我们可以获得以下界限
$$
\begin{equation}
 d \sum_{t=1}^{T} \ln \left(1-\operatorname{Lip}\left(g_{t}\right)\right) \leq \ln \left|\operatorname{det} J_{F}(x)\right|\\
 d \sum_{t=1}^{T} \ln \left(1+\operatorname{Lip}\left(g_{t}\right)\right) \geq \ln \left|\operatorname{det} J_{F}(x)\right| 
\end{equation}
$$
对于所有 $ x \in \mathbb{R}^{d} $ 都满足上式



**Proof. (Lemma 7)**

首先，各层的总和归因于函数组成，因为 $ J_{F}(x)=\prod_{t} J_{F^{t}}(x) $ 并且
$$
\begin{equation}
 \ln \left|\operatorname{det} J_{F}(x)\right|=\ln \left(\prod_{t=1}^{T} \operatorname{det} J_{F^{t}}(x)\right)=\sum_{t=1}^{T} \ln \operatorname{det} J_{F^{t}}(x) 
\end{equation}
$$
在这里我们使用了 [Lemma 6](#lem6) 行列式大于 0 的性质。此外，请注意，
$$
\begin{equation}
 \sigma_{d}(A)^{d} \leq \prod_{i} \sigma_{i}(A)=|\operatorname{det} A| \leq \sigma_{1}(A)^{d} 
\end{equation}
$$
对于矩阵 $A$ ，最大奇异值为 $\sigma_{1}$ 和最小奇异值为 $\sigma_{d}$。 此外，我们有 [Lemma 2](#lem2) 得出的 $ \sigma_{i}\left(J_{F^{t}}\right) \leq\left(1+\operatorname{Lip}\left(g_{t}\right)\right) $ 和 $ \sigma_{d}\left(J_{F^{t}}\right) \leq\left(1-\operatorname{Lip}\left(g_{t}\right)\right) $。将其插入并最终应用对数规则得出要求的范围。



---

## B. Verification of Invertibility



## C. Experimental Details



### C.3. Generative Modeling

<span id="appC3"></span>

**Toy Densities**

我们使用了 100 个残差块，其中每个残差连接是一个具有状态大小 2-64-64-64-2 的多层感知器和 ELU 非线性(Clevert，2015)。我们在每次残差块之后都使用 ActNorm (Glow) 。通过在训练期间构造完整的雅可比矩阵来精确计算对数密度的变化可视化。



**MNIST and CIFAR**

我们的生成模型的结构与Glow非常相似。该模型包括 “尺度块” 是以不同空间分辨率运行的 i-ResNet 块组。在每个 “尺度块” 之后，分开最后，我们执行压缩操作，将每个维度的空间分辨率降低 2 倍，并进行乘法运算通道数乘以 4 (可逆下采样)。

我们的 MNIST 和 CIFAR10 型号有三个尺度块。每个尺度块有 32 个 i-ResNet 块。每个 i-ResNet 块由 `3 × 3`、`1 × 1`、`3 × 3` 滤波器的三个卷积组成，其间具有 ELU 非线性。每个卷积层在 MNIST 模型中有 32 个滤波器，在CIFAR10 模型中有 512 个滤波器。

我们使用 Adamax (Kingma，2014)优化器进行了200个时期的训练，学习率为0.003。全程培训我们使用幂级数近似( [Eq. 4](#eq4)  )估计 [Eq. 3](#eq3)  中的对数行列式，其中十项用于 MNIST 模式和 CIFAR10 模式的 5 个术语。



**Evaluation**

在评估过程中，我们使用第3.3节中给出的界限来确定需要给出的术语数量偏差小于0 . 0001比特/秒的估计。然后，我们对来自哈钦森估计器的足够多的样本进行平均标准误差小于. 001 bit/dim，因此我们可以安全地报告我们模型的bit/dim精度，误差不超过.0002.



**Choice of Nonlinearity**

求对数行列式估计量的微分需要我们计算神经网络的输出。如果我们使用具有不连续导数的非线性，那么这些值是在某些地区没有定义。这会导致不稳定的优化。为了保证优化所需的数量总是存在的，我们建议使用具有连续导数的非线性，例如ELU (Clevert等人，2015)或softplus。在我们所有的实验中，我们都使用ELU。





---

## D. Fixed Point Iteration Analysis





## E. Evaluating the Bias of Our Log-determinant Estimator







## F. Additional Samples of i-ResNet flow