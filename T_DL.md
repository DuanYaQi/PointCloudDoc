# 深度学习工程师

由 deeplearning.ai 出品，网易引进的正版授权中文版深度学习工程师微专业课程，让你在了解丰富的人工智能应用案例的同时，学会在实践中搭建出最先进的神经网络模型，训练出属于你自己的 AI。



deeplearning.ai

https://www.coursera.org/learn/neural-networks-deep-learning?action=enroll



## 神经网络和深度学习

### 第一周 深度学习概论

#### 1.1. 欢迎来到深度学习工程微专业

#### 1.2. 什么是神经网络？

![1616421771173](assets/1616421771173.png)

输入为房屋面积 , 通过一个神经元（函数运算），然后输出房价 y

ReLU （Rectified Linear Unit，railu） 修正线性单元  修正是指取不小于0的值

![1616422154046](assets/1616422154046.png)

中间三个圈为隐藏单元，每个隐藏单元都来自自己学习到的权重，与输入加权求和。



---

#### 1.3. 用神经网络进行监督学习

监督学习的应用

实值估计，在线广告，



机智的选择输入和输出，解决特定问题，并把这部分学习过的组件嵌入到更大型的系统。

普通应用 对应 标准的神经网络NN

图像领域内，卷积神经网络 CNN

对于序列数据，循环神经网络 RNN

更复杂的应用 复杂的混合神经网络架构。



训练数据分为**结构化数据**和**非结构化数据**

结构化数据 	  每个特征都有清晰的定义。

非结构化数据   例如音频，图像，文本



好的网络能够同时适应结构化和非结构化数据



---

#### 1.4. 为什么深度学习会兴起？

普通的模型无法应用海量数据带来的益处，有时也无法处理海量数据，

而给规模足够大（有许多隐藏神经元）的神经网络输入海量数据，会增强performance



一些算法创新可以让神经网络运行效率更高，效果更好，是我们可以训练更大规模的网络。

![1616423575240](assets/1616423575240.png)

传统sigmod函数，让负值梯度趋近于零但不是零，学习会变得非常缓慢，因为当梯度接近0时，使用梯度下降法，参数会变化得很慢，学习也变得很慢。

而relu让负值梯度直接为0，直接不学习。加速梯度下降。

![1616423610014](assets/1616423610014.png)

很多时候，有了一个新想法，关于神经网络结构的想法，然后写代码实现想法，结果表现神经网络的效果，然后进一步赶紧神经网络结构的细节。



---

#### 1.5. 关于这门课

#### 1.6. 课程资源

coursea -> disscusion 



---

### 第二周 神经网络基础

#### 2.1. 二分分类

m个样本的训练集，遍历这个训练集，

正向过程/传播	forward pass/propagation

反向过程/传播	backward pass/propagation



计算机存储图像，用红绿蓝三个通道的矩阵表示。

![1616475094011](assets/1616475094011.png)

在进行网络训练时，通常要unroll或者reshape为一维向量。

![1616475156017](assets/1616475156017.png)

（x，y） 来表示一个单独的样本，x是n_x维的特征向量 $x \in \mathbb{R}^{n_x}$，y是标签值为 0 或 1

共有m个样本 ：$(x^{(1)},y^{(1)}) , (x^{(2)},y^{(2)}), \dots, (x^{(m)},y^{(m)})$



也可以用大写 $X$ 表示训练集

![1616474872943](assets/1616474872943.png)

m列表示m个样本，n_x行表示每个样本有n_x条特征，表示为 $X \in \mathbb{R}^{n_x \times m}$ 或者 `X.shape=(n_x,m)`，有时行列相反。

![1616475170737](assets/1616475170737.png)

m列表示m个样本，1行表示每个样本有1个输出标签，表示为 $Y \in \mathbb{R}^{1\times m}$ 或者 `Y.shape=(1,m)`



---

#### 2.2. logistic 回归

给输入 $x$ 希望输出 $\hat{y}$ 判断是不是一副 cat picture。一般  $\hat{y}$ 是一个概率，当输入特征x满足一定的条件时，y就是1。
$$
\hat{y} = P(y=1|x)
$$
输入 $X \in \mathbb{R}^{n_x \times m}$ ，logistic 参数  $w \in \mathbb{R}^{n_x}$  , $b \in \mathbb{R}$ 是一个实数。
$$
\hat{y} = w^Tx+b
$$
可能是一个上述的线性函数，但可能性不大，因为输出概率在0到1之间。

而 logistic 回归给一个 sigmoid 函数
$$
\hat{y} = \sigma (w^Tx+b)
$$
![1616476097923](assets/1616476097923.png)

输出为从 0 到 1 的光滑函数 $\sigma (z)$，其中在本例中 $z = w^Tx+b$
$$
\sigma (z) = \frac{1}{1-e^{-z}}
$$
如果 z 特别大，趋近于1；z 特别小，趋近于0。

神经网络学习 w 和 b 两个参数，通常 b 对应一个 intercepter 拦截器



---

#### 2.3. logistic 回归损失函数<span id="logistic"></span>

为了训练 w 和 b 两个参数，需要定义一个 loss function。给定输入$(x^{(1)},y^{(1)}) , (x^{(2)},y^{(2)}), \dots, (x^{(m)},y^{(m)})$ ，我们希望预测到的 $\hat{y}^{(i)} \approx  y^{(i)}$

我们可以定义损失函数，衡量预测值与实际值的差距，用误差平方不利于梯度下降，因为会将问题变成**非凸non-convex函数**（w形状，有多个局部最小值）。
$$
\begin{equation}
 \mathcal{L}(\hat{y}, y)=\frac{1}{2}(\hat{y}-y)^{2} 
\end{equation}
$$
换一种损失函数，**凸convex函数**（v形状，有一个全局最小值）。
$$
\begin{equation}
 \mathcal{L}(\hat{y}, y)=-(y \log \hat{y}+(1-y) \log (1-\hat{y})) 
\end{equation}
$$


如果 y = 1 时， $\mathcal{L}(\hat{y}, y)=- \log \hat{y}$。损失函数越小越好，即 $\log \hat{y}$ 越大越好，这时 $ \hat{y}$ 要接近 y 的值 1

如果 y = 0 时， $ \mathcal{L}(\hat{y}, y)= -\log (1-\hat{y})) $。损失函数越小越好， $\log (1-\hat{y}))$ 越大越好，这时 $ \hat{y}$ 要接近 y 的值 0



loss函数衡量了**单个**训练样本的表现。cost 函数衡量**全体**训练样本的表现。
$$
\begin{split}
 J(w, b)&=\frac{1}{m} \sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y^{(i)}\right)\\&=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \hat{y}^{(i)}+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right)\right]
\end{split}
$$
即损失函数的平均值。



---

#### 2.4. 梯度下降法

gradient descent

已知待训练sigmod函数： $ \hat{y}=\sigma\left(w^{T} x+b\right), \sigma(z)=\frac{1}{1+e^{-z}} $

成本函数： 
$$
\begin{split} J(w, b)&=\frac{1}{m} \sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y^{(i)}\right)\\&=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \hat{y}^{(i)}+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right)\right]\end{split} 
$$
找到合适的 w 和 b 让成本函数较小。

![image-20210324213334658](assets/image-20210324213334658.png)

**J(w,b) 是在水平轴 w 和 b 上的曲面，找到 J(w,b) 最小值对应的参数。**



方法:

用某个随即参数初始化一个点，朝最陡的方向走。

重复执行$ \omega=\omega-\alpha \frac{dJ(\omega)}{d \omega} $，直到算法收敛。其中 $\alpha$ 为学习率，控制每次迭代中梯度下降的步长，$\frac{dJ(\omega)}{d \omega}$ 是参数的更新量或变化量。

```c++
w = w - a * dw; // dw = deltaJ / deltaw;  dw是此点的导数 此点函数的斜率
b = b - a * db; // db = deltaJ / deltab;  pytorch自动求导
```



---

#### 2.5. 导数

derivatives

slope斜率 = 绿色极限三角形的高除以宽 = 0.003/0.001 = 3

![1616478167609](assets/1616478167609.png)

a1 = 2 			f(a1) = 6

a2 = 2.001 	f(a2) = 6.003

df = f(a2) - f(a1) / (a2 - a1) = 6.003 - 6 / (2.001 - 2) = 3

这个函数任何地方的斜率都是 3。



---

#### 2.6. 更多导数的例子

也就是复杂函数求导



---

#### 2.7. 计算图

computation graph

神经网络都是按照**前向**或者**反向传播**过程来实现的。

首先计算出神经网络的输出，紧接着进行一个**反向传输操作**。后者用来计算出对应的梯度或者导数。

$J(a,b,c) = 3(a + b * c)$ 是三个变量a,b,c的函数，我们可以设定`u = b*c`，`v = a + u`，`J = 3*v`，则有下图



![image-20210324205340803](assets/image-20210324205340803.png)



通过一个从左向右的过程，可以计算出 $J$ 的值。通过从右向左可以计算出导数。



---

#### 2.8. 计算图中的导数计算

按照上图计算，$J$ 对 $v$ 的导数，$\frac{dJ}{dv} = 3$。a的值改变，v的值就会改变，J的值也会改变。a改变，v改变量取决于 $\frac{dv}{da}$，

链式法则 $\frac{dJ}{da} = \frac{dJ}{dv}  \frac{dv}{da}$，$\frac{dJ}{db} = \frac{dJ}{dv}  \frac{dv}{du} \frac{du}{db}$，$\frac{dJ}{dc} = \frac{dJ}{dv}  \frac{dv}{du} \frac{du}{dc}$



---

#### 2.9. logistic回归中的梯度下降法

$$
z = w^Tx+b
$$

$$
\hat{y} = a =\sigma (z)
$$

$$
\mathcal{L}(\hat{y}, y)=-(y \log a+(1-y) \log (1-a))
$$

a是logistics函数的输出，y是标签真值。

如果有两个特征 $x_1$ 和 $x_2$ 则
$$
z = w_1^Tx_1+w_2^Tx_2 +b
$$
在logistic回归中，我们需要做的是，**变换参数**w和b来最小化损失函数，

![image-20210324211215820](assets/image-20210324211215820.png)

其中
$$
\frac{dL}{da} = -\frac{y}{a} + \frac{1-y}{1-a}
$$
其中
$$
\begin{split}\frac{dL}{dz} &= \frac{dL}{da} \frac{da}{dz}\\&=(-\frac{y}{a} + \frac{1-y}{1-a}) * (a(1-a))\\&=a-y\end{split}
$$
其中目标函数对三个参数的导数如下：
$$
\begin{split}
\frac{dL}{dw_1} &= x_1*\frac{dL}{dz}\\
\frac{dL}{dw_2} &= x_2*\frac{dL}{dz}\\
\frac{dL}{db} &= \frac{dL}{dz}
\end{split}
$$
然后根据下式更新参数。
$$
\begin{split}
w_1 &= w_1 - \alpha \frac{dL}{dw_1}\\
w_2 &= w_2 - \alpha \frac{dL}{dw_2}\\
b &= b - \alpha \frac{dL}{db}
\end{split}
$$

---

#### 2.10. m个样本的梯度下降

上一节均为单一样本的求导与参数更新。实际情况下，训练集会有很多样本。

$$
\begin{split}
 J(w, b)&=\frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)})\\&=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \hat{y}^{(i)}+(1-y^{(i)}) \log (1-\hat{y}^{(i)})\right]
\end{split}
$$
其中
$$
\hat{y}^{i} = a =\sigma (z^{i})=\sigma (w^Tx^{i}+b)
$$
直接求导
$$
\frac{\partial J(w, b)}{\partial w_1} = \frac{1}{m} \sum_{i=1}^{m}\frac{\partial L(\hat{y}^{(i)}, y^{(i)})}{\partial w_i}
$$
计算每一个样本的梯度值，然后求平均，会得到全局梯度值，可以直接用到梯度下降法。

![image-20210324214705487](assets/image-20210324214705487.png)

整个过程相当于一次epoch。每次将所有样本计算过一边后，梯度下降一次，更改参数。重复多次。

**显式的使用循环，会使算法很低效。**因此向量化编程有很大的帮助。



---

#### 2.11. 向量化

消除代码中显式for循环语句的艺术。**不能使用显式for循环**，numpy隐式循环。



 ```python
import numpy as np
import time 

# vectorization version
a = np.randrom.rand(1000000)
b = np.randrom.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.tiem()

print("Vectorized version:" + str(1000*(toc-tic)) + "ms")

# for loop version
c = 0
tic = time.time()
for i in range(1000000)
	c += a[i]*b[i]
toc = time.time()   
print("For loop version:" + str(1000*(toc-tic)) + "ms")  # 时间比向量化版本长

 ```

CPU 和 GPU 都有并行处理能力 **SIMD 单指令多数据流**



---

#### 2.12. 向量化的更多例子

计算
$$
\begin{split}
u &= Av\\
u_i &= \sum_i\sum_jA_{ij}v_j
\end{split}
$$

```python
u = np.zeros((n, 1))
for i 
	for j
    	u[i] = A[i][j] * v[j]

# 或
u = np.dot(A,v)       
```

计算
$$
u_i = e^{v_i}
$$

```python
u = np.zeros((n, 1))
for i in range(n)
	u[i] = math.exp(v[i])
# 或
u = np.exp(v);
np.log(v);
np.abs(v);
np.maximun(v,0) # v中所有元素和0之间相比的最大值
v**2			#v^2    
1/v     		#v的倒数
```



Logistic 回归求导

![1616648854396](assets/1616648854396.png)

``` python
J = 0, dw1 = 0, dw2 = 0, db = 0
for i = 1 to n:
    z[i] = w^T * x[i] + b
    a[i] = sigma(z[i])  #sigma  1 / (1 + e^-x)
    J += -[y[i] * log(yhat[i]) + (1 - y[i]) * log(1 - yhat[i])]
    dz = a[i] * (1 - a[i])   # dz = da/dz
    dw1 = x1[i] * dz[i]
    dw2 = x2[i] * dz[i]
    db += dz[i]
J = J / m
dw1 = dw1 / m
dw2 = dw2 / m
db = db / m
```



---

#### 2.13. 向量化 logistics 回归

```python
z = np.dot(w.t ,x) + b
a = 1 / np.exp(-z)
```

b 是一个实数，python会自动把实数b 扩展成一个` 1*m` 的行向量



---

#### 2.14. 向量化 logistics 回归的梯度输出

```python
dz = a - y;   # dz = dL/dz  不是   da/dz
dw = np.sum(np.dot(x, dz.t)) / m
db = np.sum(dz) / m
```



```python
# 总向量化编程logistics回归
z = np.dot(w.t ,x) + b
a = 1 / np.exp(-z)
dz = a - y;   # dz = dL/dz  不是   da/dz
dw = np.sum(np.dot(x, dz.t)) / m
db = np.sum(dz) / m

w = w - alpha * dw
b = b - alpha * db
```

仍然需要一个大 for 循环，实现每一次梯度更新，即eopch。



---

#### 2.15. Python 中的广播

```python
A = np.array([56 , 0 , 4.4 , 68],
            [1.2, 104, 52, 8],
            [1.8, 135, 99, 0.9])
cal = A.sum(axis = 0) #按列求和  按行求和axis = 1
percentage = 100 * A / cal.reshape(1,4)  #A 3x4         cal 1x4 
#.reshape(1,4) 可以确保矩阵形状是我们想要的

```

广播（broadcasting）即**同型复制**

**general principle**

size ：[m,n] +-*/ [1,n]  把 [1,n] 复制m行变成  [m,n]再和前项运算

size ：[m,n] +-*/ [m,1] 把 [m,1]复制n列变成 [m,n] 再和前项



---

#### 2.16. 关于 python/numpy 向量的说明

```python
# 避免使用秩为 1 的矩阵
import numpy as np

a = np.random.randn(5)

a.shape  #(5, ) 秩为1的数组 
np.dot(a, a.T)   # 算出来是内积 一个数

# 尽量不要使用秩为1的数组 即
a = np.random.randn(5) 
# 改为
a = np.random.randn(5, 1)

a.shape  #(5, 1) 秩不为1的数组 
np.dot(a, a.T)   # 算出来是外积 一个矩阵
```



```python
# 使用断言加以保障 执行很快 
assert(a.shape == (5,1))
```



```python
#确保形状
reshape
```



---

#### 2.17. Jupyter/Ipython笔记本的快速指南

shift + enter 执行代码段

kernel 重启内核

submit assignment 提交任务



---

#### 2.18 logistic 损失函数的[解释](#logistic)

$$
\hat{y} = \sigma (w^Tx+b)
$$

$$
\sigma (z) = \frac{1}{1-e^{-z}}
$$

我们设定
$$
\begin{equation}
 \hat{y}=P(y=1 \mid x) 
\end{equation}
$$
即算法的输出 $\hat{y}$ 是给定训练样本 x 条件下 y 等于 1 的概率。

换句话说，如果 y=1，那么在给定 x 得到 y=1的概率等于 $\hat{y}$ 

反过来说，如果 y=0，那么在给定 x 得到 y=0 的概率等于$1-\hat{y}$

下边有验证。



简单说  $\hat{y}$ 表示 y=1的概率。
$$
\begin{equation}
if \quad y=1: \quad p(y \mid x)=\hat{y} \\
if \quad y=0: \quad p(y \mid x)=1-\hat{y} 
\end{equation}
$$
二分类问题，y的取值只能是0或1。

0-1分布/二项分布/伯努利分布，上述两条公式可以合并成
$$
\begin{equation}
 p(y \mid x)=\hat{y}^{y}(1-\hat{y})^{(1-y)} 
\end{equation}
$$
当 y = 1或 y = 0 代入上式可以得到上上式的结论。



两边同时取**对数**，方便**展开**/**求导/优化**。
$$
\begin{equation}
 \log p\left(\left.y\right|x\right)=\log \hat{y}^{y}(1-\hat{y})^{(1-y)}=y \log \hat{y}+(1-y) \log (1-\hat{y}) 
\end{equation}
$$
概率为1时，log函数为0，概率为0时，log函数为负无穷。





假设所有样本**独立同分布**
$$
\begin{equation}
P= \prod_{i=1}^{m} p\left(y^{(i)} \mid x^{(i)}\right) 
\end{equation}
$$
由于各个样本**独立**，因此求得**全局最优**的条件便是求得**各样本最优**，也即各个样本取得**最优的概率的连乘**





两边同时取**对数**，方便**展开**/**求导/优化**。
$$
\begin{equation}
\log P= \sum_{i=1}^{m} \log p\left(y^{(i)} \mid x^{(i)}\right) 
\end{equation}
$$
最大似然估计，即求出一组参数，这里就是w和b，使这个式子取最大值。

也就是说这个式子最大值，$\hat{y}$ 和 $y$ 越接近，网络越好。



---









# 全栈深度学习训练营

https://www.bilibili.com/video/BV1BT4y1P7u6

https://fullstackdeeplearning.com/spring2021/

课外网站 http://neuralnetworksanddeeplearning.com/

每周阅读一章

---

## Week 1: Fundamentals

### Lecture 1: DL Fundamentals

**Neural Networks**

![image-20210324221541329](assets/image-20210324221541329.png)



受生物学启发，通过神经元对我们的身体进行所有计算。

![image-20210324221345170](assets/image-20210324221345170.png)

axon 轴突（神经细胞的突起，将信号发送到其他细胞）

synapse 突触（一个神经元的冲动传到另一个神经元或传到另一细胞间的相互接触的结构）

dendrite 树突(位于神经元末端的细分支，接收其他神经元传来的信号)

b就是一个偏差，因为这是想要的线性函数，对 y 截距的偏移。通过激活函数，变成非线性函数。

![image-20210324221858158](assets/image-20210324221858158.png)



神经元也就是感知器。如果将感知器如下分层排列。

![image-20210324222025335](assets/image-20210324222025335.png)

每一个感知器都有自己的权重 w 和偏差 y，这个网络表示了某个函数 $y =f(x)$

目的是让这个函数变得有用且正确。



---

**Universality**

万能近似定理（universal approximation theorem）

有一个连续函数 $f(x)$ ，如果一个两层神经网络有足够多的隐藏单元（即神经元），一定存在一组权重能够让神经网络无限近似函数$f(x)$



---

**Learning Problems**

- 无监督学习，了解数据的结构，从而了解输入。预测下一个单词，寻找相关关系，预测下一个像素，VAE，GAN，learn X

- 监督学习，图像识别，语音识别，机器翻译   learn X --> Y
- 强化学习，环境交互   learn to interact with environment  $x_t -> a_t, x_{t+1} -> a_{t+1}, ...$

![image-20210324222952463](assets/image-20210324222952463.png)



- 迁移学习
- 模仿学习
- 元学习



---

**Empirical Risk Minimization / Loss Function**

![image-20210324223817582](assets/image-20210324223817582.png)

线性规划，找到一条合适的线，表示这些数据的关系

最小化平方差
$$
\min _{w, b} \sum_{i=1}^{m}\left(w \cdot x^{(i)}+b-y^{(i)}\right)^{2} 
$$
在具体些，最小化损失函数
$$
\min _{w, b} \sum_{i=1}^{m} L\left(f_{w, b}\left(x^{(i)}\right), y^{(i)}\right)
$$
找到最好的参数，最优化损失函数。（MSE，Huber，cross-entropy）



---

**Gradient Descent**

更新参数$w_i$
$$
\begin{aligned}
w_{i} & \leftarrow w_{i}-\alpha \frac{\partial}{\partial w_{i}} \mathcal{L}(w, b) \\
\frac{\partial}{\partial w_{i}} \mathcal{L}(w, b) &=\lim _{\varepsilon \rightarrow 0} \frac{\mathcal{L}\left(w+\varepsilon e_{i}, b\right)-\mathcal{L}\left(w-\varepsilon e_{i}, b\right)}{2 \varepsilon}
\end{aligned}
$$


变换形式
$$
\begin{array}{l}
w \leftarrow w-\alpha \nabla_{w} \mathcal{L}(w, b) \\
\left(\nabla_{w} \mathcal{L}(w, b)\right)_{i}=\frac{\partial}{\partial w_{i}} \mathcal{L}(w, b)
\end{array}
$$
$\nabla$ 是场论中的符号,是矢量(向量)微分算符，所代表的的意义是：某一点上，变化最快的方向。实例：
$$
f (x,y,z) = 3xy + z^2 \\
∇f = (3y, 3x, 2z)
$$


数据在所有维度上均具有零均值和均等方差，这样梯度下降效果要好，可以让梯度最大程度的下降。

![image-20210324225104663](assets/image-20210324225104663.png)

调整策略有如下，重点看加粗部分：

- **Initialization** (more later)
- Normalization
  - **Batch norm**, weight norm, layer norm, ... (more later)
- Second order methods:
  - Exact:
    - Newton’s method
    - Natural gradient
  - Approximate second order methods:
    - Adagrad, **Adam**, Momentum



实际训练时，只计算一部分数据而不是整个数据的梯度
$$
w \leftarrow w-\alpha \nabla_{w} \sum_{i\in minibatch} L\left(w, b, x^{(i)}, y^{(i)}\right)
$$
批量梯度下降或随机梯度下降（Stochastic Gradient Descent）。因为可能一个参数适合一小批的学习和迭代。如果千万级别，取平均然后作为梯度值下降也没有意义。



---

**Backpropagation / Automatic Differentation**

链式求导法则
$$
f(x) = g(h(x))\\
f^{'}(x)=g^{'}(h(x))h^{'}(x)
$$
Automatic differentiation software 自动求导软件

- e.g. PyTorch, TensorFlow, Theano, Chainer, etc.
- Only need to program the function f(x,w).
- Software automatically computes all derivatives
- This is typically done by **caching info** during **forward** computation pass off, and then doing a backward pass = “**backpropagation**”



---

**Architectural Considerations (deep/conv/rnn)**

最简单的就是多层感知机，全连接



- Data efficiency: 数据效率
  - Extremely large networks can represent anything (see “universal function approximation theorem”) but might also need extremely large amount of data to latch onto（抓住） the right thing
  - -> Encode prior knowledge into the architecture, e.g.:
    - Computer vision: Convolutional Networks = spatial translation invariance 空间平移不变性
    - Sequence processing (e.g. NLP): Recurrent Networks = temporal invariance 时间不变性
- Optimization landscape / conditioning: 
  - Depth over Width 深度宽度
  - Skip connections 残差层
  - Batch / Weight / Layer Normalization 标准化处理
- Computational / Parameter efficiency 计算效率
  - Factorized convolutions 分解卷积
  - Strided convolutions 步长卷积



---

**CUDA / Cores of Compute**

神经网络计算只是矩阵乘法。



---

### Lab 1: Setup and Intro



---

## Week 2: CNNs

### Lecture 2A: CNNs



### Lecture 2B: Computer Vision Applications



### Lab 2: CNNs











