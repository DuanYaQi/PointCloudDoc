# 深度学习工程师

由 deeplearning.ai 出品，网易引进的正版授权中文版深度学习工程师微专业课程，让你在了解丰富的人工智能应用案例的同时，学会在实践中搭建出最先进的神经网络模型，训练出属于你自己的 AI。



deeplearning.ai

https://www.coursera.org/learn/neural-networks-deep-learning?action=enroll

https://study.163.com/my#/smarts

https://www.bilibili.com/video/av66644404





**note**

https://redstonewill.blog.csdn.net/article/details/78519599

https://redstonewill.blog.csdn.net/article/details/78600255

https://www.zhihu.com/column/DeepLearningNotebook

http://www.ai-start.com/dl2017/



**课后作业**

https://blog.csdn.net/u013733326/article/details/79827273

https://www.heywhale.com/mw/project/5e20243e2823a10036b542da





## Question

- [ ] 改善深层神经网络-[1.11 权重初始化](#winit)，





------

## 结构化机器学习项目

### 第一周 机器学习（ML）策略（1）

#### 1.1 为什么是 ML 策略

改善模型性能的方法：

- 收集更多的数据
- 收集多样的训练集
- 用梯度下降训练更久
- 试用更好的优化方法adam等
- 尝试更大/更小的网络
- 尝试 dropout
- 尝试添加 L2 正则化项
- 改变网络的架构



---

#### 1.2 正交化

orthogonalization

正交意味着互成 90 度，即每一个维度只控制其所能控制的变量，不会影响以另一维度为自变量的因变量。

1. 系统在训练集上表现的不错（训练更大的网络，切换更好的优化算法）
2. 紧接着希望在验证集上表现的不错（正则化，更大的数据集）
3. 然后在测试集上表现得不错（更大的验证集）
4. 最后在真实世界里表现良好（改变验证集或 cost 函数）

而这些调节方法（旋钮）只会对应一个“功能”，是正交的。

一般不用 early stopping。因为同时影响两个测试和验证，不具有独立性、正交性。



---

#### 1.3 单一数字评估指标

![1617684670783](assets/1617684670783.png)

precision 查准率：在模型识别出的所有猫里，有多少确实是猫

recall 查全率（召回率）：对于所有的猫，模型识别出了多少只

![1617684662336](assets/1617684662336.png)

![1617684684991](assets/1617684684991.png)



F1 score 为查准率和查全率的平均值，即单实数评估指标。
$$
\begin{equation}
 F_{1}=\frac{2}{\frac{1}{P}+\frac{1}{R}} =
 \frac{2 \cdot P \cdot R}{P+R} 
\end{equation}
$$
harmonic mean of P and R，调和平均

除了F1 Score之外，我们还可以使用平均值作为单实数评价指标来对模型进行评估。

![这里写图片描述](assets/20171113163112581)



验证集 +单一数字评估指标加速迭代过程



---

#### 1.4 满足和优化指标

Satisficing and Optimizing metrics

如果你要考虑 N 个指标，有时候选择其中 1 个指标做为优化指标是合理的。所以你想尽量优化那个指标；然后剩下 N-1 个指标都是满足指标，意味着只要它们达到一定阈值，例如运行时间快于100毫秒，但只要达到一定的阈值，你不在乎它超过那个门槛之后的表现，但它们必须达到这个门槛。

**N metric:** 1 <u>optimizing</u>, N-1 <u>satisficing</u>

![1617685956590](assets/1617685956590.png)

e.g. **maximize accuracy** subject to **running time < 100 ms**



---

#### 1.5 训练 / 验证 / 测试集划分

让验证集和测试集服从相同的分布 ← 随机地混排到验证/测试中 

选择一个验证集和测试集，以反映您期望在未来获得的数据，并考虑做好这些数据的重要性



----

#### 1.6 开发集合测试集的大小









#### 1.7 什么时候该改变开发 / 测试集和指标









#### 1.8 为什么是人的表现









#### 1.9 可避免偏差









#### 1.10 理解人的表现









#### 1.11 超过人的表现









#### 1.12 改善模型表现