
## Glow: Generative Flow with Invertible 1×1 Convolutions

NIPS2018

​	目前，生成对抗网络 GAN 被认为是在**图像生成**等任务上最为有效的方法，越来越多的学者正朝着这一方向努力：在计算机视觉顶会 CVPR 2018 上甚至有 8% 的论文标题中包含 GAN。近日来自 OpenAI 的研究科学家 Diederik Kingma 与 Prafulla Dhariwal 却另辟蹊径，提出了基于**流**的生成模型 Glow。据介绍，该模型不同于 GAN 与 VAE，而在生成图像任务上也达到了令人惊艳的效果。



### 摘要

​	由于可以追踪确切的对数似然度、潜在变量推断，以及训练与合成的可并行性，基于流的生成模型（Dinh et al., 2014）在概念上就很吸引人。在这篇论文中我们提出了 Glow，这是一种简单的使用**可逆** 1x1 卷积的生成流。使用该方法，我们展示了在标准基准上的对数似然度的显著提升。也许最引人注目的是，我们展示了仅通过普通的对数似然度目标优化，生成模型就可以高效地进行逼真图像的合成以及大尺寸图像的操作。



---

### 简介

​	Two major unsolved problems in the ﬁeld of machine learning are (1) data-efﬁciency: the ability to learn from few datapoints, like humans; and (2) generalization: robustness to changes of the task or its context. AI systems, for example, often do not work at all when given inputs that are different from their training distribution. A promise of generative models, a major branch of machine learning,is to overcome these limitations by: (1) learning realistic world models, potentially allowing agents to plan in a world model before actual interaction with the world, and (2) learning meaningful features of the input while requiring little or no human supervision or labeling. Since such features can be learned from large unlabeled datasets and are not necessarily task-speciﬁc, downstream solutions based on those features could potentially be more robust and more data efﬁcient. In this paper we work towards this ultimate vision, in addition to intermediate applications, by aiming to improve upon the state-of-the-art of generative models.

​	从机器学习数据的能力（2）方面来说，主要的两个问题是：数据的鲁棒性和人类的学习能力。例如，人工智能系统通常在输入不同于训练分布的情况下根本不工作。生成模型是机器学习的一个主要分支，它的一个承诺是克服这些局限性：（1）学习现实世界模型，潜在地允许智能体在与世界实际交互之前在世界模型中进行规划；（2）学习输入的有意义特征，同时几乎不需要或不需要人工监督或标记。由于这些特征可以从大型未标记的数据集中学习，并且不一定是任务特定的，基于这些特征的下游解决方案可能更健壮、更高效。在本文中，除了中间应用程序外，我们致力于通过改进生成模型的最新技术来实现这一终极目标。

