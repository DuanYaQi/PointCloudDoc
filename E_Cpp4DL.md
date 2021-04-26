# C++实现深度学习框架

## **书籍** 

C++模板元编程实战：一个深度学习框架的初步实现

https://github.com/bluealert/MetaNN-book

https://www.epubit.com/bookDetails?id=N39540&typeName=%E6%90%9C%E7%B4%A2



用Python实现深度学习框架

https://www.ituring.com.cn/book/2833



---

## **项目**

用C++和Python实现从头实现一个深度学习训练框架

https://github.com/Arctanxy/ToyNet



C++实现自己简单的深度学习框架

https://github.com/AifiHenryMa/MatrixNet



自己用c++搭的一个深度学习框架，自学用

https://github.com/jklp2/my_ai_engine



一个模仿caffe使用c++搭建的深度学习框架

https://github.com/xieqingxing/Piratical_caffe



自己开发的深度学习框架

https://github.com/luojiangtao/AADeepLearning_cpp



一些使用 Caffe 深度学习框架写的项目及理解

https://github.com/chaowang1994/Caffe-code

https://blog.csdn.net/ture_dream/article/details/53348301



Convolutional Neural Networks

https://github.com/pjreddie/darknet

Darknet是一个用C和CUDA编写的开源神经网络框架。 它快速，易于安装，并支持CPU和GPU计算。



A light deep learning tools box by c++

https://github.com/kymo/SUDL



chentianqi/TinyFlow: Build Your Own DL System in 2K Lines

https://github.com/tqchen/tinyflow



OneFlow is a performance-centered and open-source deep learning framework.

https://github.com/Oneflow-Inc/oneflow



tinynn

https://zhuanlan.zhihu.com/p/78713744



---

## **博客**

LazyNet-一款为理解深度学习而生的学习框架

https://blog.csdn.net/samylee/article/details/87194376

https://github.com/samylee/LazyNet



---

## **课程**

**UW**

在过去的几年中，深度学习已成为成功解决视觉，NLP，机器人技术等许多不同领域中问题的重要技术。 推动这一成功的重要因素是开发深度学习系统，该系统可有效地支持使用许多设备并可能使用分布式资源来学习和推理复杂模型的任务。 关于如何构建和优化这些深度学习系统的研究现在是研究和商业化的活跃领域，但是还没有涵盖该主题的课程。

本课程旨在填补这一空白。 我们将涵盖深度学习系统的各个方面，包括：深度学习的基础知识，用于表达机器学习模型的编程模型，自动区分，内存优化，调度，分布式学习，硬件加速，特定领域语言和模型服务。 其中许多主题与数据库，系统和网络，体系结构和编程语言中的现有研究方向相交。 目的是全面介绍深度学习系统的工作原理，讨论和执行可能的研究机会，并构建具有广泛吸引力的开源软件。

我们每周有两节课。 该课程将是讲座或实验室/讨论会。 每个讲座将研究深度学习系统的特定方面。 实验/讨论会议将包含实现该特定方面的教程，并将包括现有系统的案例研究，例如Tensorflow，Caffe，Mxnet，PyTorch等。

http://dlsys.cs.washington.edu/schedule



**UCB**

AI的最新成功很大程度上归功于硬件和软件系统的进步。 这些系统使得能够在越来越大的数据集上训练越来越复杂的模型。 在此过程中，这些系统还简化了模型开发，从而使机器学习社区得以快速发展。 这些新的硬件和软件系统包括新一代的GPU和硬件加速器（例如TPU和Nervana），Theano，TensorFlow，PyTorch，MXNet，Apache Spark，Clipper，Horovod和Ray等开源框架，以及无数的 仅在公司内部部署的一些系统。 同时，我们目睹了一系列ML / RL应用程序正在改进硬件和系统设计，作业调度，程序综合以及电路布局。
    在本课程中，我们将描述系统设计的最新趋势，以更好地支持下一代AI应用程序，以及AI应用程序以优化系统的体系结构和性能。 本课程的形式将是讲座，研讨会式的讨论和学生演讲的混合。 学生将负责阅读论文，并完成一个动手项目。 阅读材料将选自最近的会议记录和期刊。 对于项目，我们将强烈鼓励包含AI和系统专业学生的团队。

https://ucbrise.github.io/cs294-ai-sys-sp19/

---

## **难点**

### **想学着自己实现一个深度学习框架该如何入手？**

https://www.zhihu.com/question/329235391



1. 使用C++编写无缝切换在CPU与GPU之间的数据结构
2. 实现自动求导机制Auto Grad。
3. 编译成动态链接库，便于高级语言调用。
4. 封装底层算法，用高级语言声明方法。

其中比较困难的为GPU端的操作，需要熟悉CUDA编程，我当时将全连接实现完成后发现，对于很多深度学习库来说卷积或者LSTM操作都是封装在CUDNN中了，也就是调用一下NVIDIA的库即可，Thunder 框架也是到全连接实现完成就不再开发了，后期有时间将CUDNN 的一些操作也添加进来。





核心就是如何实现**自动微分框架**和**GPU算子**，可以参看[Tinyflow](https://link.zhihu.com/?target=https%3A//github.com/LB-Yu/tinyflow)，利用Python实现了自动微分框架，支持GPU加速。目前实现的算子可以建立BP网络，已经给出了在MNIST上训练的例子。如果有兴趣也可以在现有框架下直接添加卷积算子。下面两篇博客详细介绍了Tinyflow的实现细节和算法原理。

1. [Automatic Differentiation Based on Computation Graph](https://link.zhihu.com/?target=https%3A//lb-yu.github.io/2019/07/22/Automatic-Differentiation-Based-on-Computation-Graph/);
2. [Tinyflow - A Simple Neural Network Framework](https://link.zhihu.com/?target=https%3A//lb-yu.github.io/2019/07/23/Tinyflow-A-Simple-Neural-Network-Framework/)。

另外，推荐两门课程。

1. UC Berkeley的[AI-Sys Spring 2019](https://link.zhihu.com/?target=https%3A//ucbrise.github.io/cs294-ai-sys-sp19/)，偏理论，将现有的深度学习模型都过了一遍；
2. UW的[CSE 559W](https://link.zhihu.com/?target=http%3A//dlsys.cs.washington.edu/)，偏实战，两个lab设计非常不错，值得一做。





以下是我自己的几个步骤：

1. 首先先写你的数据类型，比如variable啊constant啊这些，然后把variable之间的计算都写一遍（例如加减乘除，矩阵乘法，element wise乘法）。写的同时把这个运算的 back prop 写了，这样你后面写 gradient 的时候会分方便。
2. 然后写计算图，你现在已经有了正反传播了，这个应该不难实现，可以用各种图的方式来写。
3. 然后写一些比较常用的函数，比如 sigmoid 啊，softmax 啊，包括他们的反向传播，为了之后搭网络方便。这里可能要推不少公式，我数学比较垃圾... 当时花了很久... 
4. 现在就可以写layer辣，例如FC、CNN、RNN。这里你可以先自己写写看，然后去看tf的源码看你的实现方式和他们是不是一样。当然，如果你最一开始的想法是六七层循环的那种，也可以先看看tf和torch的源码熟悉一下。
5. 写一些dropout辣这些琐碎的layer，我有朋友面试的时候被问到说怎么实现dropout，可以好好思考思考。
6. 一些你自己想实现的model，比如我是NLP组，所以还写了些embedding



深度学习框架的核心功能其实是**符号计算**，**自动求导**，**自动表达式优化**这些

如上面评论说dlsys有很多的简单教程

另外阅读一些轻量级框架也会有帮助，比如ngraph，mxnet等

其实眼界放宽点，像sympy这种符号计算框架也应该在学习的范畴里

把眼界放的更宽一点，这些应该属于编译器理论的内容，可以多了解一下编译器方面的书籍和内容，比如使用llvm自定义一门语言





平时用的深度学习框架包涵很广，gpu加速底层要和**CUDA**打交道，分布式多机多卡要和**网络**打交道，每一部分背后都是一个team在搞。

所以自己写的深度框架就限定在用 numpy 搭建网络各种**层**，**自动微分**，还有**训练评估**这些功能。本质都是对权重，op做各种操作。





### **编程达到什么水平才能编写出像caffe这样的深度学习框架？**

https://www.zhihu.com/question/297335439

1.高性能计算。这个当时时机已成熟，nvidia已经推出cuda和cudnn，只要调用就可以，并非最难。-----后来补充：如果当时没推出cudnn，那么每一个层计算都要用cuda加速，工作量和难度大增，同时运算效率要比cudnn低许多，还带来调试的困难。

2.存储和序列化设计。他为了减轻设计工作量，以及提高模型的通用性，也使用了谷歌的协议代码，还用了开源的内存数据库。里面的难度是模型格式的设计和规划，这不是代码或编程的难度，是软件工程上的难度，没有一定的开发经验是无法设计的，而一旦决定之后是极难更改的。

3.线程调度。这点比较小巧，对编程高手不算什么难度，贾大神应该是轻松搞定。

4.网络机制的设计。如何传递数据，空间效率时间效率问题，以及如何在显卡和内存传递数据。这个设计上并不复杂，他设计了blob来作为基本存储方式。

5.网络预处理，即初始化部分。这是最难的地方之一，一个是包容性，要包容各种类型的层。另一个是整体性，它是层计算、模型解释（把文本描述转化为数据结构）、过程控制的纽带，非常难以设计。我估计贾大神在net类中花了不少精力，修改试验了不少办法。

6.层注册。非常巧妙的设计，既与上面的谷歌开源协议有关，又与网络计算方式有关，使得许多代码变得清晰明了。

7.太多了，不写了。总体感觉caffe代码就像一个精致的艺术品，光凭手艺高超是不够的，还需要对当时的视觉方面的深度学习有深刻的理解，对软件工程有熟练的使用经验才能设计出来。特别是具有通用性的软件，因为通用，所以设计必须考虑一切可能，设计难度比个人自用提升十倍。





### 自动求导框架综述

https://www.jianshu.com/p/4c2032c685dc

https://github.com/zakheav/automatic-differentiation-framework









### 用C++开发深度学习框架，例如PyTorch这种，涉及到哪些技术栈？

https://www.zhihu.com/question/391004285

我们把目标定在写一个简单版本的tensorflow或pytorch，一个后端是c++前端是python的深度学习框架。除去你肯定要比较了解c++和python之外，主要需要关心的有以下几点：

存储

框架们都需要定义一个tensor类用于保存计算结果，只支持cpu的话就直接放内存就行；如果要支持gpu，就需要了解cuda和内存相关的api。很快你会发现，每次创建新tensor都cudaMalloc太慢了，所以需要像tensorflow一样运行前把所有显存都拿出来，自己做一个显存池分配，或者像pytorch一样不立即释放被垃圾回收的显存，而是缓存起来，给之后的新tensor用。那么就需要了解内存池相关的知识，例如tf就是在内部写了一个简单版本的dlmalloc。

计算

有了用来存储的结果的数据结构，我们就需要开始写进行计算的结构，一般称为kernel。对于gpu来说，常见的深度学习用的kernel都已经由nv写在cudnn中了，所以需要学一下cudnn的使用方法，而对于cpu，两个框架好像都用的是Eigen，所以学习下Eigen就好了～当然一些特殊的kernel需要手撸cuda或者c++，所以这俩应该要熟悉，好在框架们大多没有用SIMD这个级别的优化，所以暂时不用学和cpu指令集相关的内容。

有了这些kernel我们就可以尝试运行我们的框架了。那我们就要需要一个python前端，pytorch使用的是pybind11，tf是SWIG，貌似也有rfc说完转向pybind11，所以推荐学习一下pybind11。以及你会发现一个运算可能分别会有cpu，gpu两个kernel，这时你就需要在kernel之上加一层抽象，一般叫op，然后去根据配置dispatch kernel，这里可能会涉及一些c++的设计模式，也需要补充一下～

有了这些我们就可以像计算器一样用我们的框架了。但是缺了深度学习框架的很重要的一部分，autodiff，也就是自动求导。tensorflow和pytorch在自动求导上的设计很不一样，tensorflow是在python那边推断出来每个op的反向运算是哪个op，然后加入图中，在c++那边就不区分前向还是反向了，pytorch则是每个op在调用的时候会在c++里面注册一个类似回调函数的反向计算。这里的图是指运算和运算结果构成的DAG。这两者的差异也就牵扯到深度学习框架的两大派，动态图和静态图，简单来说动态图就是用户一边定义运算一边运行计算，静态图是用户先定义完整个图再运行。你可以上网看看这两者的对比，然后选一个。在选了之后，如果你用的是动态图，那么就像pytorch一样写反向计算，具体的运行时就交给python就好了～也就是让python来一条一条地运行你定义的计算，python帮你做垃圾回收。选了静态图的话，你就需要写一个运行时来跑你的图，写个拓扑排序加Eigen里面的线程池就好了，tensorflow差不多就是这么弄得，垃圾回收用引用计数。

截至到这里你就应该能有一个可以跑模型的框架了。

通信

通信算是选写的功能。主要有2个方向的：数据并行，就是好多个进程跑一个模型，让训练更快一点；模型并行，主要是让单卡放不下的模型在多卡上放下。前者你需要了解一下nccl和MPI，以及ring allreduce及其变种，他们是用于多个进程之间同步数据的；后者在单机内你研究好cuda的一些接口，多机你用grpc应该就行～

暂时想到这儿，手机打字，估计漏洞和错误挺多的，望指正。