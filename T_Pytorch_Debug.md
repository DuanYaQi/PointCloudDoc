# Debug

#### 1. bool value of Tensor with more than one value is ambiguous

函数或者可调用对象使用时候没有加括号。



#### 2. 注意：关于减少时间消耗

(1)只要是用到for循环都是在cpu上进行的，会消耗巨量的时间

(2)只要是用到生成矩阵这种操作都是在cpu上进行的，会很消耗时间。

(3)数据往cuda()上搬运会比较消耗时间，也就是说 .cuda() 会比较消耗时间，能去掉就去掉。

(4)在服务器上，如果可以在一块 gpu 上运行就不要采用 `net = nn.DataParallel(net)`，这种 gpu 并行方式比单个 gpu 要耗时。













# 吃掉Pytorch

## TODO

- [ ] 创建自己的数据集、自定义Dataset

- [ ] 详解Transformer （Attention Is All You Need） https://zhuanlan.zhihu.com/p/48508221
- [ ] FAIR何恺明等人提出组归一化：替代批归一化，不受批量大小限制 https://zhuanlan.zhihu.com/p/34858971
- [ ] Pytorch详解NLLLoss和CrossEntropyLoss  https://blog.csdn.net/qq_22210253/article/details/85229988
- [ ] log_softmax与softmax的区别在哪里？ https://www.zhihu.com/question/358069078
- [ ] PyTorch学习笔记——softmax和log_softmax的区别、CrossEntropyLoss() 与 NLLLoss() 的区别、log似然代价函数  https://blog.csdn.net/hao5335156/article/details/80607732
- [ ] Pytorch中Softmax和LogSoftmax的使用  https://zhuanlan.zhihu.com/p/137791367
- [ ] PyTorch 学习笔记（六）：PyTorch的十八个损失函数  https://zhuanlan.zhihu.com/p/61379965
- [ ] 《5分钟理解Focal Loss与GHM——解决样本不平衡利器》https://zhuanlan.zhihu.com/p/80594704

- [ ] 



---

## Start

如果是工程师，应该优先选 TensorFlow2.

如果是学生或者研究人员，应该优先选择 Pytorch.

如果时间足够，最好 TensorFlow2 和 Pytorch 都要学习掌握。

原因：

​	1，在工业界最重要的是模型落地，目前国内的大部分互联网企业只支持TensorFlow 模型的在线部署，不支持Pytorch。 并且工业界更加注重的是模型的高可用性，许多时候使用的都是成熟的模型架构，调试需求并不大。
​	2，研究人员最重要的是快速迭代发表文章，需要尝试一些较新的模型架构。而Pytorch 在易用性上相比 TensorFlow2 有一些优势，更加方便调试。 并且在2019年以来在学术界占领了大半壁江山，能够找到的相应最新研究成果更多。
​	3，TensorFlow2 和 Pytorch 实际上整体风格已经非常相似了，学会了其中一个，学习另外一个将比较容易。两种框架都掌握的话，能够参考的开源模型案例更多，并且可以方便地在两种框架之间切换。



## 1. Pytorch建模流程

​	1，准备数据

​	2，定义模型

​	3，训练模型

​	4，评估模型

​	5，使用模型

​	6，保存模型



---

### 1.1. 结构化数据建模

####　1.1.1. 准备数据

```python
import torch 
from torch import nn

#生成张量
a = torch.tensor([[2,1]])

#矩阵相乘 t()转置
c = a@b.t()
#矩阵点乘 
d = a*b.t()
#输出元素
c.item()

```



```python
from torch.utils.data import Dataset,DataLoader,TensorDataset

dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).float()),
                     shuffle = True, batch_size = 8)
dl_valid = DataLoader(TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).float()),
                     shuffle = False, batch_size = 8)


#Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作。
#TensorDataset()将数据包装成Dataset类


#DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)

#dataloader本质是一个可迭代对象，使用iter()访问，不能使用next()访问；
dataset(Dataset): 传入的数据集
shuffle在每个epoch开始的时候，对数据进行重新排序
batch_size每个batch有多少个样本
```



---

```python
import pandas as pd

#读取csv文件
dftrain_raw = pd.read_csv('/home/kesci/input/data6936/data/titanic/train.csv')
dftest_raw = pd.read_csv('/home/kesci/input/data6936/data/titanic/test.csv')
#显示数据的前五个
dftrain_raw.head()
#显示数据的前十个
dftest_raw.head(10)


#plot参数 https://blog.csdn.net/u012155582/article/details/100132543
    
#选择Survived列，全部计数   设置图像种类为柱状图bar
#figsize图像大小12*8   fontsize数字刻度字体大小15   rot设置轴标签（轴刻度）的显示旋转度数
ax = dftrain_raw['Survived'].value_counts().plot(kind = 'bar',figsize = (12,8),fontsize=15,rot = 0)

#设置y轴x轴标签和字体大小
ax.set_ylabel('Counts',fontsize = 15)
ax.set_xlabel('Survived',fontsize = 15)
#显示
plt.show()
```



```python
#选择Survived列，全部计数   设置图像种类为直方图hist bins分20类
#color颜色为purple  figsize图像大小12*8   fontsize数字刻度字体大小15   
ax = dftrain_raw['Age'].plot(kind = 'hist',bins = 20,color= 'purple',
                    figsize = (12,8),fontsize=15)
```



```python
#选择Survived列，全部计数   设置图像种类为'density'密度估计图
#figsize图像大小12*8   fontsize数字刻度字体大小15  
ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind = 'density',
                      figsize = (12,8),fontsize=15)
dftrain_raw.query('Survived == 1')['Age'].plot(kind = 'density',
                      figsize = (12,8),fontsize=15)
ax.legend(['Survived==0','Survived==1'],fontsize = 12)
```



```python
#数据预处理
def preprocessing(dfdata):

    dfresult= pd.DataFrame() 	#数据保存格式

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])  #one-hot encoding将离散型特征的每一种取值都看成一种状态，若你的这一特征中有N个不相同的取值，那么我们就可以将该特征抽象成N种不同的状态，one-hot编码保证了每一个取值只会使得一种状态处于“激活态”，也就是说这N种状态中只有一个状态位值为1，其他状态位都是0。r

    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1) #数据的拼接 横向连接，axis = 0

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)  #fillna 缺省值用0填充
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')#isna缺省值判断 astype类型转换

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True) #dummy_na : bool, default False增加一列表示空缺值，如果False就忽略空缺值
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return(dfresult)

    x_train = preprocessing(dftrain_raw).values
    y_train = dftrain_raw[['Survived']].values

    x_test = preprocessing(dftest_raw).values
    y_test = dftest_raw[['Survived']].values

    print("x_train.shape =", x_train.shape )
    print("x_test.shape =", x_test.shape )

    print("y_train.shape =", y_train.shape )
    print("y_test.shape =", y_test.shape )

```



------

#### 1.1.2. 定义模型

​	使用Pytorch通常有三种方式构建模型：使用**nn.Sequential按层顺序构建模型**；**继承nn.Module基类构建自定义模型**；**继承nn.Module基类构建模型并辅助应用模型容器进行封装**。

​	此处选择使用最简单的nn.Sequential，按层顺序模型。

```python
def create_net():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(15,20))
    net.add_module("relu1",nn.ReLU())
    net.add_module("linear2",nn.Linear(20,15))
    net.add_module("relu2",nn.ReLU())
    net.add_module("linear3",nn.Linear(15,1))
    net.add_module("sigmoid",nn.Sigmoid())
    return net
    
net = create_net()
print(net)
```



```python
#torchkeras 打印Pytorch模型结构和基本参数信息
from torchkeras import summary
summary(net,input_shape=(15,))
```



---

#### 1.1.3. 训练模型

​	Pytorch通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。

​	有3类典型的训练循环代码风格：**脚本形式训练循环**，**函数形式训练循环**，**类形式训练循环**。

​	此处介绍一种较通用的**脚本**形式。

```python
# 计算目标值和预测值之间的二进制交叉熵损失函数。默认对一个batch里面的数据做二元交叉熵并且求平均
loss_func = nn.BCELoss()
# Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
# 参数params-可用于迭代优化的参数或者定义参数组的dicts;   lr-学习率(默认: 1e-3)
optimizer = torch.optim.Adam(params=net.parameters(),lr = 0.01)
# 计算准确率的函数 lambda表达式
metric_func = lambda y_pred,y_true: accuracy_score(y_true.data.numpy(),y_pred.data.numpy()>0.5)
# 函数名为准确率
metric_name = "accuracy"
```



```python
epochs = 10  #循环次数
log_step_freq = 30 #

dfhistory = pd.DataFrame(columns = ["epoch","loss",metric_name,"val_loss","val_"+metric_name]) 
print("Start Training...")
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("=========="*8 + "%s"%nowtime)

for epoch in range(1,epochs+1):  

    # 1，训练循环-------------------------------------------------
    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1
    
    for step, (features,labels) in enumerate(dl_train, 1):
    
        # 梯度清零
        optimizer.zero_grad()

        # 正向传播求损失
        predictions = net(features)
        loss = loss_func(predictions,labels)
        metric = metric_func(predictions,labels)
        
        # 反向传播求梯度
        loss.backward()
        optimizer.step()

        # 打印batch级别日志
        loss_sum += loss.item()
        metric_sum += metric.item()
        if step%log_step_freq == 0:   
            print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                  (step, loss_sum/step, metric_sum/step))
            
    # 2，验证循环-------------------------------------------------
    net.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1

    for val_step, (features,labels) in enumerate(dl_valid, 1):
        # 关闭梯度计算
        with torch.no_grad():
            predictions = net(features)
            val_loss = loss_func(predictions,labels)
            val_metric = metric_func(predictions,labels)
        val_loss_sum += val_loss.item()
        val_metric_sum += val_metric.item()

    # 3，记录日志-------------------------------------------------
    info = (epoch, loss_sum/step, metric_sum/step, 
            val_loss_sum/val_step, val_metric_sum/val_step)
    dfhistory.loc[epoch-1] = info
    
    # 打印epoch级别日志
    print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
          "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") 
          %info)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
        
print('Finished Training...')
```



---

#### 1.1.4. 评估模型

​	查看1.1.3.中的记录日志dfhistory

```
dfhistory
```

​	绘制结果图

```python
import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
    
plot_metric(dfhistory,"loss")
plot_metric(dfhistory,"accuracy")
```



---

#### 1.1.5. 使用模型

```python
#预测概率
y_pred_probs = net(torch.tensor(x_test[0:10]).float()).data #直接调用net()函数
print(y_pred_probs)

#预测类别
y_pred = torch.where(y_pred_probs>0.5,
        torch.ones_like(y_pred_probs),torch.zeros_like(y_pred_probs))#0.5为界限 判断类别
print(y_pred)
```



---

#### 1.1.6. 保存模型

​	Pytorch 有两种保存模型的方式，都是通过调用pickle序列化方法实现的。

​	第一种方法只保存模型参数；第二种方法保存完整模型。

​	推荐使用第一种，第二种方法可能在切换设备和目录的时候出现各种问题。

##### 保存模型参数

```python
print(net.state_dict().keys())

torch.save(net.state_dict(), "./model/net_parameter.pkl")

net_clone = create_net()
net_clone.load_state_dict(torch.load("./model/net_parameter.pkl"))

net_clone.forward(torch.tensor(x_test[0:10]).float()).data

```



---

##### 保存完整模型

```python
torch.save(net, './model/net_model.pkl')
net_loaded = torch.load('./model/net_model.pkl')
net_loaded(torch.tensor(x_test[0:10]).float()).data
```



---

### 1.2. 图片数据建模流程范例

#### 1.2.1. 准备数据

​	cifar2 数据集为cifar10数据集的子集，只包括前两种类别 airplane 和 automobile。训练集有 airplane 和automobile 图片各5000张，测试集有 airplane 和 automobile 图片各1000张。

​	cifar2 任务的目标是训练一个模型来对飞机 airplane 和机动车 automobile 两种图片进行分类。



​	在Pytorch中构建图片数据管道通常有三种方法。

​	第一种是使用 torchvision 中的 **datasets.ImageFolder** 来读取图片然后用 DataLoader 来并行加载。第二种是通过继承 **torch.utils.data.Dataset** 实现用户自定义读取逻辑然后用 DataLoader 来并行加载。第三种方法是读取用户自定义数据集的通用方法，既可以读取图片数据集，也可以读取文本数据集。

​	本篇我们介绍第一种方法。

```python
import torch 
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets 

#串联多个图片变换的操作-图像变换 
#将transforms列表里面的transform操作进行遍历
transform_train = transforms.Compose([transforms.ToTensor()])
transform_valid = transforms.Compose([transforms.ToTensor()])

#root : 在指定的root路径下面寻找图片
#transform: 对PIL Image进行转换操作,transform 输入是loader读取图片返回的对象
#target_transform :对label进行变换
ds_train = datasets.ImageFolder("./cifar2/train/",
            transform = transform_train,target_transform= lambda t:torch.tensor([t]).float())
ds_valid = datasets.ImageFolder("./cifar2/test/",
            transform = transform_train,target_transform= lambda t:torch.tensor([t]).float())

print(ds_train.class_to_idx) #对应文件夹的label



dl_train = DataLoader(ds_train,batch_size = 50,shuffle = True,num_workers=3)
dl_valid = DataLoader(ds_valid,batch_size = 50,shuffle = True,num_workers=3)
'''
DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)

dataloader本质是一个可迭代对象，使用iter()访问，不能使用next()访问；
dataset(Dataset): 传入的数据集
shuffle在每个epoch开始的时候，对数据进行重新排序
batch_size每个batch有多少个样本
'''
```



```python
#展示一个块中的图像

plt.figure(figsize=(8,8))# 图片大小8*8
for i in range(9):
    img,label = ds_train[i]
    #将tensor的维度换位
    #比如图片img的size比如是（28，28，3）就可以利用img.permute(2,0,1)得到一个size为（3，28，28）的tensor
    img = img.permute(1,2,0) 
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label.item())
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```



```python
# Pytorch的图片默认顺序是 Batch,Channel,Width,Height
for x,y in dl_train:
    print(x.shape,y.shape) 
    break
    
torch.Size([50, 3, 32, 32]) torch.Size([50, 1])
```



---

#### 1.2.2. 定义模型

​	使用Pytorch通常有三种方式构建模型：使用**nn.Sequential按层顺序构建模型**；**继承nn.Module基类构建自定义模型**；**继承nn.Module基类构建模型并辅助应用模型容器进行封装**。

​	此处选择通过**继承nn.Module基类**构建自定义模型。

```python
#自适应最大池化 平均池化AdaptiveAvgPool2d
#输出张量的大小都是给定的output_size。例如输入张量大小为(1, 64, 8, 9)，设定输出大小为(5,7)，通过Adaptive Pooling层，可以得到大小为(1, 64, 5, 7)的张量。
pool = nn.AdaptiveMaxPool2d((1, 1))
t = torch.randn(10, 8, 32, 32)
print(pool(t).shape)    #10 8 1 1
```

​	Net类

```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
        self.dropout = nn.Dropout2d(p = 0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y
net = Net()
print(net)

torchkeras.summary(net,input_shape= (3,32,32))
```



---

#### 1.2.3. 训练模型

​	Pytorch通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。

​	有3类典型的训练循环代码风格：**脚本形式训练循环**，**函数形式训练循环**，**类形式训练循环**。

​	此处介绍一种较通用的**函数形式**训练循环。

```python
model = net
model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
model.loss_func = torch.nn.BCELoss()
model.metric_func = lambda y_pred,y_true: roc_auc_score(y_true.data.numpy(),y_pred.data.numpy())
model.metric_name = "auc"
```



```python
def train_step(model,features,labels):
    
    # 训练模式，dropout层发生作用
    model.train()
    
    # 梯度清零
    model.optimizer.zero_grad()
    
    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)

    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()

    return loss.item(),metric.item()

def valid_step(model,features,labels):
    
    # 预测模式，dropout层不发生作用
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions,labels)
        metric = model.metric_func(predictions,labels)
    
    return loss.item(), metric.item()


# 测试train_step效果
features,labels = next(iter(dl_train))
train_step(model,features,labels)
```



```python
def train_model(model,epochs,dl_train,dl_valid,log_step_freq):

    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns = ["epoch","loss",metric_name,"val_loss","val_"+metric_name]) 
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)

    for epoch in range(1,epochs+1):  

        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features,labels) in enumerate(dl_train, 1):

            loss,metric = train_step(model,features,labels)

            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step%log_step_freq == 0:   
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                      (step, loss_sum/step, metric_sum/step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features,labels) in enumerate(dl_valid, 1):

            val_loss,val_metric = valid_step(model,features,labels)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum/step, metric_sum/step, 
                val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
              "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") 
              %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)

    print('Finished Training...')
    
    return dfhistory
    
    
epochs = 20
dfhistory = train_model(model,epochs,dl_train,dl_valid,log_step_freq = 50)
print(dfhistory)
```



---

#### 1.2.4. 评估模型

```python
import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
    
plot_metric(dfhistory,"loss")
plot_metric(dfhistory,"auc")
```



---

#### 1.2.5. 使用模型

```python
def predict(model,dl):
    model.eval()
    with torch.no_grad():
        result = torch.cat([model.forward(t[0]) for t in dl])
    return(result.data)

#预测概率
y_pred_probs = predict(model,dl_valid)
print(y_pred_probs)

#预测类别
y_pred = torch.where(y_pred_probs>0.5,torch.ones_like(y_pred_probs),torch.zeros_like(y_pred_probs))
print(y_pred)
```



---

#### 1.2.6. 保存模型

```python
print(model.state_dict().keys())

# 保存模型参数

torch.save(model.state_dict(), "./model/model_parameter.pkl")

net_clone = Net()
net_clone.load_state_dict(torch.load("./model/model_parameter.pkl"))

predict(net_clone,dl_valid)
```



---

### 1.3. 文本数据建模流程范例

#### 1.3.1. 准备数据

​	imdb数据集的目标是根据电影评论的文本内容预测评论的情感标签。

​	训练集有20000条电影评论文本，测试集有5000条电影评论文本，其中正面评论和负面评论都各占一半。

​	文本数据预处理较为繁琐，包括中文切词（本示例不涉及），构建词典，编码转换，序列填充，构建数据管道等等。

​	在torch中预处理文本数据一般使用torchtext或者自定义Dataset，torchtext功能非常强大，可以构建文本分类，序列标注，问答模型，机器翻译等NLP任务的数据集。

​	下面仅演示使用它来构建文本分类数据集的方法。

​	较完整的教程可以参考以下知乎文章：《pytorch学习笔记—Torchtext》https://zhuanlan.zhihu.com/p/65833208

```python
#对数据进行处理，处理后 使用dataloader导入
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=4)

for features,labels in dl_train:
    print(features)
    print(labels)
    break
```



---

#### 1.3.2. 定义模型

​	使用Pytorch通常有三种方式构建模型：使用**nn.Sequential按层顺序构建模型**；**继承nn.Module基类构建自定义模型**；**继承nn.Module基类构建模型并辅助应用模型容器进行封装**。

​	此处选择通过**继承nn.Module基类构建模型并辅助应用模型容器进行封装**。

​	由于接下来使用类形式的训练循环，我们将模型封装成 **torchkeras.Model** 类来获得类似 Keras 中高阶模型接口的功能。

​	Model类实际上继承自nn.Module类。

```python
class Net(torchkeras.Model):

    def __init__(self):
        super(Net, self).__init__()

        # 设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量
        self.embedding = nn.Embedding(num_embeddings=MAX_WORDS, embedding_dim=3, padding_idx=1)
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5))
        self.conv.add_module("pool_1", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_1", nn.ReLU())
        self.conv.add_module("conv_2", nn.Conv1d(in_channels=16, out_channels=128, kernel_size=2))
        self.conv.add_module("pool_2", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_2", nn.ReLU())

        self.dense = nn.Sequential()
        self.dense.add_module("flatten", nn.Flatten())
        self.dense.add_module("linear", nn.Linear(6144, 1))
        self.dense.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        y = self.dense(x)
        return y
    
model = Net.Net()
print(model)

# torchkeras库里的的方法
model.summary(input_shape=(200,), input_dtype=torch.LongTensor)
```



---

#### 1.3.3. 训练模型

​	Pytorch通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。

​	有3类典型的训练循环代码风格：**脚本形式训练循环**，**函数形式训练循环**，**类形式训练循环**。

​	此处介绍一种较通用的**类形式**训练循环。

​	我们仿照 Keras 定义了一个高阶的模型接口 Model,实现 **fit, validate，predict, summary** 方法，相当于用户自定义高阶API。

```python
# 准确率
def accuracy(y_pred,y_true):
    y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                      torch.zeros_like(y_pred,dtype = torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc


# torchkeras库里的的方法
model.compile(loss_func = nn.BCELoss(), optimizer = torch.optim.Adagrad(model.parameters(),lr = 0.02),
             metrics_dict = {"accuracy":accuracy})

# torchkeras库里的的方法
dfhistory = model.fit(20, dl_train, dl_val=dl_test, log_step_freq= 200)
```



---

#### 1.3.4. 评估模型

```python
import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
    
plot_metric(dfhistory,"loss")
plot_metric(dfhistory,"accuracy")
# torchkeras库里的的方法
model.evaluate(dl_test)
```



---

#### 1.3.5. 使用模型

```python
# torchkeras库里的的方法
model.predict(dl_test)
```



---

#### 1.3.6. 保存模型

```python
print(model.state_dict().keys())

torch.save(model.state_dict(), "./data/model_parameter.pkl")

model_clone = Net()
model_clone.load_state_dict(torch.load("./data/model_parameter.pkl"))

model_clone.compile(loss_func = nn.BCELoss(),optimizer= torch.optim.Adagrad(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy})

# 评估模型
model_clone.evaluate(dl_test)
```



---

### 1.4. 时间序列数据建模流程范例

#### 1.4.1. 准备数据

​	2020年发生的新冠肺炎疫情灾难给各国人民的生活造成了诸多方面的影响。

​	有的同学是收入上的，有的同学是感情上的，有的同学是心理上的，还有的同学是体重上的。

​	本文基于中国2020年3月之前的疫情数据，建立时间序列RNN模型，对中国的新冠肺炎疫情结束时间进行预测。

```python
# 数据较小，可以将全部训练数据放入到一个batch中，提升性能
dl_train = DataLoader(ds_train, batch_size = 38)
```



---

#### 1.4.2. 定义模型

​	使用Pytorch通常有三种方式构建模型：使用**nn.Sequential按层顺序构建模型**；**继承nn.Module基类构建自定义模型**；**继承nn.Module基类构建模型并辅助应用模型容器进行封装**。

​	此处选择**第二种**方式构建模型。

​	由于接下来使用类形式的训练循环，我们将模型封装成 **torchkeras.Model** 类来获得类似 Keras 中高阶模型接口的功能。

​	Model类实际上继承自nn.Module类。

```python
import torch
from torch import nn 
import importlib 
import torchkeras 

torch.random.seed()

class Block(nn.Module):
    def __init__(self):
        super(Block,self).__init__()
    
    def forward(self,x,x_input):
        x_out = torch.max((1+x)*x_input[:,-1,:],torch.tensor(0.0))
        return x_out
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3层lstm
        self.lstm = nn.LSTM(input_size = 3,hidden_size = 3,num_layers = 5,batch_first = True)
        self.linear = nn.Linear(3,3)
        self.block = Block()
        
    def forward(self,x_input):
        x = self.lstm(x_input)[0][:,-1,:]
        x = self.linear(x)
        y = self.block(x,x_input)
        return y
        
net = Net()
model = torchkeras.Model(net)
print(model)

model.summary(input_shape=(8,3),input_dtype = torch.FloatTensor)
```

​	

---

#### 1.4.3. 训练模型

​	Pytorch通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。

​	有3类典型的训练循环代码风格：**脚本形式训练循环**，**函数形式训练循环**，**类形式训练循环**。

​	此处介绍一种较通用的**类形式**训练循环。

​	我们仿照 Keras 定义了一个高阶的模型接口 Model,实现 **fit, validate，predict, summary** 方法，相当于用户自定义高阶API。

​	注：循环神经网络调试较为困难，需要设置多个不同的学习率多次尝试，以取得较好的效果。

```python
def mspe(y_pred,y_true):
    err_percent = (y_true - y_pred)**2/(torch.max(y_true**2,torch.tensor(1e-7)))
    return torch.mean(err_percent)

model.compile(loss_func = mspe,optimizer = torch.optim.Adagrad(model.parameters(),lr = 0.1))

dfhistory = model.fit(100,dl_train,log_step_freq=10)
```



---

#### 1.4.4. 评估模型

评估模型一般要设置验证集或者测试集，由于此例数据较少，我们仅仅可视化损失函数在训练集上的迭代情况。

```python
import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.title('Training '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric])
    plt.show()
    
plot_metric(dfhistory,"loss")
```



---

#### 1.4.5. 使用模型

此处我们使用模型预测疫情结束时间，即新增确诊病例为 0 的时间。

```python
#使用dfresult记录现有数据以及此后预测的疫情数据
dfresult = dfdiff[["confirmed_num","cured_num","dead_num"]].copy()
dfresult.tail()
```



```python
#预测此后500天的新增走势,将其结果添加到dfresult中
for i in range(500):
    arr_input = torch.unsqueeze(torch.from_numpy(dfresult.values[-38:,:]),axis=0)
    arr_predict = model.forward(arr_input)

    dfpredict = pd.DataFrame(torch.floor(arr_predict).data.numpy(),
                columns = dfresult.columns)
    dfresult = dfresult.append(dfpredict,ignore_index=True)
```



```python
dfresult.query("confirmed_num==0").head()
# 第50天开始新增确诊降为0，第45天对应3月10日，也就是5天后，即预计3月15日新增确诊降为0
# 注：该预测偏乐观

dfresult.query("cured_num==0").head()
# 第365天开始新增治愈降为0，即大概1年后。
# 注: 该预测偏悲观，并且存在问题，如果将每天新增治愈人数加起来，将超过累计确诊人数。

dfresult.query("dead_num==0").head()
# 第54天开始新增确诊降为0
# 注：该预测偏乐观
```



---

#### 1.4.6. 保存模型

```python
print(model.net.state_dict().keys())

# 保存模型参数

torch.save(model.net.state_dict(), "./model/model_parameter1_4.pkl")
net_clone = Net()
net_clone.load_state_dict(torch.load("./model/model_parameter1_4.pkl"))
model_clone = torchkeras.Model(net_clone)
model_clone.compile(loss_func = mspe)

# 评估模型
model_clone.evaluate(dl_train)
```





---

## 2. Pytorch核心概念

​	Pytorch是一个基于Python的机器学习库。它广泛应用于计算机视觉，自然语言处理等深度学习领域。它主要提供了以下两种核心功能：

​		1，支持GPU加速的张量计算。

​		2，方便优化模型的自动微分机制。

​	Pytorch的主要优点：

- 简洁易懂：Pytorch的API设计的相当简洁一致。基本上就是**tensor, autograd, nn**三级封装。学习起来非常容易。
- 便于调试：Pytorch采用动态图，可以像普通Python代码一样进行调试。
- 强大高效：Pytorch提供了非常丰富的模型组件，可以快速实现想法。并且运行速度很快。

​    Pytorch底层最核心的概念是**张量**，**动态计算图**以及**自动微分**。

---

### 2.1. 张量数据结构

​	Pytorch的基本数据结构是**张量Tensor**。张量即**多维数组**。Pytorch的**张量**和numpy中的**array**很类似。

​	本节我们主要介绍张量的数据类型、张量的维度、张量的尺寸、张量和numpy数组等基本概念。

---

#### 2.1.1. 张量的数据类型

​	张量的数据类型和numpy.array基本一一对应，但是不支持str类型。包括：

torch.float64(torch.double),

**torch.float32(torch.float)**,

torch.float16,

torch.int64(torch.long),

torch.int32(torch.int),

torch.int16,

torch.int8,

torch.uint8,

torch.bool

​	一般神经网络建模使用的都是**torch.float32**类型。

```python
import numpy as np
import torch 

# 自动推断数据类型
i = torch.tensor(1);print(i,i.dtype)   	# tensor(1) torch.int64
x = torch.tensor(2.0);print(x,x.dtype)	# tensor(2.) torch.float32
b = torch.tensor(True);print(b,b.dtype)	# tensor(True) torch.bool

# 指定数据类型dtype
i = torch.tensor(1,dtype = torch.int32);print(i,i.dtype) # tensor(1) torch.int32
x = torch.tensor(2.0,dtype = torch.double);print(x,x.dtype) # tensor(2.) torch.double

# 使用特定类型构造函数IntTensor BoolTensor Tensor
i = torch.IntTensor(1);print(i,i.dtype) 	# tensor(1) torch.int32
x = torch.Tensor(np.array(2.0));print(x,x.dtype) #等价于torch.FloatTensor
b = torch.BoolTensor(np.array([1,0,2,0])); print(b,b.dtype)
```



---

#### 2.1.2. 张量的维度

​	不同类型的数据可以用不同维度(dimension)的张量来表示。

​	标量为0维张量，向量为1维张量，矩阵为2维张量；彩色图像有rgb三个通道，可以表示为3维张量；视频还有时间维，可以表示为4维张量。

​	可以简单地总结为：有几层中括号，就是多少维的张量。

```python
print("------------------------------------------------------------")
scalar = torch.tensor(True) # 标量，0维张量
print(scalar)
print(scalar.dim())  

print("------------------------------------------------------------")
vector = torch.tensor([1.0,2.0,3.0,4.0]) #向量，1维张量
print(vector)
print(vector.dim())

print("------------------------------------------------------------")
matrix = torch.tensor([[1.0,2.0],[3.0,4.0]]) #矩阵, 2维张量
print(matrix)
print(matrix.dim()) 

print("------------------------------------------------------------")   
tensor3 = torch.tensor([[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]])  # 3维张量
print(tensor3)
print(tensor3.dim())

print("------------------------------------------------------------")
tensor4 = torch.tensor([[[[1.0,1.0],[2.0,2.0]],[[3.0,3.0],[4.0,4.0]]],
                        [[[5.0,5.0],[6.0,6.0]],[[7.0,7.0],[8.0,8.0]]]])  # 4维张量
print(tensor4)
print(tensor4.dim())
```



---

#### 2.1.3. 张量的尺寸

​	可以使用 shape属性或者 size()方法查看张量在每一维的长度.

​	可以使用view方法改变张量的尺寸；如果view方法改变尺寸失败，可以使用reshape方法.

```python
print("------------------------------------------------------------")
scalar = torch.tensor(True)
print(scalar.size())   	# 没有维度 	  torch.Size([])
print(scalar.shape)
print("------------------------------------------------------------")
vector = torch.tensor([1.0,2.0,3.0,4.0])
print(vector.size())	# 1维 		torch.Size([4])
print(vector.shape)
print("------------------------------------------------------------")
matrix = torch.tensor([[1.0,2.0],[3.0,4.0]])
print(matrix.size()) 	# 2维		torch.Size([2,2])
print(matrix.shape)


torch.Size([])
torch.Size([])
------------------------------------------------------------
torch.Size([4])
torch.Size([4])
------------------------------------------------------------
torch.Size([2, 2])
torch.Size([2, 2])
```



```python
# 使用view可以改变张量尺寸

vector = torch.arange(0,12)
print(vector)
print(vector.shape)

matrix34 = vector.view(3,4) #1*12变为3*4
print(matrix34)
print(matrix34.shape)

matrix43 = vector.view(4,-1) #-1表示该位置长度由程序自动推断   4*3
print(matrix43)
print(matrix43.shape)


## output
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
torch.Size([12])

tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
torch.Size([3, 4])

tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
torch.Size([4, 3])
```



```python
matrix26 = torch.arange(0,12).view(2,6)
print(matrix26)
print(matrix26.shape)

# 转置操作让张量存储结构扭曲 导致其不连续
matrix62 = matrix26.t()
print(matrix62)
print(matrix62.is_contiguous())

# 直接使用view方法会失败，可以使用reshape方法
matrix34 = matrix62.view(3,4)    #error!!!!!!!!!!!!! 不连续所以不能view
matrix34 = matrix62.reshape(3,4) #等价于matrix34 = matrix62.contiguous().view(3,4)
print(matrix34)



## output
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
torch.Size([2, 6])
tensor([[ 0,  6],
        [ 1,  7],
        [ 2,  8],
        [ 3,  9],
        [ 4, 10],
        [ 5, 11]])
False
tensor([[ 0,  6,  1,  7],
        [ 2,  8,  3,  9],
        [ 4, 10,  5, 11]])
```



----

#### 2.1.4. 张量和numpy数组

​	可以用**numpy**方法从Tensor得到numpy数组，也可以用**torch.from_numpy**从numpy数组得到Tensor。

​	这两种方法关联的Tensor和numpy数组是**共享数据内存**的。如果改变其中一个，另外一个的值也会发生改变。如果有需要，可以用张量的**clone**方法拷贝张量，中断这种关联。

​	此外，还可以使用item方法从标量张量得到对应的Python数值。使用tolist方法从张量得到对应的Python数值列表。

```python
#torch.from_numpy函数从numpy数组得到Tensor

arr = np.zeros(3)
tensor = torch.from_numpy(arr)
print("before add 1:")
print(arr)          
print(tensor)

print("\nafter add 1:")
np.add(arr,1, out = arr) #给arr增加1，tensor也随之改变
print(arr)		
print(tensor)	


## output
before add 1:
tensor([0., 0., 0.])
[0. 0. 0.]

after add 1:
tensor([1., 1., 1.])
[1. 1. 1.]
```



```python
# numpy方法从Tensor得到numpy数组

tensor = torch.zeros(3)
arr = tensor.numpy()
print("before add 1:")
print(tensor)	#[0,0,0]
print(arr)		#[0,0,0]

print("\nafter add 1:")

#使用带下划线的方法表示计算结果会返回给调用 张量
tensor.add_(1) #给 tensor增加1，arr也随之改变 
#或： torch.add(tensor,1,out = tensor)
print(tensor)	#[1,1,1]
print(arr)		#[1,1,1]
```



```python
# 可以用clone() 方法拷贝张量，中断这种关联

tensor = torch.zeros(3)

#使用clone方法拷贝张量, 拷贝后的张量和原始张量内存独立
arr = tensor.clone().numpy() # 也可以使用tensor.data.numpy()
print("before add 1:")
print(tensor)		#[0,0,0]
print(arr)			#[0,0,0]

print("\nafter add 1:")

#使用 带下划线的方法表示计算结果会返回给调用 张量
tensor.add_(1) #给 tensor增加1，arr不再随之改变
print(tensor)		#[1,1,1]
print(arr)			#[0,0,0]
```



```python
# item方法和tolist方法可以将张量转换成Python数值和数值列表格式
scalar = torch.tensor(1.0)
s = scalar.item()
print(s)
print(type(s))

tensor = torch.rand(2,2)
t = tensor.tolist()
print(t)
print(type(t))
```



---

### 2.2. 自动微分机制

​	神经网络通常依赖**反向传播求梯度来更新网络参数**，求梯度过程通常是一件非常复杂而容易出错的事情。而深度学习框架可以帮助我们自动地完成这种求梯度运算。

​	Pytorch一般通过反向传播 **backward** 方法 实现这种求梯度计算。该方法求得的梯度将存在对应自变量张量的**grad**属性下。

​	除此之外，也能够调用 **torch.autograd.grad** 函数来实现求梯度计算。这就是Pytorch的自动微分机制。

---

#### 2.2.1. backward求导

​	backward 方法通常在一个标量张量上调用，该方法求得的梯度将存在对应自变量张量的grad属性下。

​	如果调用的张量非标量，则要传入一个和它同形状的 **gradient** 参数张量。

​	相当于用该 **gradient** 参数张量与调用张量作向量点乘，得到的标量结果再反向传播。

##### 2.2.1.1 标量的反向传播

```python
# f(x) = a*x**2 + b*x + c的导数

x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x + c 

#自动求导函数backward()
y.backward()
dy_dx = x.grad
print(dy_dx)  

''' output 
tensor(-2.)
'''
```



##### 2.2.1.2. 非标量的反向传播

```python
# f(x) = a*x**2 + b*x + c

x = torch.tensor([[0.0,0.0],[1.0,2.0]],requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x + c 

gradient = torch.tensor([[1.0,1.0],[1.0,1.0]])

print("x:\n",x)
print("y:\n",y)
y.backward(gradient = gradient)
x_grad = x.grad
print("x_grad:\n",x_grad)

''' output 
x:
 tensor([[0., 0.],
        [1., 2.]], requires_grad=True)
y:
 tensor([[1., 1.],
        [0., 1.]], grad_fn=<AddBackward0>)
x_grad:
 tensor([[-2., -2.],
        [ 0.,  2.]])
'''
```



##### 2.2.1.3. 非标量的反向传播可以用标量的反向传播实现

```python
# f(x) = a*x**2 + b*x + c

x = torch.tensor([[0.0,0.0],[1.0,2.0]],requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x + c 

gradient = torch.tensor([[1.0,1.0],[1.0,1.0]])
z = torch.sum(y*gradient) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!关键一句

print("x:",x)
print("y:",y)
z.backward()
x_grad = x.grad
print("x_grad:\n",x_grad)

''' output 
x: tensor([[0., 0.],
        [1., 2.]], requires_grad=True)
y: tensor([[1., 1.],
        [0., 1.]], grad_fn=<AddBackward0>)
x_grad:
 tensor([[-2., -2.],
        [ 0.,  2.]])
'''
```



---

#### 2.2.2. autograd.grad求导

```python
# f(x) = a*x**2 + b*x + c的导数
x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x + c


# create_graph 设置为 True 将允许创建更高阶的导数 
dy_dx = torch.autograd.grad(y,x,create_graph=True)[0]
print(dy_dx.data)

# 求二阶导数
dy2_dx2 = torch.autograd.grad(dy_dx,x)[0] 
print(dy2_dx2.data)

''' output 
tensor(-2.)
tensor(2.)
'''
```



```python
x1 = torch.tensor(1.0,requires_grad = True) # x需要被求导
x2 = torch.tensor(2.0,requires_grad = True)

y1 = x1*x2
y2 = x1+x2

# 允许同时对多个自变量求导数
(dy1_dx1,dy1_dx2) = torch.autograd.grad(outputs=y1,inputs = [x1,x2],retain_graph = True)
print(dy1_dx1,dy1_dx2)

# 如果有多个因变量，相当于把多个因变量y的梯度结果求和
(dy12_dx1,dy12_dx2) = torch.autograd.grad(outputs=[y1,y2],inputs = [x1,x2])
print(dy12_dx1,dy12_dx2)

''' output 
tensor(2.) tensor(1.)
tensor(3.) tensor(2.)
'''
```



---

#### 2.2.3. 自动微分和优化器求minimum

```python
# f(x) = a*x**2 + b*x + c的最小值

x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

#随机梯度下降算法 params 待优化参数的iterable或者是定义了参数组的dict
#lr 学习率
optimizer = torch.optim.SGD(params=[x],lr = 0.01)


def f(x):
    result = a*torch.pow(x,2) + b*x + c 
    return(result)

for i in range(500):
    optimizer.zero_grad()  #所有Variable的grad成员数值变为0
    y = f(x)
    y.backward()
    optimizer.step()       #所有Variable的grad成员和lr的数值自动更新Variable的数值
   
    
print("y=",f(x).data,";","x=",x.data)
''' output 
y= tensor(0.) ; x= tensor(1.0000)
'''
```



---

### 2.3. 动态计算图

​	本节我们将介绍 Pytorch的动态计算图。

---

#### 2.3.1. 简介

​	Pytorch的计算图由**节点**和**边**组成，节点表示**张量**或者**Function**，边表示张量和Function之间的**依赖关系**。

​	Pytorch中的计算图是动态图。这里的动态主要有两重含义。

​	第一层含义是：计算图的**正向传播**是**立即执行**的。无需等待完整的计算图创建完毕，每条语句都会在计算图中动态添加节点和边，并立即执行正向传播得到计算结果。

​	第二层含义是：计算图在**反向传播**后**立即销毁**。下次调用需要重新构建计算图。如果在程序中使用了backward方法执行了反向传播，或者利用 torch.autograd.grad 方法计算了梯度，那么创建的计算图会被立即销毁，释放存储空间，下次调用需要重新创建。

##### 2.3.1.1. 正向传播-立即执行

```python
import torch 
w = torch.tensor([[3.0,1.0]],requires_grad=True)
b = torch.tensor([[3.0]],requires_grad=True)
X = torch.randn(10,2)
Y = torch.randn(10,1)
Y_hat = X@w.t() + b  # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关  @ 矩阵乘积
loss = torch.mean(torch.pow(Y_hat-Y,2))  

print(loss.data)
print(Y_hat.data)

''' output 
tensor(18.5233)
tensor([[5.3596],
        [1.7647],
        [1.4672],
        [4.3500],
        [3.0064],
        [4.4400],
        [3.4316],
        [3.0412],
        [5.1182],
        [3.4482]])
'''
```



---

##### 2.3.1.2. 反向传播-立即销毁

```python
w = torch.tensor([[3.0,1.0]],requires_grad=True)
b = torch.tensor([[3.0]],requires_grad=True)
X = torch.randn(10,2)
Y = torch.randn(10,1)
Y_hat = X@w.t() + b  # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
loss = torch.mean(torch.pow(Y_hat-Y,2))

#计算图在反向传播后立即销毁，如果需要保留计算图, 需要设置retain_graph = True
loss.backward()  #loss.backward(retain_graph = True) 

loss.backward() #如果再次执行反向传播将报错
```



---

#### 2.3.2. 计算图中的Function

​	计算图中的张量我们已经比较熟悉了, 计算图中的另外一种节点是 **Function**, 实际上就是 Pytorch 中各种对张量操作的函数。

​	这些 Function 和我们 Python 中的函数有一个较大的区别，那就是它同时包括**正向计算**逻辑和**反向传播**的逻辑。

​	我们可以通过继承 **torch.autograd.Function** 来创建这种支持反向传播的 Function

```python
class MyReLU(torch.autograd.Function):
   
    #正向传播逻辑，可以用ctx存储一些值，供反向传播使用。
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)    #将输入input张量每个元素的夹紧到一个区间 最小为0

    #反向传播逻辑
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input+
    
    
w = torch.tensor([[3.0,1.0]],requires_grad=True)
b = torch.tensor([[3.0]],requires_grad=True)
X = torch.tensor([[-1.0,-1.0],[1.0,1.0]])
Y = torch.tensor([[2.0,3.0]])

relu = MyReLU.apply # relu现在也可以具有正向传播和反向传播功能
Y_hat = relu(X@w.t() + b)
loss = torch.mean(torch.pow(Y_hat-Y,2))

loss.backward()

print(w.grad)
print(b.grad)

# Y_hat的梯度函数即是我们自己所定义的 MyReLU.backward
print(Y_hat.grad_fn)

''' output 
tensor([[4.5000, 4.5000]])
tensor([[4.5000]])
<torch.autograd.function.MyReLUBackward object at 0x7fe3e366a588>
'''
```



---

#### 2.3.3. 计算图与反向传播

了解了Function的功能，我们可以简单地理解一下反向传播的原理和过程。理解该部分原理需要一些高等数学中求导链式法则的基础知识。

```python
import torch 

x = torch.tensor(3.0,requires_grad=True)
y1 = x + 1
y2 = 2*x
loss = (y1-y2)**2

loss.backward()
```

loss.backward() 语句调用后，依次发生以下计算过程。

1. loss 自己的 grad 梯度赋值为1，即对自身的梯度为1。

2. loss 根据其自身梯度以及关联的 backward 方法，计算出其对应的自变量即 y1 和 y2 的梯度，将该值赋值到 y1.grad 和 y2.grad。

3. y2 和 y1 根据其自身梯度以及关联的 backward 方法, 分别计算出其对应的自变量 x 的梯度，x.grad 将其收到的多个梯度值累加。

（注意，1,2,3步骤的求梯度顺序和**对多个梯度值的累加规则**恰好是求导链式法则的程序表述）



​	正因为求导链式法则衍生的梯度累加规则，张量的 grad 梯度不会自动清零，在需要的时候需要手动置零。

![img](/home/duan/windows/PointCloud/assets/20180822152951285)



---

#### 2.3.4. 叶子节点和非叶子节点

执行下面代码，我们会发现 loss.grad 并不是我们期望的 1,而是 None。类似地 y1.grad 以及 y2.grad 也是 None.这是由于它们不是叶子节点张量。

在反向传播过程中，只有 is_leaf=True 的叶子节点，需要求导的张量的导数结果才会被最后保留下来。

​	那么什么是叶子节点张量呢？叶子节点张量需要满足两个条件。

1. 叶子节点张量是由用户直接创建的张量，**而非由某个 Function 通过计算得到的张量**。

2. 叶子节点张量的 requires_grad 属性必须为True.

​    Pytorch设计这样的规则主要是为了节约内存或者显存空间，因为几乎所有的时候，用户只会关心他自己直接创建的张量的梯度。

​	所有依赖于叶子节点张量的张量, 其requires_grad 属性必定是True的，但其梯度值只在计算过程中被用到，不会最终存储到grad属性中。如果需要保留中间计算结果的梯度到grad属性中，可以使用 retain_grad 方法。如果仅仅是为了调试代码查看梯度值，可以利用 register_hook 打印日志。

```python
x = torch.tensor(3.0,requires_grad=True)
y1 = x + 1
y2 = 2*x
loss = (y1-y2)**2

loss.backward()
print("loss.grad:", loss.grad)
print("y1.grad:", y1.grad)
print("y2.grad:", y2.grad)
print(x.grad)

print(x.is_leaf)
print(y1.is_leaf)
print(y2.is_leaf)
print(loss.is_leaf)

''' output 
loss.grad: None
y1.grad: None
y2.grad: None
tensor(4.)
True
False
False
False
'''
```

​	利用 **retain_grad** 可以保留非叶子节点的梯度值/或利用 **register_hook** 可以查看非叶子节点的梯度值。

```python
#正向传播
x = torch.tensor(3.0,requires_grad=True)
y1 = x + 1
y2 = 2*x
loss = (y1-y2)**2

#非叶子节点梯度显示控制
y1.register_hook(lambda grad: print('y1 grad: ', grad))
y2.register_hook(lambda grad: print('y2 grad: ', grad))
loss.retain_grad()

#反向传播
loss.backward()
print("loss.grad:", loss.grad)
print("x.grad:", x.grad)
```



---

#### 2.3.5. 计算图在TensorBoard中的可视化

​	可以利用 torch.utils.tensorboard 将计算图导出到 TensorBoard 进行可视化。

```python

```





------

## 3. Pytorch层次结构

​	Pytorch中5个不同的层次结构：即**硬件层，内核层，低阶API，中阶API，高阶API**【torchkeras】。并以线性回归和DNN二分类模型为例，直观对比展示在不同层级实现模型的特点。

​	Pytorch的层次结构从低到高可以分成如下五层。

​		第一层为硬件层，Pytorch支持**CPU**、**GPU**加入计算资源池。

​		第二层为**C++**实现的**内核**。

​		第三层为**Python**实现的**操作符**，提供了**封装C++内核的低级API**指令，主要包括各种**张量操作算子、自动微分、变量管理**。如**torch.tensor，torch.cat，torch.autograd.grad，nn.Module**。如果把模型比作一个房子，那么第三层API就是【模型之砖】。

​		第四层为**Python**实现的**模型组件**，对低级API进行了函数封装，主要包括各种**模型层，损失函数，优化器，数据管道**等等。如**torch.nn.Linear，torch.nn.BCE，torch.optim.Adam，torch.utils.data.DataLoader**。如果把模型比作一个房子，那么第四层API就是【模型之墙】。

​		第五层为**Python**实现的**模型接口**。Pytorch没有官方的高阶API。为了便于训练模型，作者仿照keras中的模型接口，使用了不到300行代码，封装了Pytorch的高阶模型接口**torchkeras.Model**。如果把模型比作一个房子，那么第五层API就是模型本身，即【模型之屋】。

---

### 3.1. 低阶API示范

​	下面的范例使用Pytorch的低阶API实现线性回归模型和DNN二分类模型。

​	低阶API主要包括张量操作，计算图和自动微分。

```python
import os
import datetime

#打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

#mac系统上pytorch和matplotlib在jupyter中同时跑需要更改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
```



#### 3.1.1 线性回归模型

##### 3.1.1.1. 准备数据

```python
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
import torch
from torch import nn


#样本数量
n = 400

# 生成测试用数据集
X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0],[-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动


plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0].numpy(),Y[:,0].numpy(), c = "b",label = "samples")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)

ax2 = plt.subplot(122)
ax2.scatter(X[:,1].numpy(),Y[:,0].numpy(), c = "g",label = "samples")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)
plt.show()
```



##### 3.1.1.2. 打乱数据

```python
# 构建数据管道迭代器
def data_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  #样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        indexs = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield  features.index_select(0, indexs), labels.index_select(0, indexs)
        
# 测试数据管道效果   
batch_size = 8
(features,labels) = next(data_iter(X,Y,batch_size))
print(features)
print(labels)
```



##### 3.1.1.3. 网络模型

```python
class LinearRegression: 
    
    def __init__(self):
        self.w = torch.randn_like(w0,requires_grad=True)
        self.b = torch.zeros_like(b0,requires_grad=True)
        
    #正向传播
    def forward(self,x): 
        return x@self.w + self.b

    # 损失函数
    def loss_func(self,y_pred,y_true):  
        return torch.mean((y_pred - y_true)**2/2)

model = LinearRegression()
```



##### 3.1.1.4. 训练模型

```python
# 训练一次，目的是loss最小，沿着loss下降最快的地方更新w和b的值，即梯度下降
def train_step(model, features, labels):
    
    predictions = model.forward(features)
    loss = model.loss_func(predictions,labels)
        
    # 反向传播求梯度
    loss.backward()
    
    # 使用torch.no_grad()避免梯度记录，也可以通过操作 model.w.data 实现避免梯度记录 
    with torch.no_grad():
        # 梯度下降法更新参数
        model.w -= 0.001*model.w.grad
        model.b -= 0.001*model.b.grad

        # 梯度清零
        model.w.grad.zero_()
        model.b.grad.zero_()
    return loss
```



```python
# 训练函数
def train_model(model,epochs):
    for epoch in range(1,epochs+1):
        for features, labels in data_iter(X,Y,10):   # 打乱数据 10为分块数目
            loss = train_step(model,features,labels)

        if epoch%200==0:
            printbar()
            print("epoch =",epoch,"loss = ",loss.item())
            print("model.w =",model.w.data)
            print("model.b =",model.b.data)
```



```python
# 测试train
train_model(model,epochs = 1000)
```



```python
#可视化
plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0].numpy(),Y[:,0].numpy(), c = "b",label = "samples")
ax1.plot(X[:,0].numpy(),(model.w[0].data*X[:,0]+model.b[0].data).numpy(),"-r",linewidth = 5.0,label = "model") # 画出模型的结果
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)


ax2 = plt.subplot(122)
ax2.scatter(X[:,1].numpy(),Y[:,0].numpy(), c = "g",label = "samples")
ax2.plot(X[:,1].numpy(),(model.w[1].data*X[:,1]+model.b[0].data).numpy(),"-r",linewidth = 5.0,label = "model") # 画出模型的结果
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)

plt.show()
```



---

#### 3.1.2. DNN二分类模型

##### 3.1.2.1. 准备数据

```python
#正负样本数量
n_positive,n_negative = 2000,2000

#生成正样本, 小圆环分布
r_p = 5.0 + torch.normal(0.0,1.0,size = [n_positive,1]) 
theta_p = 2*np.pi*torch.rand([n_positive,1])
Xp = torch.cat([r_p*torch.cos(theta_p),r_p*torch.sin(theta_p)],axis = 1)
Yp = torch.ones_like(r_p)

#生成负样本, 大圆环分布
r_n = 8.0 + torch.normal(0.0,1.0,size = [n_negative,1]) 
theta_n = 2*np.pi*torch.rand([n_negative,1])
Xn = torch.cat([r_n*torch.cos(theta_n),r_n*torch.sin(theta_n)],axis = 1)
Yn = torch.zeros_like(r_n)

#汇总样本
X = torch.cat([Xp,Xn],axis = 0)
Y = torch.cat([Yp,Yn],axis = 0)

#可视化
plt.figure(figsize = (6,6))
plt.scatter(Xp[:,0].numpy(),Xp[:,1].numpy(),c = "r")
plt.scatter(Xn[:,0].numpy(),Xn[:,1].numpy(),c = "g")
plt.legend(["positive","negative"])
plt.show()
```



##### 3.1.2.2. 打乱数据

```python
# 构建数据管道迭代器
def data_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  #样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        indexs = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield  features.index_select(0, indexs), labels.index_select(0, indexs)
        
# 测试数据管道效果   
batch_size = 8
(features,labels) = next(data_iter(X,Y,batch_size))
print(features)
print(labels)
```



##### 3.1.2.3. 网络模型

​	此处范例我们利用nn.Module来组织模型变量。

```python
class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.w1 = nn.Parameter(torch.randn(2,4))
        self.b1 = nn.Parameter(torch.zeros(1,4))
        self.w2 = nn.Parameter(torch.randn(4,8))
        self.b2 = nn.Parameter(torch.zeros(1,8))
        self.w3 = nn.Parameter(torch.randn(8,1))
        self.b3 = nn.Parameter(torch.zeros(1,1))

    # 正向传播
    def forward(self,x):
        x = torch.relu(x@self.w1 + self.b1)
        x = torch.relu(x@self.w2 + self.b2)
        y = torch.sigmoid(x@self.w3 + self.b3)
        return y
    
    # 损失函数(二元交叉熵)
    def loss_func(self,y_pred,y_true):  
        #将预测值限制在1e-7以上, 1- (1e-7)以下，避免log(0)错误
        eps = 1e-7
        y_pred = torch.clamp(y_pred,eps,1.0-eps) # 数值限制在一个给定的区间[min, max]内：
        bce = - y_true*torch.log(y_pred) - (1-y_true)*torch.log(1-y_pred)
        return torch.mean(bce)
    
    # 评估指标(准确率)
    def metric_func(self,y_pred,y_true):
        y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                          torch.zeros_like(y_pred,dtype = torch.float32))
        acc = torch.mean(1-torch.abs(y_true-y_pred))
        return acc
    
model = DNNModel()
```



```python
# 测试模型结构
batch_size = 10
(features,labels) = next(data_iter(X,Y,batch_size))

predictions = model(features)

loss = model.loss_func(labels,predictions)
metric = model.metric_func(labels,predictions)

print("init loss:", loss.item())
print("init metric:", metric.item())

len(list(model.parameters())) 

''' output
init loss: 7.618113040924072
init metric: 0.5249226689338684
6
''' 
```



##### 3.1.2.4. 训练模型

```python
def train_step(model, features, labels):   
    
    # 正向传播求损失
    predictions = model.forward(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)
        
    # 反向传播求梯度
    loss.backward()
    
    # 梯度下降法更新参数
    for param in model.parameters():
        #注意是对param.data进行重新赋值,避免此处操作引起梯度记录
        param.data = (param.data - 0.01*param.grad.data) 
        
    # 梯度清零
    model.zero_grad()
        
    return loss.item(),metric.item()
 

def train_model(model,epochs):
    for epoch in range(1,epochs+1):
        loss_list,metric_list = [],[]
        for features, labels in data_iter(X,Y,20):
            lossi,metrici = train_step(model,features,labels)
            loss_list.append(lossi)
            metric_list.append(metrici)
        loss = np.mean(loss_list)
        metric = np.mean(metric_list)

        if epoch%100==0:
            printbar()
            print("epoch =",epoch,"loss = ",loss,"metric = ",metric)
        
train_model(model,epochs = 1000)
```





---

### 3.2. 中阶API示范

#### 3.2.1. 线性回归模型

##### 3.2.1.1 准备数据

​	同3.1.1.1. 准备数据

---

##### 3.2.1.2. 打乱数据

```python
ds = TensorDataset(X,Y)
dl = DataLoader(ds,batch_size = 10,shuffle=True,num_workers=2)
```



----

##### 3.2.1.3. 网络模型

```python
model = nn.Linear(2,1) #线性层
model.loss_func = nn.MSELoss() #损失函数
model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01) #SGD优化器
```



##### 3.2.1.4. 训练模型

```python
def train_step(model, features, labels):
    
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    # 反向传播求梯度
    loss.backward()

    # 避免梯度记录
    model.optimizer.step()
    model.optimizer.zero_grad()
    return loss.item()

# 测试train_step效果
features,labels = next(iter(dl))
train_step(model,features,labels)
```



```python
def train_model(model,epochs):
    for epoch in range(1,epochs+1):
        for features, labels in dl:
            loss = train_step(model,features,labels)
        if epoch%50==0:
            printbar()
            w = model.state_dict()["weight"]
            b = model.state_dict()["bias"]
            print("epoch =",epoch,"loss = ",loss)
            print("w =",w)
            print("b =",b)
train_model(model,epochs = 200)
```



---

#### 3.2.2. DNN二分类模型

##### 3.2.2.1. 准备数据

​	同3.2.1.1. 准备数据

---

##### 3.2.2.2. 打乱顺序

```python
#构建输入数据管道
ds = TensorDataset(X,Y)
dl = DataLoader(ds,batch_size = 10,shuffle=True,num_workers=2)
```



##### 3.2.2.3. 网络模型

```python
class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(2,4)  
        self.fc2 = nn.Linear(4,8) 
        self.fc3 = nn.Linear(8,1)
        
    # 正向传播
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = nn.Sigmoid()(self.fc3(x))
        return y
    
    # 损失函数 Binary 交叉熵
    def loss_func(self,y_pred,y_true):
        return nn.BCELoss()(y_pred,y_true)
    
    # 评估函数(准确率)
    def metric_func(self,y_pred,y_true):
        y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                          torch.zeros_like(y_pred,dtype = torch.float32))
        acc = torch.mean(1-torch.abs(y_true-y_pred))
        return acc
    
    # 优化器
    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(),lr = 0.001)
    
model = DNNModel()

'''
	self.w1 = nn.Parameter(torch.randn(2,4))
	self.b1 = nn.Parameter(torch.zeros(1,4))
替换为
	self.fc1 = nn.Linear(2,4) 



	torch.relu(x@self.w1 + self.b1)
替换为
    F.relu(self.fc1(x))

	torch.sigmoid(x@self.w3 + self.b3)
替换为
	nn.Sigmoid()(self.fc3(x))
'''
```



##### 3.2.2.4. 训练模型

```python
def train_step(model, features, labels):
    
    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)
    
    # 反向传播求梯度
    loss.backward()
    
    # 更新模型参数
    model.optimizer.step()
    model.optimizer.zero_grad()
    
    return loss.item(),metric.item()

# 测试train_step效果
features,labels = next(iter(dl))
train_step(model,features,labels)
```



```python
def train_model(model,epochs):
   同3.2.1.4. 训练模型 def train_model()
        
train_model(model,epochs = 300)
```





---

### 3.3. 高阶API示范

#### 3.3.1. 线性回归模型

##### 3.3.1.1. 准备数据

​	同3.1.1.1. 准备数据

---

##### 3.3.1.2. 打乱数据

```python
'''
    打乱数据
'''
#构建输入数据管道
ds = TensorDataset(X,Y)
ds_train,ds_valid = torch.utils.data.random_split(ds,[int(400*0.7),400-int(400*0.7)]) # 划分训练集和数据集
# 各自的索引以indices储存在ds_train/valid

dl_train = DataLoader(ds_train,batch_size = 10,shuffle=True,num_workers=2)
dl_valid = DataLoader(ds_valid,batch_size = 10,num_workers=2)
```





##### 3.3.1.3. 网络模型

```python
from torchkeras import Model
class LinearRegression(Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(2,1)
    
    def forward(self,x):
        return self.fc(x)

model = LinearRegression()
model.summary(input_shape = (2,)
```



##### 3.3.1.4. 训练模型

```python
### 使用fit方法进行训练
def mean_absolute_error(y_pred,y_true):
    return torch.mean(torch.abs(y_pred-y_true))

def mean_absolute_percent_error(y_pred,y_true):
    absolute_percent_error = (torch.abs(y_pred-y_true)+1e-7)/(torch.abs(y_true)+1e-7)
    return torch.mean(absolute_percent_error)

model.compile(loss_func = nn.MSELoss(),
              optimizer= torch.optim.Adam(model.parameters(),lr = 0.01),
              metrics_dict={"mae":mean_absolute_error,"mape":mean_absolute_percent_error})

dfhistory = model.fit(200,dl_train = dl_train, dl_val = dl_valid,log_step_freq = 20)
```





##### 3.3.1.5. 可视化训练过程

```python
dfhistory.tail() # 模型训练的细节

# 可视化
def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
    
plot_metric(dfhistory,"loss")
plot_metric(dfhistory,"mape")
```



##### 3.3.1.6. 评估模型

```python
# 评估
model.evaluate(dl_valid)
'''
{'val_loss': 3.619454284509023,
 'val_mae': 1.532033880551656,
 'val_mape': 0.7486907380322615}
'''
```



##### 3.3.1.7. 使用模型

```python
#这是学习到的参数
w,b = model.state_dict()["fc.weight"],model.state_dict()["fc.bias"] 

# 或者使用封装方法
dl = DataLoader(TensorDataset(X))
# 预测
model.predict(dl)[0:10]
# 预测
model.predict(dl_valid)[0:10]
```



----

#### 3.3.2. DNN二分类模型

##### 3.3.2.1. 准备数据

​	同3.1.2.1. 准备数据

---

##### 3.3.2.2. 打乱数据

```python
ds = TensorDataset(X,Y)
ds_train,ds_valid = torch.utils.data.random_split(ds,[int(len(ds)*0.7), len(ds)-int(len(ds)*0.7)])  # len(ds)
dl_train = DataLoader(ds_train,batch_size = 100, shuffle=True,num_workers=2)   # batch变为100
dl_valid = DataLoader(ds_valid,batch_size = 100, num_workers=2)
```



##### 3.3.2.3. 网络模型

```python
from torchkeras import Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,8) 
        self.fc3 = nn.Linear(8,1)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = nn.Sigmoid()(self.fc3(x))
        return y

model = torchkeras.Model(Net())  # net添加为参数
model.summary(input_shape = (2,)) # 输出模型
```



##### 3.3.2.4. 训练模型

```python
# 准确率
def accuracy(y_pred,y_true):
    y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                      torch.zeros_like(y_pred,dtype = torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc

model.compile(loss_func = nn.BCELoss(),   # loss为二元交叉熵
              optimizer= torch.optim.Adam(model.parameters(),lr = 0.01),
              metrics_dict={"accuracy":accuracy}) #评价指标有accuracy

dfhistory = model.fit(100,dl_train = dl_train, dl_val = dl_valid,log_step_freq = 10)

```



##### 3.3.2.5. 可视化训练过程

​	同3.3.1.5.可视化训练过程

---

##### 3.3.2.6. 评估模型

​	同3.3.1.6.评估模型

---

##### 3.3.2.7. 使用模型

​	同3.3.1.7.使用模型







------

## 4. Pytorch的低阶API

​	Pytorch的低阶API主要包括**张量操作，动态计算图和自动微分**。

​	在低阶API层次上，可以把Pytorch当做一个增强版的numpy来使用。Pytorch提供的方法比numpy更全面，运算速度更快，如果需要的话，还可以使用GPU进行加速。

​	前面几章我们对低阶API已经有了一个整体的认识，本章我们将重点详细介绍张量操作和动态计算图。

​	张量的操作主要包括张量的**结构操作**和张量的**数学运算**。

- 张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。
- 张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。

​	动态计算图我们将主要介绍动态计算图的特性，计算**图中的Function**，计算**图与反向传播**。



---

### 4.1. 张量的结构操作

#### 4.1.1. 创建张量

​	张量创建的许多方法和numpy中创建array的方法很像。

```python
-------------------------------------------------------------------
a = torch.tensor([1,2,3],dtype = torch.float)
tensor([1., 2., 3.])
-------------------------------------------------------------------
b = torch.arange(1,10,step = 2)
tensor([1, 3, 5, 7, 9])
-------------------------------------------------------------------
c = torch.linspace(0.0,2*3.14,10)
tensor([0.0000, 0.6978, 1.3956, 2.0933, 2.7911, 3.4889, 4.1867, 4.8844, 5.5822,
        6.2800])
-------------------------------------------------------------------
d = torch.zeros((3,3))
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
-------------------------------------------------------------------
a = torch.ones((3,3),dtype = torch.int)
tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]], dtype=torch.int32)
-------------------------------------------------------------------
b = torch.zeros_like(a,dtype = torch.float)
torch.fill_(b,5)
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
tensor([[5., 5., 5.],
        [5., 5., 5.],
        [5., 5., 5.]])
-------------------------------------------------------------------
#均匀随机分布
torch.manual_seed(0)
minval,maxval = 0,10
a = minval + (maxval-minval)*torch.rand([5])
tensor([4.9626, 7.6822, 0.8848, 1.3203, 3.0742])
-------------------------------------------------------------------
#正态分布随机
b = torch.normal(mean = torch.zeros(3,3), std = torch.ones(3,3))
tensor([[ 0.5507,  0.2704,  0.6472],
        [ 0.2490, -0.3354,  0.4564],
        [-0.6255,  0.4539, -1.3740]])
-------------------------------------------------------------------
#正态分布随机
mean,std = 2,5
c = std*torch.randn((3,3))+mean
tensor([[16.2371, -1.6612,  3.9163],
        [ 7.4999,  1.5616,  4.0768],
        [ 5.2128, -8.9407,  6.4601]])
-------------------------------------------------------------------
#整数随机排列
d = torch.randperm(20)
tensor([ 3, 17,  9, 19,  1, 18,  4, 13, 15, 12,  0, 16,  7, 11,  2,  5,  8, 10,
         6, 14])
-------------------------------------------------------------------
#特殊矩阵
I = torch.eye(3,3) #单位矩阵
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
-------------------------------------------------------------------
t = torch.diag(torch.tensor([1,2,3])) #对角矩阵
tensor([[1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]])
-------------------------------------------------------------------
```



---

#### 4.1.2. 索引切片

​	张量的索引切片方式和numpy几乎是一样的。切片时支持缺省参数和省略号。

​	可以通过索引和切片对部分元素进行修改。

​	此外，对于不规则的切片提取,可以使用`torch.index_select(), torch.masked_select(), torch.take()`

​	如果要通过修改张量的某些元素得到新的张量，可以使用`torch.where(),torch.masked_fill(),torch.index_fill()`

```python
-------------------------------------------------------------------
#均匀随机分布
torch.manual_seed(0)
minval,maxval = 0,10
t = torch.floor(minval + (maxval-minval)*torch.rand([5,5])).int()
tensor([[4, 7, 0, 1, 3],
        [6, 4, 8, 4, 6],
        [3, 4, 0, 1, 2],
        [5, 6, 8, 1, 2],
        [6, 9, 3, 8, 4]], dtype=torch.int32)
-------------------------------------------------------------------
#第1行 索引从0开始
print(t[0])
tensor([4, 7, 0, 1, 3], dtype=torch.int32)
-------------------------------------------------------------------
#最后一行
print(t[-1])
tensor([6, 9, 3, 8, 4], dtype=torch.int32)
-------------------------------------------------------------------
#第2行第4列
print(t[1,3])
print(t[1][3])
tensor(4, dtype=torch.int32)
tensor(4, dtype=torch.int32)
-------------------------------------------------------------------
#第2行至第4行   !!!!!!!!!注意只有3行1、2、3 而不是 1、2、3、4
print(t[1:4,:])
tensor([[6, 4, 8, 4, 6],
        [3, 4, 0, 1, 2],
        [5, 6, 8, 1, 2]], dtype=torch.int32)
-------------------------------------------------------------------
#第1行至最后一行，第0列到最后一列，且每隔两列取一列
print(t[1:5,:5:2])
tensor([[6, 8, 6],
        [3, 0, 2],
        [5, 8, 2],
        [6, 3, 4]], dtype=torch.int32)
-------------------------------------------------------------------
#可以使用索引和切片修改部分元素
x = torch.tensor([[1,2],[3,4]],dtype = torch.float32,requires_grad=True)
x.data[1,:] = torch.tensor([0.0,0.0])
tensor([[1., 2.],
        [3., 4.]], requires_grad=True)
tensor([[1., 2.],
        [0., 0.]], requires_grad=True)
-------------------------------------------------------------------
a = torch.arange(27).view(3,3,3)
tensor([[[ 0,  1,  2],
         [ 3,  4,  5],
         [ 6,  7,  8]],

        [[ 9, 10, 11],
         [12, 13, 14],
         [15, 16, 17]],

        [[18, 19, 20],
         [21, 22, 23],
         [24, 25, 26]]])
-------------------------------------------------------------------
#省略号可以表示多个冒号
print(a[...,1])
tensor([[ 1,  4,  7],
        [10, 13, 16],
        [19, 22, 25]])
-------------------------------------------------------------------
```

​	以上切片方式相对规则，对于不规则的切片提取,可以使用`torch.index_select, torch.take, torch.gather, torch.masked_select`.

​	考虑班级成绩册的例子，有4个班级，每个班级10个学生，每个学生7门科目成绩。可以用一个4×10×7的张量来表示。

```python
-------------------------------------------------------------------
minval=0
maxval=100
scores = torch.floor(minval + (maxval-minval)*torch.rand([4,10,7])).int()
-------------------------------------------------------------------
tensor([[[55, 95,  3, 18, 37, 30, 93],
         [17, 26, 15,  3, 20, 92, 72],
         [74, 52, 24, 58,  3, 13, 24],
         [81, 79, 27, 48, 81, 99, 69],
         [56, 83, 20, 59, 11, 15, 24],
         [72, 70, 20, 65, 77, 43, 51],
         [61, 81, 98, 11, 31, 69, 91],
         [93, 94, 59,  6, 54, 18,  3],
         [94, 88,  0, 59, 41, 41, 27],
         [69, 20, 68, 75, 85, 68,  0]],

        [[17, 74, 60, 10, 21, 97, 83],
         [28, 37,  2, 49, 12, 11, 47],
         [57, 29, 79, 19, 95, 84,  7],
         [37, 52, 57, 61, 69, 52, 25],
         [73,  2, 20, 37, 25, 32,  9],
         [39, 60, 17, 47, 85, 44, 51],
         [45, 60, 81, 97, 81, 97, 46],
         [ 5, 26, 84, 49, 25, 11,  3],
         [ 7, 39, 77, 77,  1, 81, 10],
         [39, 29, 40, 40,  5,  6, 42]],

        [[50, 27, 68,  4, 46, 93, 29],
         [95, 68,  4, 81, 44, 27, 89],
         [ 9, 55, 39, 85, 63, 74, 67],
         [37, 39,  8, 77, 89, 84, 14],
         [52, 14, 22, 20, 67, 20, 48],
         [52, 82, 12, 15, 20, 84, 32],
         [92, 68, 56, 49, 40, 56, 38],
         [49, 56, 10, 23, 90,  9, 46],
         [99, 68, 51,  6, 74, 14, 35],
         [33, 42, 50, 91, 56, 94, 80]],

        [[18, 72, 14, 28, 64, 66, 87],
         [33, 50, 75,  1, 86,  8, 50],
         [41, 23, 56, 91, 35, 20, 31],
         [ 0, 72, 25, 16, 21, 78, 76],
         [88, 68, 33, 36, 64, 91, 63],
         [26, 26,  2, 60, 21,  5, 93],
         [17, 44, 64, 51, 16,  9, 89],
         [58, 91, 33, 64, 38, 47, 19],
         [66, 65, 48, 38, 19, 84, 12],
         [70, 33, 25, 58, 24, 61, 59]]], dtype=torch.int32)
-------------------------------------------------------------------
#抽取每个班级第1个学生，第6个学生，第10个学生的全部成绩
torch.index_select(scores,dim = 1,index = torch.tensor([0,5,9]))
tensor([[[55, 95,  3, 18, 37, 30, 93],
         [72, 70, 20, 65, 77, 43, 51],
         [69, 20, 68, 75, 85, 68,  0]],

        [[17, 74, 60, 10, 21, 97, 83],
         [39, 60, 17, 47, 85, 44, 51],
         [39, 29, 40, 40,  5,  6, 42]],

        [[50, 27, 68,  4, 46, 93, 29],
         [52, 82, 12, 15, 20, 84, 32],
         [33, 42, 50, 91, 56, 94, 80]],

        [[18, 72, 14, 28, 64, 66, 87],
         [26, 26,  2, 60, 21,  5, 93],
         [70, 33, 25, 58, 24, 61, 59]]], dtype=torch.int32)
-------------------------------------------------------------------
#抽取每个班级第1个学生，第6个学生，第10个学生的第2门课程，第4门课程，第7门课程成绩   
# 嵌套，先找出来满足第一个条件的，再找满足第二个条件的
q = torch.index_select(torch.index_select(scores,dim = 1,index = torch.tensor([0,5,9]))
                   ,dim=2,index = torch.tensor([1,3,6]))
tensor([[[95, 18, 93],
         [70, 65, 51],
         [20, 75,  0]],

        [[74, 10, 83],
         [60, 47, 51],
         [29, 40, 42]],

        [[27,  4, 29],
         [82, 15, 32],
         [42, 91, 80]],

        [[72, 28, 87],
         [26, 60, 93],
         [33, 58, 59]]], dtype=torch.int32)
-------------------------------------------------------------------
#抽取第1个班级第1个学生的第1门课程，第3个班级的第5个学生的第2门课程，第4个班级的第10个学生第7门课程成绩
#take将输入看成一维数组，输出和index同形状  
#按照索引值取元素
s = torch.take(scores,torch.tensor([0*10*7+0,2*10*7+4*7+1,3*10*7+9*7+6]))
tensor([55, 14, 59], dtype=torch.int32)
-------------------------------------------------------------------
#抽取分数大于等于80分的分数（布尔索引）
#结果是1维张量
g = torch.masked_select(scores,scores>=80)
tensor([95, 93, 92, 81, 81, 99, 83, 81, 98, 91, 93, 94, 94, 88, 85, 97, 83, 95,
        84, 85, 81, 97, 81, 97, 84, 81, 93, 95, 81, 89, 85, 89, 84, 82, 84, 92,
        90, 99, 91, 94, 80, 87, 86, 91, 88, 91, 93, 89, 91, 84],
       dtype=torch.int32)
-------------------------------------------------------------------
```

​	以上这些方法仅能提取张量的部分元素值，但不能更改张量的部分元素值得到新的张量。

​	如果要通过修改张量的部分元素值得到新的张量，可以使用`torch.where、 torch.index_fill、torch.masked_fill`

​	`torch.where`可以理解为if的张量版本。

​	`torch.index_fill`的选取元素逻辑和`torch.index_select`相同。

​	`torch.masked_fill`的选取元素逻辑和`torch.masked_select`相同。

```python
-------------------------------------------------------------------
ifpass = torch.where(scores>60,torch.tensor(1),torch.tensor(0))
tensor([[[0, 1, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 1, 1],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 1, 1, 1],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 1, 0, 1, 1, 0, 0],
         [1, 1, 1, 0, 0, 1, 1],
         [1, 1, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0],
         [1, 0, 1, 1, 1, 1, 0]],

        [[0, 1, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 1, 0],
         [0, 0, 0, 1, 1, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0]],

        [[0, 0, 1, 0, 0, 1, 0],
         [1, 1, 0, 1, 0, 0, 1],
         [0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [1, 1, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 1, 1]],

        [[0, 1, 0, 0, 1, 1, 1],
         [0, 0, 1, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 1, 1],
         [1, 1, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0, 1],
         [0, 1, 0, 1, 0, 0, 0],
         [1, 1, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 1, 0]]])
-------------------------------------------------------------------
#将每个班级第1个学生，第6个学生，第10个学生的全部成绩赋值成满分
torch.index_fill(scores,dim = 1,index = torch.tensor([0,5,9]),value = 100)
#等价于 scores.index_fill(dim = 1,index = torch.tensor([0,5,9]),value = 100)
ensor([[[100, 100, 100, 100, 100, 100, 100],
         [ 17,  26,  15,   3,  20,  92,  72],
         [ 74,  52,  24,  58,   3,  13,  24],
         [ 81,  79,  27,  48,  81,  99,  69],
         [ 56,  83,  20,  59,  11,  15,  24],
         [100, 100, 100, 100, 100, 100, 100],
         [ 61,  81,  98,  11,  31,  69,  91],
         [ 93,  94,  59,   6,  54,  18,   3],
         [ 94,  88,   0,  59,  41,  41,  27],
         [100, 100, 100, 100, 100, 100, 100]],

        [[100, 100, 100, 100, 100, 100, 100],
         [ 28,  37,   2,  49,  12,  11,  47],
         [ 57,  29,  79,  19,  95,  84,   7],
         [ 37,  52,  57,  61,  69,  52,  25],
         [ 73,   2,  20,  37,  25,  32,   9],
         [100, 100, 100, 100, 100, 100, 100],
         [ 45,  60,  81,  97,  81,  97,  46],
         [  5,  26,  84,  49,  25,  11,   3],
         [  7,  39,  77,  77,   1,  81,  10],
         [100, 100, 100, 100, 100, 100, 100]],

        [[100, 100, 100, 100, 100, 100, 100],
         [ 95,  68,   4,  81,  44,  27,  89],
         [  9,  55,  39,  85,  63,  74,  67],
         [ 37,  39,   8,  77,  89,  84,  14],
         [ 52,  14,  22,  20,  67,  20,  48],
         [100, 100, 100, 100, 100, 100, 100],
         [ 92,  68,  56,  49,  40,  56,  38],
         [ 49,  56,  10,  23,  90,   9,  46],
         [ 99,  68,  51,   6,  74,  14,  35],
         [100, 100, 100, 100, 100, 100, 100]],

        [[100, 100, 100, 100, 100, 100, 100],
         [ 33,  50,  75,   1,  86,   8,  50],
         [ 41,  23,  56,  91,  35,  20,  31],
         [  0,  72,  25,  16,  21,  78,  76],
         [ 88,  68,  33,  36,  64,  91,  63],
         [100, 100, 100, 100, 100, 100, 100],
         [ 17,  44,  64,  51,  16,   9,  89],
         [ 58,  91,  33,  64,  38,  47,  19],
         [ 66,  65,  48,  38,  19,  84,  12],
         [100, 100, 100, 100, 100, 100, 100]]], dtype=torch.int32)
-------------------------------------------------------------------
#将分数小于60分的分数赋值成60分
b = torch.masked_fill(scores,scores<60,60)
#等价于b = scores.masked_fill(scores<60,60)
tensor([[[60, 95, 60, 60, 60, 60, 93],
         [60, 60, 60, 60, 60, 92, 72],
         [74, 60, 60, 60, 60, 60, 60],
         [81, 79, 60, 60, 81, 99, 69],
         [60, 83, 60, 60, 60, 60, 60],
         [72, 70, 60, 65, 77, 60, 60],
         [61, 81, 98, 60, 60, 69, 91],
         [93, 94, 60, 60, 60, 60, 60],
         [94, 88, 60, 60, 60, 60, 60],
         [69, 60, 68, 75, 85, 68, 60]],

        [[60, 74, 60, 60, 60, 97, 83],
         [60, 60, 60, 60, 60, 60, 60],
         [60, 60, 79, 60, 95, 84, 60],
         [60, 60, 60, 61, 69, 60, 60],
         [73, 60, 60, 60, 60, 60, 60],
         [60, 60, 60, 60, 85, 60, 60],
         [60, 60, 81, 97, 81, 97, 60],
         [60, 60, 84, 60, 60, 60, 60],
         [60, 60, 77, 77, 60, 81, 60],
         [60, 60, 60, 60, 60, 60, 60]],

        [[60, 60, 68, 60, 60, 93, 60],
         [95, 68, 60, 81, 60, 60, 89],
         [60, 60, 60, 85, 63, 74, 67],
         [60, 60, 60, 77, 89, 84, 60],
         [60, 60, 60, 60, 67, 60, 60],
         [60, 82, 60, 60, 60, 84, 60],
         [92, 68, 60, 60, 60, 60, 60],
         [60, 60, 60, 60, 90, 60, 60],
         [99, 68, 60, 60, 74, 60, 60],
         [60, 60, 60, 91, 60, 94, 80]],

        [[60, 72, 60, 60, 64, 66, 87],
         [60, 60, 75, 60, 86, 60, 60],
         [60, 60, 60, 91, 60, 60, 60],
         [60, 72, 60, 60, 60, 78, 76],
         [88, 68, 60, 60, 64, 91, 63],
         [60, 60, 60, 60, 60, 60, 93],
         [60, 60, 64, 60, 60, 60, 89],
         [60, 91, 60, 64, 60, 60, 60],
         [66, 65, 60, 60, 60, 84, 60],
         [70, 60, 60, 60, 60, 61, 60]]], dtype=torch.int32)
-------------------------------------------------------------------
```



---

#### 4.1.3. 维度变换

​	维度变换相关函数主要有 `torch.reshape(或者调用张量的view方法), torch.squeeze, torch.unsqueeze, torch.transpose`

​	`torch.reshape` 可以改变张量的形状。

​	`torch.squeeze` 可以减少维度。	

​	`torch.unsqueeze` 可以增加维度。

​	`torch.transpose` 可以交换维度。

```python
-------------------------------------------------------------------# 张量的view方法有时候会调用失败，可以使用reshape方法。
torch.manual_seed(0)
minval,maxval = 0,255
a = (minval + (maxval-minval)*torch.rand([1,3,3,2])).int()
print(a.shape)
print(a)

torch.Size([1, 3, 3, 2])
tensor([[[[126, 195],
          [ 22,  33],
          [ 78, 161]],

         [[124, 228],
          [116, 161],
          [ 88, 102]],

         [[  5,  43],
          [ 74, 132],
          [177, 204]]]], dtype=torch.int32)
-------------------------------------------------------------------
# 改成 （3,6）形状的张量
b = a.view([3,6]) #torch.reshape(a,[3,6])
print(b.shape)
print(b)

torch.Size([3, 6])
tensor([[126, 195,  22,  33,  78, 161],
        [124, 228, 116, 161,  88, 102],
        [  5,  43,  74, 132, 177, 204]], dtype=torch.int32)
-------------------------------------------------------------------
# 改回成 [1,3,3,2] 形状的张量
c = torch.reshape(b,[1,3,3,2]) # b.view([1,3,3,2]) 
tensor([[[[126, 195],
          [ 22,  33],
          [ 78, 161]],

         [[124, 228],
          [116, 161],
          [ 88, 102]],

         [[  5,  43],
          [ 74, 132],
          [177, 204]]]], dtype=torch.int32)
-------------------------------------------------------------------
```

​	如果张量在某个维度上只有一个元素，利用 `torch.squeeze` 可以消除这个维度。

​	`torch.unsqueeze` 的作用和 `torch.squeeze` 的作用相反。

```python
-------------------------------------------------------------------
a = torch.tensor([[1.0,2.0]])
s = torch.squeeze(a)
print(a);print(a.shape)
print(s);print(s.shape)

tensor([[1., 2.]])   torch.Size([1, 2])
tensor([1., 2.])     torch.Size([2])
-------------------------------------------------------------------
#在第0维插入长度为1的一个维度
d = torch.unsqueeze(s, axis=0)  
print(s);print(s.shape)
print(d);print(d.shape)
tensor([1., 2.])     torch.Size([2])
tensor([[1., 2.]])   torch.Size([1, 2])
-------------------------------------------------------------------
```

​	`torch.transpose`可以交换张量的维度，`torch.transpose`常用于图片存储格式的变换上。

​	如果是二维的矩阵，通常会调用矩阵的转置方法 `matrix.t()`，等价于 `torch.transpose(matrix,0,1)`。

```python
-------------------------------------------------------------------minval=0
maxval=255
# Batch,Height,Width,Channel
data = torch.floor(minval + (maxval-minval)*torch.rand([100,256,256,4])).int()
print(data.shape)

# 转换成 Pytorch默认的图片格式 Batch,Channel,Height,Width 
# 需要交换两次
data_t = torch.transpose(torch.transpose(data,1,2),1,3)
print(data_t.shape)

torch.Size([100, 256, 256, 4])
torch.Size([100, 4, 256, 256])
-------------------------------------------------------------------
matrix = torch.tensor([[1,2,3],[4,5,6]])
print(matrix)
print(matrix.t()) #等价于torch.transpose(matrix,0,1)

tensor([[1, 2, 3],
        [4, 5, 6]])
tensor([[1, 4],
        [2, 5],
        [3, 6]])
-------------------------------------------------------------------
```



---

#### 4.1.4. 合并分割

​	可以用 `torch.cat` 方法和 `torch.stack` 方法将多个张量合并，可以用`torch.split` 方法把一个张量分割成多个张量。

​	`torch.cat` 和 `torch.stack` 有略微的区别，`torch.cat` 是连接，不会增加维度，而 `torch.stack` 是堆叠，会增加维度。

```python
-------------------------------------------------------------------
a = torch.tensor([[1.0,2.0],[3.0,4.0]])
b = torch.tensor([[5.0,6.0],[7.0,8.0]])
c = torch.tensor([[9.0,10.0],[11.0,12.0]])
abc_cat = torch.cat([a,b,c],dim = 0)
print(abc_cat.shape)
print(abc_cat)

torch.Size([6, 2])
tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.]])
-------------------------------------------------------------------
abc_stack = torch.stack([a,b,c],axis = 0) #torch中dim和axis参数名可以混用
print(abc_stack.shape)
print(abc_stack)

torch.Size([3, 2, 2])
tensor([[[ 1.,  2.],
         [ 3.,  4.]],

        [[ 5.,  6.],
         [ 7.,  8.]],

        [[ 9., 10.],
         [11., 12.]]])
-------------------------------------------------------------------
torch.cat([a,b,c],axis = 1)
tensor([[ 1.,  2.,  5.,  6.,  9., 10.],
        [ 3.,  4.,  7.,  8., 11., 12.]])
-------------------------------------------------------------------
torch.stack([a,b,c],axis = 1)
tensor([[[ 1.,  2.],
         [ 5.,  6.],
         [ 9., 10.]],

        [[ 3.,  4.],
         [ 7.,  8.],
         [11., 12.]]])
-------------------------------------------------------------------
```

​	`torch.split` 是 `torch.cat` 的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割。

```python
-------------------------------------------------------------------
print(abc_cat)
a,b,c = torch.split(abc_cat,split_size_or_sections = 2,dim = 0) #每份2个进行分割
print(a)
print(b)
print(c)

tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.]])
tensor([[1., 2.],
        [3., 4.]])
tensor([[5., 6.],
        [7., 8.]])
tensor([[ 9., 10.],
        [11., 12.]])
-------------------------------------------------------------------
print(abc_cat)
p,q,r = torch.split(abc_cat,split_size_or_sections =[4,1,1],dim = 0) #每份分别为[4,1,1]
print(p)
print(q)
print(r)

tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.]])
tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])
tensor([[ 9., 10.]])
tensor([[11., 12.]])
-------------------------------------------------------------------
```





---

### 4.2. 张量的数学计算

​	张量的操作主要包括张量的**结构操作**和张量的**数学运算**。

​	张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。

​	张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。

​	本篇我们介绍张量的数学运算。



---

#### 4.2.1. 标量运算

​	张量的数学运算符可以分为标量运算符、向量运算符、以及矩阵运算符。

​	加减乘除乘方，以及三角函数，指数，对数等常见函数，逻辑比较运算符等都是标量运算符。

​	标量运算符的特点是对张量实施逐元素运算。

​	有些标量运算符对常用的数学运算符进行了**重载**。并且支持类似numpy的广播特性。

```python
-------------------------------------------------------------------a = torch.tensor([[1.0,2],[-3,4.0]])
b = torch.tensor([[5.0,6],[7.0,8.0]])
a+b  #运算符重载

tensor([[ 6.,  8.],
        [ 4., 12.]])
-------------------------------------------------------------------
a-b 
tensor([[ -4.,  -4.],
        [-10.,  -4.]])
-------------------------------------------------------------------
a*b 
tensor([[  5.,  12.],
        [-21.,  32.]])
-------------------------------------------------------------------
a/b
tensor([[ 0.2000,  0.3333],
        [-0.4286,  0.5000]])
-------------------------------------------------------------------
a**2
tensor([[ 1.,  4.],
        [ 9., 16.]])
-------------------------------------------------------------------
a**(0.5) # 同torch.sqrt(a)
tensor([[1.0000, 1.4142],
        [   nan, 2.0000]])
-------------------------------------------------------------------
a%3 #求模
tensor([[1., 2.],
        [0., 1.]])
-------------------------------------------------------------------
a//3  #地板除法floor division 即商  两个整数的除法仍然是整数 
tensor([[ 0.,  0.],
        [-1.,  1.]])

-------------------------------------------------------------------
a>=2 # torch.ge(a,2)  #ge: greater_equal缩写
tensor([[False,  True],
        [False,  True]])
-------------------------------------------------------------------
(a>=2)&(a<=3)
tensor([[False,  True],
        [False, False]])
-------------------------------------------------------------------
(a>=2)|(a<=3)
tensor([[True, True],
        [True, True]])
-------------------------------------------------------------------
a==5 #torch.eq(a,5)
tensor([[False, False],
        [False, False]])
-------------------------------------------------------------------
```



```python
-------------------------------------------------------------------
a = torch.tensor([1.0,8.0])
b = torch.tensor([5.0,6.0])
c = torch.tensor([6.0,7.0])
d = a+b+c
print(d)

tensor([12., 21.])
-------------------------------------------------------------------
print(torch.max(a,b))
tensor([5., 8.])
-------------------------------------------------------------------
print(torch.min(a,b))
tensor([1., 6.])
-------------------------------------------------------------------
x = torch.tensor([2.6,-2.7])
print(torch.round(x)) #保留整数部分，四舍五入
print(torch.floor(x)) #保留整数部分，向下归整
print(torch.ceil(x))  #保留整数部分，向上归整
print(torch.trunc(x)) #保留整数部分，向0归整

tensor([ 3., -3.])
tensor([ 2., -3.])
tensor([ 3., -2.])
tensor([ 2., -2.])
-------------------------------------------------------------------
x = torch.tensor([2.6,-2.7])
print(torch.fmod(x,2)) #作除法取余数 
print(torch.remainder(x,2)) #作除法取剩余的部分，结果恒正

tensor([ 0.6000, -0.7000])
tensor([0.6000, 1.3000])
-------------------------------------------------------------------
# 幅值裁剪 收缩范围
x = torch.tensor([0.9,-0.8,100.0,-20.0,0.7])
y = torch.clamp(x,min=-1,max = 1)
z = torch.clamp(x,max = 1)
print(y)
print(z)

tensor([ 0.9000, -0.8000,  1.0000, -1.0000,  0.7000])
tensor([  0.9000,  -0.8000,   1.0000, -20.0000,   0.7000])
-------------------------------------------------------------------
```



---

#### 4.2.2. 向量运算

​	向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另外一个向量。

```python
-------------------------------------------------------------------
#统计值
a = torch.arange(1,10).float() # 不包括10、且dtype为int64
print(torch.sum(a))
print(torch.mean(a))
print(torch.max(a))
print(torch.min(a))
print(torch.prod(a)) #累乘
print(torch.std(a))  #标准差
print(torch.var(a))  #方差
print(torch.median(a)) #中位数

tensor(45.)
tensor(5.)
tensor(9.)
tensor(1.)
tensor(362880.)
tensor(2.7386)
tensor(7.5000)
tensor(5.)
-------------------------------------------------------------------
#指定维度计算统计值
b = a.view(3,3)
print(b)
print(torch.max(b,dim = 0))
print(torch.max(b,dim = 1))

tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]])
torch.return_types.max(
values=tensor([7., 8., 9.]),
indices=tensor([2, 2, 2]))
torch.return_types.max(
values=tensor([3., 6., 9.]),
indices=tensor([2, 2, 2]))
-------------------------------------------------------------------
#cum扫描
a = torch.arange(1,10)
print(torch.cumsum(a,0)) # 累加
print(torch.cumprod(a,0))# 累乘

tensor([ 1,  3,  6, 10, 15, 21, 28, 36, 45])
tensor([     1,      2,      6,     24,    120,    720,   5040,  40320, 362880])
-------------------------------------------------------------------
#torch.sort和torch.topk可以对张量排序
a = torch.tensor([[9,7,8],[1,3,2],[5,6,4]]).float()
print(torch.topk(a,2,dim = 0),"\n") #找出每一列中最大的前2个值
print(torch.topk(a,2,dim = 1),"\n") #找出每一行中最大的前2个值
print(torch.sort(a,dim = 1),"\n")   #在行上排序
#利用torch.topk可以在Pytorch中实现KNN算法

torch.return_types.topk(
values=tensor([[9., 7., 8.],
        [5., 6., 4.]]),
indices=tensor([[0, 0, 0],
        [2, 2, 2]])) 

torch.return_types.topk(
values=tensor([[9., 8.],
        [3., 2.],
        [6., 5.]]),
indices=tensor([[0, 2],
        [1, 2],
        [1, 0]])) 

torch.return_types.sort(
values=tensor([[7., 8., 9.],
        [1., 2., 3.],
        [4., 5., 6.]]),
indices=tensor([[1, 2, 0],
        [0, 2, 1],
        [2, 0, 1]])) 
-------------------------------------------------------------------
```



---

#### 4.2.3. 矩阵运算

​	矩阵必须是二维的。类似`torch.tensor([1,2,3])`这样的不是矩阵。

​	矩阵运算包括：矩阵乘法，矩阵转置，矩阵逆，矩阵求迹，矩阵范数，矩阵行列式，矩阵求特征值，矩阵分解等运算。

```python
-------------------------------------------------------------------
#矩阵乘法
a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[2,0],[0,2]])
print(a@b)  #等价于torch.matmul(a,b) 或 torch.mm(a,b)

tensor([[2, 4],
        [6, 8]])
-------------------------------------------------------------------
#矩阵转置
a = torch.tensor([[1.0,2],[3,4]])
print(a.t())

tensor([[1., 3.],
        [2., 4.]])
-------------------------------------------------------------------
#矩阵逆，必须为浮点类型
a = torch.tensor([[1.0,2],[3,4]])
print(torch.inverse(a))

tensor([[-2.0000,  1.0000],
        [ 1.5000, -0.5000]])
-------------------------------------------------------------------
#矩阵求trace
a = torch.tensor([[1.0,2],[3,4]])
print(torch.trace(a))

tensor(5.)
-------------------------------------------------------------------
#矩阵求范数
a = torch.tensor([[1.0,2],[3,4]])
print(torch.norm(a)) # 默认2范数

tensor(5.4772)
-------------------------------------------------------------------
#矩阵行列式
a = torch.tensor([[1.0,2],[3,4]])
print(torch.det(a))

tensor(-2.0000)
-------------------------------------------------------------------
#矩阵特征值和特征向量
a = torch.tensor([[1.0,2],[-5,4]],dtype = torch.float)
print(torch.eig(a,eigenvectors=True))
#两个特征值分别是 -2.5+2.7839j, 2.5-2.7839j 

torch.return_types.eig(
eigenvalues=tensor([[ 2.5000,  2.7839],
        [ 2.5000, -2.7839]]),
eigenvectors=tensor([[ 0.2535, -0.4706],
        [ 0.8452,  0.0000]]))
-------------------------------------------------------------------
#矩阵QR分解, 将一个方阵分解为一个正交矩阵q和上三角矩阵r
#QR分解实际上是对矩阵a实施Schmidt正交化得到q
a  = torch.tensor([[1.0,2.0],[3.0,4.0]])
q,r = torch.qr(a)
print(q,"\n")
print(r,"\n")
print(q@r)

tensor([[-0.3162, -0.9487],
        [-0.9487,  0.3162]]) 

tensor([[-3.1623, -4.4272],
        [ 0.0000, -0.6325]]) 

tensor([[1.0000, 2.0000],
        [3.0000, 4.0000]])
-------------------------------------------------------------------
#矩阵svd分解
#svd分解可以将任意一个矩阵分解为一个正交矩阵u,一个对角阵s和一个正交矩阵v.t()的乘积
#svd常用于矩阵压缩和降维
a=torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]])
u,s,v = torch.svd(a)
print(u,"\n")
print(s,"\n")
print(v,"\n")
print(u@torch.diag(s)@v.t())
#利用svd分解可以在Pytorch中实现主成分分析降维

tensor([[-0.2298,  0.8835],
        [-0.5247,  0.2408],
        [-0.8196, -0.4019]]) 

tensor([9.5255, 0.5143]) 

tensor([[-0.6196, -0.7849],
        [-0.7849,  0.6196]]) 

tensor([[1.0000, 2.0000],
        [3.0000, 4.0000],
        [5.0000, 6.0000]])
```



---

#### 4.2.4. 广播机制

​    Pytorch的广播规则和numpy是一样的:

- 1、如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样。
- 2、如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为1，那么我们就说这两个张量在该维度上是相容的。
- 3、如果两个张量在所有维度上都是相容的，它们就能使用广播。
- 4、广播之后，每个维度的长度将取两个张量在该维度长度的较大值。
- 5、在任何一个维度上，如果一个张量的长度为1，另一个张量长度大于1，那么在该维度上，就好像是对第一个张量进行了复制。

​    `torch.broadcast_tensors`可以将多个张量根据广播规则转换成相同的维度。

​	在进行==按位运算==的时候，广播机制broadcasting可以帮助减少代码量。

```python
-------------------------------------------------------------------
a = torch.tensor([1,2,3])
b = torch.tensor([[0,0,0],[1,1,1],[2,2,2]])
print(b + a) 

tensor([[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]])
-------------------------------------------------------------------
a_broad,b_broad = torch.broadcast_tensors(a,b)
print(a_broad,"\n")
print(b_broad,"\n")
print(a_broad + b_broad) 

tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]]) 

tensor([[0, 0, 0],
        [1, 1, 1],
        [2, 2, 2]]) 

tensor([[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]])
-------------------------------------------------------------------
```





---

### 4.3. nn.function & nn.Module

#### 4.3.1. nn.functional 和 nn.Module

​	前面我们介绍了Pytorch的张量的结构操作和数学运算中的一些常用API。

​	利用这些张量的API我们可以构建出神经网络相关的组件(如激活函数，模型层，损失函数)。

​	Pytorch和神经网络相关的功能组件大多都封装在 torch.nn模块下。

​	这些功能组件的绝大部分既有函数形式实现，也有类形式实现。

##### nn.functional

​	其中`nn.functional`(一般引入后改名为`F`)有各种功能组件的函数实现。例如：

```python
# (激活函数)
F.relu
F.sigmoid
F.tanh
F.softmax
# (模型层)
F.linear
F.conv2d
F.max_pool2d
F.dropout2d
F.embedding
# (损失函数)
F.binary_cross_entropy
F.mse_loss
F.cross_entropy
```

##### nn.Module

​	为了便于对参数进行管理，一般通过继承 `nn.Module `转换成为类的实现形式，并直接封装在 `nn` 模块下。例如：

```python
# (激活函数)
nn.ReLU
nn.Sigmoid
nn.Tanh
nn.Softmax
# (模型层)
nn.Linear
nn.Conv2d
nn.MaxPool2d
nn.Dropout2d
nn.Embedding
# (损失函数)
nn.BCELoss
nn.MSELoss
nn.CrossEntropyLoss
```

​	实际上`nn.Module`除了可以管理其引用的各种参数，还可以管理其引用的子模块，功能十分强大。



---

#### 4.3.2. 使用nn.Module来管理参数

​	在Pytorch中，模型的参数是需要被优化器训练的，因此，通常要设置参数为 `requires_grad = True` 的张量。

​	同时，在一个模型中，往往有许多的参数，要手动管理这些参数并不是一件容易的事情。

​	Pytorch一般将参数用 `nn.Parameter` 来表示，并且用 `nn.Module` 来管理其结构下的所有参数。

```python
-------------------------------------------------------------------
# nn.Parameter 具有 requires_grad = True 属性
w = nn.Parameter(torch.randn(2,2))
print(w)
print(w.requires_grad)

Parameter containing:
tensor([[ 1.2790,  0.6851],
        [-1.9961,  0.4121]], requires_grad=True)
True
-------------------------------------------------------------------
# nn.ParameterList 可以将多个nn.Parameter组成一个列表
params_list = nn.ParameterList([nn.Parameter(torch.rand(8,i)) for i in range(1,3)])
print(params_list)
print(params_list[0].requires_grad)

ParameterList(
    (0): Parameter containing: [torch.FloatTensor of size 8x1]
    (1): Parameter containing: [torch.FloatTensor of size 8x2]
)
True
-------------------------------------------------------------------
# nn.ParameterDict 可以将多个nn.Parameter组成一个字典
params_dict = nn.ParameterDict({"a":nn.Parameter(torch.rand(2,2)),
                               "b":nn.Parameter(torch.zeros(2))})
print(params_dict)
print(params_dict["a"].requires_grad)

ParameterDict(
    (a): Parameter containing: [torch.FloatTensor of size 2x2]
    (b): Parameter containing: [torch.FloatTensor of size 2]
)
True
-------------------------------------------------------------------
# 可以用Module将它们管理起来
# module.parameters()返回一个生成器，包括其结构下的所有parameters
module = nn.Module()
module.w = w
module.params_list = params_list
module.params_dict = params_dict

num_param = 0
for param in module.parameters():
    print(param,"\n")
    num_param = num_param + 1
print("number of Parameters =",num_param)

'''
	output
'''
Parameter containing:
tensor([[ 1.2790,  0.6851],
        [-1.9961,  0.4121]], requires_grad=True) 

Parameter containing:
tensor([[0.5894],
        [0.4028],
        [0.0762],
        [0.0865],
        [0.4465],
        [0.5436],
        [0.0937],
        [0.8081]], requires_grad=True) 

Parameter containing:
tensor([[0.3550, 0.6751],
        [0.8561, 0.4741],
        [0.6387, 0.8738],
        [0.2147, 0.8263],
        [0.5310, 0.1348],
        [0.9466, 0.0573],
        [0.7939, 0.8348],
        [0.7429, 0.7065]], requires_grad=True) 

Parameter containing:
tensor([[0.5320, 0.0495],
        [0.1472, 0.7734]], requires_grad=True) 

Parameter containing:
tensor([0., 0.], requires_grad=True) 

number of Parameters = 5
```

​	实践当中，一般通过继承 `nn.Module` 来构建模块类，并将所有含有需要学习的参数的部分放在构造函数中。

```python
#以下范例为Pytorch中nn.Linear的源码的简化版本
#可以看到它将需要学习的参数放在了__init__构造函数中，并在forward中调用F.linear函数来实现计算逻辑。

class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
-------------------------------------------------------------------
```



---

#### 4.3.3. 使用nn.Module来管理子模块

​	一般情况下，我们都很少直接使用 `nn.Parameter` 来定义参数构建模型，而是通过一些拼装一些常用的模型层来构造模型。

​	这些模型层也是继承自 `nn.Module` 的对象,本身也包括参数，属于我们要定义的模块的子模块。

​	`nn.Module` 提供了一些方法可以管理这些子模块。

- `children() ` 方法: 返回生成器，包括模块下的所有子模块。
- `named_children()` 方法：返回一个生成器，包括模块下的所有子模块，以及它们的名字。
- `modules()` 方法：返回一个生成器，包括模块下的所有各个层级的模块，包括模块本身。
- `named_modules()` 方法：返回一个生成器，包括模块下的所有各个层级的模块以及它们的名字，包括模块本身。

​    其中 `chidren()` 方法和 `named_children()` 方法较多使用。

​    `modules()` 方法和 `named_modules()` 方法较少使用，其功能可以通过多个 `named_children()` 的嵌套使用实现。

```python
-------------------------------------------------------------------
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings = 10000,embedding_dim = 3,padding_idx = 1)
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1",nn.Conv1d(in_channels = 3,out_channels = 16,kernel_size = 5))
        self.conv.add_module("pool_1",nn.MaxPool1d(kernel_size = 2))
        self.conv.add_module("relu_1",nn.ReLU())
        self.conv.add_module("conv_2",nn.Conv1d(in_channels = 16,out_channels = 128,kernel_size = 2))
        self.conv.add_module("pool_2",nn.MaxPool1d(kernel_size = 2))
        self.conv.add_module("relu_2",nn.ReLU())
        
        self.dense = nn.Sequential()
        self.dense.add_module("flatten",nn.Flatten())
        self.dense.add_module("linear",nn.Linear(6144,1))
        self.dense.add_module("sigmoid",nn.Sigmoid())
        
    def forward(self,x):
        x = self.embedding(x).transpose(1,2)
        x = self.conv(x)
        y = self.dense(x)
        return y
    
net = Net()
-------------------------------------------------------------------
i = 0
for child in net.children():
    i+=1
    print(child,"\n")
print("child number",i)

'''
	OUTPUT
'''
Embedding(10000, 3, padding_idx=1) 

Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()
) 

Sequential(
  (flatten): Flatten()
  (linear): Linear(in_features=6144, out_features=1, bias=True)
  (sigmoid): Sigmoid()
) 

child number 3
-------------------------------------------------------------------
i = 0
for module in net.modules():
    i+=1
    print(module)
print("module number:",i)

Net(
  (embedding): Embedding(10000, 3, padding_idx=1)
  (conv): Sequential(
    (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
    (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (relu_1): ReLU()
    (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
    (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (relu_2): ReLU()
  )
  (dense): Sequential(
    (flatten): Flatten()
    (linear): Linear(in_features=6144, out_features=1, bias=True)
    (sigmoid): Sigmoid()
  )
)
Embedding(10000, 3, padding_idx=1)
Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()
)
Conv1d(3, 16, kernel_size=(5,), stride=(1,))
MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
ReLU()
Conv1d(16, 128, kernel_size=(2,), stride=(1,))
MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
ReLU()
Sequential(
  (flatten): Flatten()
  (linear): Linear(in_features=6144, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
Flatten()
Linear(in_features=6144, out_features=1, bias=True)
Sigmoid()
module number: 13
-------------------------------------------------------------------
```

​	下面我们通过 `named_children` 方法找到 `embedding` 层，并将其参数设置为不可训练(相当于冻结 `embedding`层)。

```python
children_dict = {name:module for name,module in net.named_children()}
print(children_dict)
embedding = children_dict["embedding"]
embedding.requires_grad_(False) #冻结其参数
for param in embedding.parameters():
    print(param.requires_grad)
    print(param.numel())
    
'''
	OUTPUT
'''
#可以看到其第一层的参数已经不可以被训练了。
{'embedding': Embedding(10000, 3, padding_idx=1), 
 'conv': Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()), 
 'dense': Sequential(
  (flatten): Flatten()
  (linear): Linear(in_features=6144, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)}
False
30000
```





------

## 5. Pytorch的中阶API

我们将主要介绍Pytorch的如下中阶API

- 数据管道
- 模型层
- 损失函数
- TensorBoard可视化

---

### 5.1. Dataset & DataLoader

​	`Pytorch` 通常使用 `Dataset` 和 `DataLoader` 这两个工具类来构建**数据管道**。

​	`Dataset` 定义了数据集的内容，它相当于一个类似列表的**数据结构**，具有确定的长度，能够用索引获取数据集中的元素。

​	而 `DataLoader` 定义了按 `batch` 加载数据集的方法，它是一个实现了`__iter__`方法的可迭代对象，每次迭代输出一个 `batch`的数据。

​	`DataLoader` 能够控制 `batch` 的大小，`batch` 中元素的采样方法，以及将 `batch` 结果整理成模型所需输入形式的方法，并且能够使用多进程读取数据。

​	在绝大部分情况下，用户只需实现 `Dataset` 的`__len__`方法和`__getitem__`方法，就可以轻松构建自己的数据集，并用默认数据管道进行加载。



#### 5.1.1. Dataset和DataLoader概述

##### 5.1.1.1. 获取一个batch数据的步骤

​	让我们考虑一下从一个数据集中获取一个batch的数据需要哪些步骤。

​	(假定数据集的特征和标签分别表示为张量`X`和`Y`，数据集可以表示为`(X,Y)`, 假定batch大小为`m`)

1. 首先我们要确定数据集的长度`n`。

   ​	结果类似：`n = 1000`。

2. 然后我们从`0`到`n-1`的范围中抽样出`m`个数(batch大小)。

   ​	假定`m=4`, 拿到的结果是一个列表，类似：`indices = [1,4,8,9]`

3. 接着我们从数据集中去取这`m`个数对应下标的元素。

   ​	拿到的结果是一个元组列表，类似：`samples = [(X[1],Y[1]),(X[4],Y[4]),(X[8],Y[8]),(X[9],Y[9])]`

4. 最后我们将结果整理成两个张量作为输出。

   ​	拿到的结果是两个张量，类似`batch = (features,labels)`，

   ​	其中 `features = torch.stack([X[1],X[4],X[8],X[9]])`、

   ​             `labels = torch.stack([Y[1],Y[4],Y[8],Y[9]])`



---

##### 5.1.1.2. Dataset和DataLoader的功能分工

​	上述第1个步骤确定数据集的长度是由 Dataset 的`__len__` 方法实现的。

​	第2个步骤从`0`到`n-1`的范围中抽样出`m`个数的方法是由 DataLoader的 `sampler`和 `batch_sampler`参数指定的。

​	`sampler`参数指定单个元素抽样方法，一般无需用户设置，程序默认在DataLoader的参数`shuffle=True`时采用随机抽样，`shuffle=False`时采用顺序抽样。

​	`batch_sampler`参数将多个抽样的元素整理成一个列表，一般无需用户设置，默认方法在DataLoader的参数`drop_last=True`时会丢弃数据集最后一个长度不能被batch大小整除的批次，在`drop_last=False`时保留最后一个批次。

​	第3个步骤的核心逻辑根据下标取数据集中的元素 是由 Dataset的 `__getitem__`方法实现的。

​	第4个步骤的逻辑由DataLoader的参数`collate_fn`指定。一般情况下也无需用户设置。



---

##### 5.1.1.3. Dataset和DataLoader的主要接口

```python
# 伪代码
import torch 
class Dataset(object):
    def __init__(self):
        pass
    
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self,index):
        raise NotImplementedError
        

class DataLoader(object):
    def __init__(self,dataset,batch_size,collate_fn,shuffle = True,drop_last = False):
        self.dataset = dataset
        self.sampler =torch.utils.data.RandomSampler if shuffle else \
           torch.utils.data.SequentialSampler
        self.batch_sampler = torch.utils.data.BatchSampler
        self.sample_iter = self.batch_sampler(
            self.sampler(range(len(dataset))),
            batch_size = batch_size,drop_last = drop_last)
        
    def __next__(self):
        indices = next(self.sample_iter)
        batch = self.collate_fn([self.dataset[i] for i in indices])
        return batch
```





---

#### 5.1.2. 使用Dataset创建数据集

Dataset创建数据集常用的方法有：

- 使用 `torch.utils.data.TensorDataset `根据Tensor创建数据集(numpy的array，Pandas的DataFrame需要先转换成Tensor)。
- 使用 `torchvision.datasets.ImageFolder` 根据图片目录创建图片数据集。
- 继承 `torch.utils.data.Dataset` 创建自定义数据集。

此外，还可以通过

- `torch.utils.data.random_split` 将一个数据集分割成多份，常用于分割训练集，验证集和测试集。
- 调用Dataset的加法运算符(`+`)将多个数据集合并成一个数据集。



##### 5.1.2.1. 根据Tensor创建数据集

```python
# 根据Tensor创建数据集

from sklearn import datasets 
iris = datasets.load_iris()
ds_iris = TensorDataset(torch.tensor(iris.data),torch.tensor(iris.target))

# 分割成训练集和预测集
n_train = int(len(ds_iris)*0.8)
n_valid = len(ds_iris) - n_train
ds_train,ds_valid = random_split(ds_iris,[n_train,n_valid])

print(type(ds_iris))
print(type(ds_train))

'''output'''
<class 'torch.utils.data.dataset.TensorDataset'>
<class 'torch.utils.data.dataset.Subset'>
```



```python
# 使用DataLoader加载数据集
dl_train,dl_valid = DataLoader(ds_train,batch_size = 8),DataLoader(ds_valid,batch_size = 8)
print(type(dl_train))
for features,labels in dl_train:
    print(features,labels)
    break
    
'''output'''
<class 'torch.utils.data.dataloader.DataLoader'>
tensor([[6.2000, 2.8000, 4.8000, 1.8000],
        [4.6000, 3.6000, 1.0000, 0.2000],
        [4.8000, 3.1000, 1.6000, 0.2000],
        [5.1000, 2.5000, 3.0000, 1.1000],
        [6.3000, 2.9000, 5.6000, 1.8000],
        [5.8000, 2.7000, 4.1000, 1.0000],
        [5.6000, 2.7000, 4.2000, 1.3000],
        [5.0000, 3.4000, 1.5000, 0.2000]], dtype=torch.float64) tensor([2, 0, 0, 1, 2, 1, 1, 0])
```



```python
# 演示加法运算符（`+`）的合并作用
ds_data = ds_train + ds_valid
print('len(ds_train) = ',len(ds_train))
print('len(ds_valid) = ',len(ds_valid))
print('len(ds_train+ds_valid) = ',len(ds_data))
print(type(ds_train))
print(type(ds_data))

'''output'''
len(ds_train) =  120
len(ds_valid) =  30
len(ds_train+ds_valid) =  150
<class 'torch.utils.data.dataset.Subset'>
<class 'torch.utils.data.dataset.ConcatDataset'>
```



---

##### 5.1.2.2. 根据图片目录创建图片数据集

```python
from PIL import Image
img = Image.open('picture.jpg')

# 随机数值翻转 翻转变换
transforms.RandomVerticalFlip()(img)

# 旋转45度  旋转变换
transforms.RandomRotation(45)(img)
```



```python
# 定义图片增强操作
transform_train = transforms.Compose([
   transforms.RandomHorizontalFlip(), #随机水平翻转
   transforms.RandomVerticalFlip(), #随机垂直翻转
   transforms.RandomRotation(45),  #随机在45度角度内旋转
   transforms.ToTensor() #转换成张量
  ]
) 

transform_valid = transforms.Compose([
    transforms.ToTensor()
  ]
)


# 根据图片目录创建数据集
ds_train = datasets.ImageFolder("./data/cifar2/train/",
            transform = transform_train,target_transform= lambda t:torch.tensor([t]).float())
ds_valid = datasets.ImageFolder("./data/cifar2/test/",
            transform = transform_train,target_transform= lambda t:torch.tensor([t]).float())
print(ds_train.class_to_idx)


# 使用DataLoader加载数据集
dl_train = DataLoader(ds_train,batch_size = 50,shuffle = True,num_workers=3)
dl_valid = DataLoader(ds_valid,batch_size = 50,shuffle = True,num_workers=3)

for features,labels in dl_train:
    print(features.shape)
    print(labels.shape)
    break
    
    
'''output'''
{'0_airplane': 0, '1_automobile': 1}
torch.Size([50, 3, 32, 32])
torch.Size([50, 1])
```



---

##### 5.1.2.3. 创建自定义数据集

​	下面通过继承`Dataset类`创建`imdb`文本分类任务的自定义数据集。

​	大概思路如下：首先，对训练集文本分词构建词典。然后将训练集文本和测试集文本数据转换成`token`单词编码。

​	接着将转换成单词编码的训练集数据和测试集数据按样本分割成多个文件，一个文件代表一个样本。

​	最后，我们可以根据文件名列表获取对应序号的样本内容，从而构建`Dataset`数据集。

---

​	首先我们构建词典，并保留最高频的MAX_WORDS个词。

```python
#清洗文本
def clean_text(text):
    lowercase = text.lower().replace("\n"," ") # 小写
    stripped_html = re.sub('<br />', ' ',lowercase) # 去除网页字符<br>
    cleaned_punctuation = re.sub('[%s]'%re.escape(string.punctuation),'',stripped_html) # 去除标点符号
    return cleaned_punctuation

with open(train_data_path,"r",encoding = 'utf-8') as f:  # open r只读
    for line in f:
        label,text = line.split("\t") # 每行分割一下
        cleaned_text = clean_text(text)
        for word in cleaned_text.split(" "):
            word_count_dict[word] = word_count_dict.get(word,0)+1 
```

​	然后我们利用构建好的词典，将文本转换成token序号。

```python
def text_to_token(text_file,token_file):
    with open(text_file,"r",encoding = 'utf-8') as fin,\
      open(token_file,"w",encoding = 'utf-8') as fout:   # open w新建
        for line in fin:
            label,text = line.split("\t")
            cleaned_text = clean_text(text)
            word_token_list = [word_id_dict.get(word, 0) for word in cleaned_text.split(" ")]
            pad_list = pad(word_token_list,MAX_LEN)
            out_line = label+"\t"+" ".join([str(x) for x in pad_list])
            fout.write(out_line+"\n")
```

​	接着将token文本按照样本分割，每个文件存放一个样本的数据。

```python
# 分割样本 
def split_samples(token_path,samples_dir):
    with open(token_path,"r",encoding = 'utf-8') as fin:
        i = 0
        for line in fin:
            with open(samples_dir+"%d.txt"%i,"w",encoding = "utf-8") as fout:
                fout.write(line)
            i = i+1

split_samples(train_token_path,train_samples_path)
split_samples(test_token_path,test_samples_path)
```

​	一切准备就绪，我们可以创建数据集`Dataset`, 从文件名称列表中读取文件内容了。

```python
class imdbDataset(Dataset):
    def __init__(self,samples_dir):
        self.samples_dir = samples_dir
        self.samples_paths = os.listdir(samples_dir)
    
    def __len__(self):
        return len(self.samples_paths)
    
    def __getitem__(self,index):
        path = self.samples_dir + self.samples_paths[index]
        with open(path,"r",encoding = "utf-8") as f:
            line = f.readline()
            label,tokens = line.split("\t")
            label = torch.tensor([float(label)],dtype = torch.float)
            feature = torch.tensor([int(x) for x in tokens.split(" ")],dtype = torch.long)
            return  (feature,label)
# dataset
ds_train = imdbDataset(train_samples_path)
ds_test = imdbDataset(test_samples_path).
# dataloader
dl_train = DataLoader(ds_train,batch_size = BATCH_SIZE,shuffle = True,num_workers=4)
dl_test = DataLoader(ds_test,batch_size = BATCH_SIZE,num_workers=4)
```





---

#### 5.1.3. 使用DataLoader加载数据集

​	`DataLoader` 能够控制batch的大小，`batch` 中元素的采样方法，以及将`batch` 结果整理成模型所需输入形式的方法，并且能够使用多进程读取数据。

​	`DataLoader` 的函数签名如下。

```python
DataLoader(  
    dataset,  
    batch_size=1,  
    shuffle=False,  
    sampler=None,  
    batch_sampler=None,  
    num_workers=0,  
    collate_fn=None,  
    pin_memory=False,  
    drop_last=False,  
    timeout=0,  
    worker_init_fn=None,  
    multiprocessing_context=None,  
)
```

​	一般情况下，我们仅仅会配置 `dataset, batch_size, shuffle, num_workers, drop_last`这五个参数，其他参数使用默认值即可。

​	`DataLoader` 除了可以加载我们前面讲的 `torch.utils.data.Dataset`  外，还能够加载另外一种数据集 `torch.utils.data.IterableDataset`。和`Dataset`数据集相当于一种列表结构不同，`IterableDataset` 相当于一种迭代器结构。 它更加复杂，一般较少使用。

- dataset : 数据集
- batch_size: 批次大小
- shuffle: 是否乱序
- sampler: 样本采样函数，一般无需设置。
- batch_sampler: 批次采样函数，一般无需设置。
- num_workers: 使用多进程读取数据，设置的进程数。
- collate_fn: 整理一个批次数据的函数。
- pin_memory: 是否设置为锁业内存。默认为False，锁业内存不会使用虚拟内存(硬盘)，从锁业内存拷贝到GPU上速度会更快。
- drop_last: 是否丢弃最后一个样本数量不足batch_size批次数据。
- timeout: 加载一个数据批次的最长等待时间，一般无需设置。
- worker_init_fn: 每个worker中dataset的初始化函数，常用于 IterableDataset。一般不使用。



```python
#构建输入数据管道
ds = TensorDataset(torch.arange(1,50))
dl = DataLoader(ds,
                batch_size = 10,
                shuffle= True,
                num_workers=2,
                drop_last = True)
#迭代数据
for batch, in dl:
    print(batch)
```



---

### 5.2. 模型层

​	深度学习模型一般由各种模型层组合而成。

​	`torch.nn` 中内置了非常丰富的各种模型层。它们都属于 `nn.Module` 的子类，具备参数管理功能。

例如：

- nn.Linear, nn.Flatten, nn.Dropout, nn.BatchNorm2d
- nn.Conv2d,nn.AvgPool2d,nn.Conv1d,nn.ConvTranspose2d
- nn.Embedding,nn.GRU,nn.LSTM
- nn.Transformer

​	如果这些内置模型层不能够满足需求，我们也可以通过继承 `nn.Module` 基类构建自定义的模型层。

​	实际上，pytorch不区分模型和模型层，都是通过继承 `nn.Module` 进行构建。

​	因此，我们只要继承 `nn.Module` 基类并实现 `forward` 方法即可自定义模型层。



---

#### 5.2.1. 内置模型层

一些常用的内置模型层简单介绍如下。

##### 5.2.1.1 基础层

- nn.Linear：全连接层。参数个数 = 输入层特征数× 输出层特征数(weight)＋ 输出层特征数(bias)
- nn.Flatten：压平层，用于将多维张量样本压成一维张量样本。
- nn.BatchNorm1d：一维批标准化层。通过线性变换将输入批次缩放平移到稳定的均值和标准差。可以增强模型对输入不同分布的适应性，加快模型训练速度，有轻微正则化效果。一般在激活函数之前使用。可以用afine参数设置该层是否含有可以训练的参数。
- nn.BatchNorm2d：二维批标准化层。
- nn.BatchNorm3d：三维批标准化层。
- nn.Dropout：一维随机丢弃层。一种正则化手段。
- nn.Dropout2d：二维随机丢弃层。
- nn.Dropout3d：三维随机丢弃层。
- nn.Threshold：限幅层。当输入大于或小于阈值范围时，截断之。
- nn.ConstantPad2d： 二维常数填充层。对二维张量样本填充常数扩展长度。
- nn.ReplicationPad1d： 一维复制填充层。对一维张量样本通过复制边缘值填充扩展长度。
- nn.ZeroPad2d：二维零值填充层。对二维张量样本在边缘填充0值.
- nn.GroupNorm：组归一化。一种替代批归一化的方法，将通道分成若干组进行归一。不受batch大小限制，据称性能和效果都优于BatchNorm。
- nn.LayerNorm：层归一化。较少使用。
- nn.InstanceNorm2d: 样本归一化。较少使用。



---

##### 5.2.1.2. 卷积网络相关层

- nn.Conv1d：普通一维卷积，常用于文本。参数个数 = 输入通道数×卷积核尺寸(如3)×卷积核个数 + 卷积核尺寸(如3）
- nn.Conv2d：普通二维卷积，常用于图像。参数个数 = 输入通道数×卷积核尺寸(如3乘3)×卷积核个数 + 卷积核尺寸(如3乘3)
  通过调整dilation参数大于1，可以变成空洞卷积，增大卷积核感受野。
  通过调整groups参数不为1，可以变成分组卷积。分组卷积中不同分组使用相同的卷积核，显著减少参数数量。
  当groups参数等于通道数时，相当于tensorflow中的二维深度卷积层tf.keras.layers.DepthwiseConv2D。
  利用分组卷积和1乘1卷积的组合操作，可以构造相当于Keras中的二维深度可分离卷积层tf.keras.layers.SeparableConv2D。
- nn.Conv3d：普通三维卷积，常用于视频。参数个数 = 输入通道数×卷积核尺寸(如3乘3乘3)×卷积核个数 + 卷积核尺寸(如3乘3乘3) 。
- nn.MaxPool1d: 一维最大池化。
- nn.MaxPool2d：二维最大池化。一种下采样方式。没有需要训练的参数。
- nn.MaxPool3d：三维最大池化。
- nn.AdaptiveMaxPool2d：二维自适应最大池化。无论输入图像的尺寸如何变化，输出的图像尺寸是固定的。
  该函数的实现原理，大概是通过输入图像的尺寸和要得到的输出图像的尺寸来反向推算池化算子的padding,stride等参数。
- nn.FractionalMaxPool2d：二维分数最大池化。普通最大池化通常输入尺寸是输出的整数倍。而分数最大池化则可以不必是整数。分数最大池化使用了一些随机采样策略，有一定的正则效果，可以用它来代替普通最大池化和Dropout层。
- nn.AvgPool2d：二维平均池化。
- nn.AdaptiveAvgPool2d：二维自适应平均池化。无论输入的维度如何变化，输出的维度是固定的。
- nn.ConvTranspose2d：二维卷积转置层，俗称反卷积层。并非卷积的逆操作，但在卷积核相同的情况下，当其输入尺寸是卷积操作输出尺寸的情况下，卷积转置的输出尺寸恰好是卷积操作的输入尺寸。在语义分割中可用于上采样。
- nn.Upsample：上采样层，操作效果和池化相反。可以通过mode参数控制上采样策略为"nearest"最邻近策略或"linear"线性插值策略。
- nn.Unfold：滑动窗口提取层。其参数和卷积操作nn.Conv2d相同。实际上，卷积操作可以等价于nn.Unfold和nn.Linear以及nn.Fold的一个组合。
  其中nn.Unfold操作可以从输入中提取各个滑动窗口的数值矩阵，并将其压平成一维。利用nn.Linear将nn.Unfold的输出和卷积核做乘法后，再使用
  nn.Fold操作将结果转换成输出图片形状。
- nn.Fold：逆滑动窗口提取层。



---

##### 5.2.1.3.循环网络相关层

- nn.Embedding：嵌入层。一种比Onehot更加有效的对离散特征进行编码的方法。一般用于将输入中的单词映射为稠密向量。嵌入层的参数需要学习。
- nn.LSTM：长短记忆循环网络层【支持多层】。最普遍使用的循环网络层。具有携带轨道，遗忘门，更新门，输出门。可以较为有效地缓解梯度消失问题，从而能够适用长期依赖问题。设置bidirectional = True时可以得到双向LSTM。需要注意的时，默认的输入和输出形状是(seq,batch,feature), 如果需要将batch维度放在第0维，则要设置batch_first参数设置为True。
- nn.GRU：门控循环网络层【支持多层】。LSTM的低配版，不具有携带轨道，参数数量少于LSTM，训练速度更快。
- nn.RNN：简单循环网络层【支持多层】。容易存在梯度消失，不能够适用长期依赖问题。一般较少使用。
- nn.LSTMCell：长短记忆循环网络单元。和nn.LSTM在整个序列上迭代相比，它仅在序列上迭代一步。一般较少使用。
- nn.GRUCell：门控循环网络单元。和nn.GRU在整个序列上迭代相比，它仅在序列上迭代一步。一般较少使用。
- nn.RNNCell：简单循环网络单元。和nn.RNN在整个序列上迭代相比，它仅在序列上迭代一步。一般较少使用。



---

##### 5.2.1.4. Transformer相关层

- nn.Transformer：Transformer网络结构。Transformer网络结构是替代循环网络的一种结构，解决了循环网络难以并行，难以捕捉长期依赖的缺陷。它是目前NLP任务的主流模型的主要构成部分。Transformer网络结构由TransformerEncoder编码器和TransformerDecoder解码器组成。编码器和解码器的核心是MultiheadAttention多头注意力层。
- nn.TransformerEncoder：Transformer编码器结构。由多个 nn.TransformerEncoderLayer编码器层组成。
- nn.TransformerDecoder：Transformer解码器结构。由多个 nn.TransformerDecoderLayer解码器层组成。
- nn.TransformerEncoderLayer：Transformer的编码器层。
- nn.TransformerDecoderLayer：Transformer的解码器层。
- nn.MultiheadAttention：多头注意力层。



---

#### 5.2.2. 自定义模型层

​	如果Pytorch的内置模型层不能够满足需求，我们也可以通过继承`nn.Module`基类构建自定义的模型层。

​	实际上，pytorch不区分模型和模型层，都是通过继承`nn.Module`进行构建。

​	因此，我们只要继承`nn.Module`基类并实现`forward`方法即可自定义模型层。

​	下面是Pytorch的`nn.Linear`层的源码，我们可以仿照它来自定义模型层。



```python
class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
```



----

### 5.3. 损失函数

​	一般来说，监督学习的目标函数由损失函数和正则化项组成。`(Objective = Loss + Regularization)`

​	Pytorch中的损失函数一般在训练模型时候指定。

​	注意Pytorch中内置的损失函数的参数和tensorflow不同，是`y_pred`在前，`y_true`在后，而Tensorflow是y_true在前，y_pred在后。

​	对于回归模型，通常使用的内置损失函数是均方损失函数`nn.MSELoss `。

​	对于二分类模型，通常使用的是二元交叉熵损失函数`nn.BCELoss `(输入已经是sigmoid激活函数之后的结果)
或者 `nn.BCEWithLogitsLoss` (输入尚未经过nn.Sigmoid激活函数) 。

​	对于多分类模型，一般推荐使用交叉熵损失函数 `nn.CrossEntropyLoss`。(`y_true` 需要是一维的，是类别编码。`y_pred` 未经过`nn.Softmax`激活。)    

​	此外，如果多分类的`y_pred`经过了 `nn.LogSoftmax` 激活，可以使用 `nn.NLLLoss` 损失函数(The negative log likelihood loss)。这种方法和直接使用`nn.CrossEntropyLoss`等价。



​	如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量`y_pred，y_true`作为输入参数，并输出一个标量作为损失函数值。

​	`Pytorch` 中的正则化项一般通过自定义的方式和损失函数一起添加作为目标函数。如果仅仅使用L2正则化，也可以利用优化器的 `weight_decay` 参数来实现相同的效果。



---

#### 5.3.1. 内置损失函数

```python
y_pred = torch.tensor([[10.0,0.0,-10.0],[8.0,8.0,8.0]])
y_true = torch.tensor([0,2])

# 直接调用交叉熵损失
ce = nn.CrossEntropyLoss()(y_pred,y_true)
print(ce)

# 等价于先计算nn.LogSoftmax激活，再调用NLLLoss
y_pred_logsoftmax = nn.LogSoftmax(dim = 1)(y_pred)
nll = nn.NLLLoss()(y_pred_logsoftmax,y_true)
print(nll)
```

​	内置的损失函数一般有 `类的实现` 和 `函数` 的实现两种形式。

​	如：`nn.BCE` 和 `F.binary_cross_entropy`  都是二元交叉熵损失函数，前者是类的实现形式，后者是函数的实现形式。

​	实际上**类的实现形式**通常是调用**函数的实现形式**并用**nn.Module**封装后得到的。

​	一般我们常用的是类的实现形式。它们封装在`torch.nn`模块下，并且类名以`Loss`结尾。

常用的一些内置损失函数说明如下。

- nn.MSELoss（均方误差损失，也叫做L2损失，用于回归）
- nn.L1Loss （L1损失，也叫做绝对值误差损失，用于回归）
- nn.SmoothL1Loss (平滑L1损失，当输入在-1到1之间时，平滑为L2损失，用于回归)
- nn.BCELoss (二元交叉熵，用于二分类，输入已经过nn.Sigmoid激活，对不平衡数据集可以用weigths参数调整类别权重)
- nn.BCEWithLogitsLoss (二元交叉熵，用于二分类，输入未经过nn.Sigmoid激活)
- nn.CrossEntropyLoss (交叉熵，用于多分类，要求label为稀疏编码，输入未经过nn.Softmax激活，对不平衡数据集可以用weigths参数调整类别权重)
- nn.NLLLoss (负对数似然损失，用于多分类，要求label为稀疏编码，输入经过nn.LogSoftmax激活)
- nn.CosineSimilarity(余弦相似度，可用于多分类)
- nn.AdaptiveLogSoftmaxWithLoss (一种适合非常多类别且类别分布很不均衡的损失函数，会自适应地将多个小类别合成一个cluster)



---

#### 5.3.2. 自定义损失函数

​	自定义损失函数接收两个张量`y_pred,y_true`作为输入参数，并输出一个标量作为损失函数值。也可以对`nn.Module`进行**子类化**，**重写 forward 方法**实现损失的计算逻辑，从而得到损失函数的类的实现。

​	下面是一个`Focal Loss`的自定义实现示范。`Focal Loss`是一种对`binary_crossentropy`的改进损失函数形式。

​	它在样本不均衡和存在较多易分类的样本时相比`binary_crossentropy`具有明显的优势。

​	它有两个可调参数，`alpha`参数和`gamma`参数。其中`alpha`参数主要用于**衰减负样本的权重**，`gamma`参数主要用于**衰减容易训练样本的权重**。从而让模型更加**聚焦**在**正**样本和**困难**样本上。这就是为什么这个损失函数叫做`Focal Loss`。

《5分钟理解Focal Loss与GHM——解决样本不平衡利器》https://zhuanlan.zhihu.com/p/80594704

```python
class FocalLoss(nn.Module):
    
    def __init__(self,gamma=2.0,alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self,y_pred,y_true):
        bce = torch.nn.BCELoss(reduction = "none")(y_pred,y_true)
        #bce = - y_true*torch.log(y_pred) - (1-y_true)*torch.log(1-y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        loss = torch.mean(alpha_factor * modulating_factor * bce)
        return loss
```

$$
C E=\left\{\begin{aligned}
-\log (p), & \text { if } \quad y=1 \\
-\log (1-p), & \text { if } \quad y=0
\end{aligned}\right.
$$


$$
F L=\left\{\begin{aligned}
-\alpha(1-p)^{\gamma} \log (p), & \text { if } & y=1 \\
-(1-\alpha) p^{\gamma} \log (1-p), & \text { if } & y=0
\end{aligned}\right.
$$




---

#### 5.3.3. 自定义L1和L2正则化项

​	通常认为 `L1` 正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择。而 `L2` 正则化可以防止模型过拟合（overfitting）。一定程度上，`L1` 也可以防止过拟合。

##### 5.3.3.1. 准备数据

​	同3.1.2.1. 准备数据	

---

##### 5.3.3.2. 打乱数据

​	同3.3.2.2. 打乱数据

---

##### 5.3.3.3.  网络模型

```python
class DNNModel(torchkeras.Model):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,8) 
        self.fc3 = nn.Linear(8,1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = nn.Sigmoid()(self.fc3(x))
        return y
        
model = DNNModel()
model.summary(input_shape =(2,))
```



---

##### 5.3.3.4. 训练模型

```python
# 准确率
def accuracy(y_pred,y_true):
    y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                      torch.zeros_like(y_pred,dtype = torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc

# L2正则化
def L2Loss(model,alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name: #一般不对偏置项使用正则
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(param, 2)))
    return l2_loss

# L1正则化
def L1Loss(model,beta):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss +  beta * torch.sum(torch.abs(param))
    return l1_loss

# 将L2正则和L1正则添加到FocalLoss损失，一起作为目标函数
def focal_loss_with_regularization(y_pred,y_true):
    focal = FocalLoss()(y_pred,y_true) 
    l2_loss = L2Loss(model,0.001) #注意设置正则化项系数
    l1_loss = L1Loss(model,0.001)
    total_loss = focal + l2_loss + l1_loss
    return total_loss

model.compile(loss_func = focal_loss_with_regularization,
              optimizer = torch.optim.Adam(model.parameters(),lr = 0.01),
             metrics_dict = {"accuracy":accuracy})
```



---

#### 5.3.4. 通过优化器实现L2正则化

​	如果仅仅需要使用 `L2` 正则化，那么也可以利用**优化器optim**的 `weight_decay` 参数来实现。

​	`weight_decay` 参数可以设置参数在训练过程中的衰减，这和 `L2` 正则化的作用效果等价。

```python
# before L2 regularization:  
gradient descent: w = w - lr * dloss_dw   

# after L2 regularization:  
gradient descent: w = w - lr * (dloss_dw+beta*w) = (1-lr*beta)*w - lr*dloss_dw  

# so （1-lr*beta）is the weight decay ratio.
```

​	Pytorch 的优化器支持一种称之为 `Per-parameter options` 的操作，就是对每一个参数进行特定的**学习率**，**权重衰减率**指定，以满足更为细致的要求。

```python
weight_params = [param for name, param in model.named_parameters() if "bias" not in name]
bias_params = [param for name, param in model.named_parameters() if "bias" in name]

optimizer = torch.optim.SGD([{'params': weight_params, 'weight_decay':1e-5},
                             {'params': bias_params, 'weight_decay':0}],
                            lr=1e-2, momentum=0.9)
```



----

### 5.4. TensorBoard可视化

​	在我们的炼丹过程中，如果能够使用丰富的图像来展示模型的结构，指标的变化，参数的分布，输入的形态等信息，无疑会提升我们对问题的洞察力，并增加许多炼丹的乐趣。

​	`TensorBoard`正是这样一个神奇的炼丹可视化辅助工具。Pytorch中利用`TensorBoard`可视化的大概过程如下：

​	首先在`Pytorch`中指定一个目录创建一个`torch.utils.tensorboard.SummaryWriter`日志写入器。

​	然后根据需要可视化的信息，利用日志写入器将相应信息日志写入我们指定的目录。

​	最后就可以传入日志目录作为参数启动`TensorBoard`，然后就可以在`TensorBoard`中愉快地看图了。

​	我们主要介绍Pytorch中利用`TensorBoard`进行如下方面信息的可视化的方法。

- 可视化模型结构： writer.add_graph
- 可视化指标变化： writer.add_scalar
- 可视化参数分布： writer.add_histogram
- 可视化原始图像： writer.add_image 或 writer.add_images
- 可视化人工绘图： writer.add_figure



---

##### 5.4.1. 可视化模型结构

```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
        self.dropout = nn.Dropout2d(p = 0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y
        
net = Net()
print(net)
```



---

##### 5.4.2. 可视化指标变化

​	有时候在训练过程中，如果能够实时动态地查看 `loss` 和各种 `metric` 的变化曲线，那么无疑可以帮助我们更加直观地了解模型的训练情况。

​	注意，`writer.add_scalar` 仅能对标量的值的变化进行可视化。因此它一般用于对 `loss` 和 `metric` 的变化进行可视化分析。



```python
import numpy as np 
import torch 
from torch.utils.tensorboard import SummaryWriter

# f(x) = a*x**2 + b*x + c的最小值
x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x],lr = 0.01)


def f(x):
    result = a*torch.pow(x,2) + b*x + c 
    return(result)

writer = SummaryWriter('./data/tensorboard')
for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()
    writer.add_scalar("x",x.item(),i) #日志中记录x在第step i 的值
    writer.add_scalar("y",y.item(),i) #日志中记录y在第step i 的值

writer.close()
    
print("y=",f(x).data,";","x=",x.data)
```



---

##### 5.4.3. 可视化参数分布

​	如果需要对模型的参数(一般非标量)在训练过程中的变化进行可视化，可以使用 `writer.add_histogram`。

​	它能够观测张量值分布的直方图随训练步骤的变化趋势。

```python
# 创建正态分布的张量模拟参数矩阵
def norm(mean,std):
    t = std*torch.randn((100,20))+mean
    return t

writer = SummaryWriter('./data/tensorboard')
for step,mean in enumerate(range(-10,10,1)):
    w = norm(mean,1)
    writer.add_histogram("w",w, step)
    writer.flush()
writer.close()
```



---

##### 5.4.4. 可视化原始图像

​	如果我们做图像相关的任务，也可以将原始的图片在`tensorboard`中进行可视化展示。

​	如果只写入一张图片信息，可以使用`writer.add_image`。如果要写入多张图片信息，可以使用`writer.add_images`。也可以用 `torchvision.utils.make_grid` 将多张图片拼成一张图片，然后用 `writer.add_image` 写入。

​	注意，传入的是代表图片信息的Pytorch中的张量数据。

```python
ds_train = datasets.ImageFolder("./data/cifar2/train/",
            transform = transform_train,target_transform= lambda t:torch.tensor([t]).float())
ds_valid = datasets.ImageFolder("./data/cifar2/test/",
            transform = transform_train,target_transform= lambda t:torch.tensor([t]).float())

print(ds_train.class_to_idx)

dl_train = DataLoader(ds_train,batch_size = 50,shuffle = True,num_workers=3)
dl_valid = DataLoader(ds_valid,batch_size = 50,shuffle = True,num_workers=3)

dl_train_iter = iter(dl_train)
images, labels = dl_train_iter.next()

# 仅查看一张图片
writer = SummaryWriter('./data/tensorboard')
writer.add_image('images[0]', images[0])
writer.close()

# 将多张图片拼接成一张图片，中间用黑色网格分割
writer = SummaryWriter('./data/tensorboard')
# create grid of images
img_grid = torchvision.utils.make_grid(images)
writer.add_image('image_grid', img_grid)
writer.close()

# 将多张图片直接写入
writer = SummaryWriter('./data/tensorboard')
writer.add_images("images",images,global_step = 0)
writer.close()
```



---

##### 5.4.5. 可视化人工绘图

​	如果我们将matplotlib绘图的结果再 tensorboard中展示，可以使用 `add_figure.`

​	注意，和writer.add_image不同的是，`writer.add_figure`需要传入matplotlib的`figure`对象。

```python
figure = plt.figure(figsize=(8,8)) 
for i in range(9):
    img,label = ds_train[i]
    img = img.permute(1,2,0)
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label.item())
    ax.set_xticks([])
    ax.set_yticks([]) 
plt.show()


writer = SummaryWriter('./data/tensorboard')
writer.add_figure('figure',figure,global_step=0)
writer.close()    
```

​	

------

## 6. Pytorch的高阶API

Pytorch没有官方的高阶API。一般通过nn.Module来构建模型并编写自定义训练循环。

为了更加方便地训练模型，作者编写了仿keras的Pytorch模型接口：torchkeras， 作为Pytorch的高阶API。

本章我们主要详细介绍Pytorch的高阶API如下相关的内容。

- 构建模型的3种方法（继承nn.Module基类，使用nn.Sequential，辅助应用模型容器）
- 训练模型的3种方法（脚本风格，函数风格，torchkeras.Model类风格）
- 使用GPU训练模型（单GPU训练，多GPU训练）

---

### 6.1. 构建模型的3种方法

​	可以使用以下3种方式构建模型：

​		1. 继承nn.Module基类构建自定义模型。

​		2. 使用nn.Sequential按层顺序构建模型。

​		3. 继承nn.Module基类构建模型并辅助应用模型容器进行封装

​	其中 第1种方式最为常见，第2种方式最简单，第3种方式最为灵活也较为复杂。推荐使用**第1种方式**构建模型。



#### 6.1.1. 继承nn.Module基类构建自定义模型

​	以下是继承nn.Module基类构建自定义模型的一个范例。模型中的用到的层一般在`__init__`函数中定义，然后在`forward`方法中定义模型的正向传播逻辑。

```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
        self.pool1 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        
    def forward(self,x):
        x = self.conv1(x)
        y = self.pool1(x)
        return y
        
net = Net()
print(net)

summary(net,input_shape= (3,32,32))
```



#### 6.1.2. 使用nn.Sequential按层顺序构建模型

​	使用nn.Sequential按层顺序构建模型无需定义forward方法。仅仅适合于简单的模型。

​	以下是使用nn.Sequential搭建模型的一些等价方法。

##### 6.1.2.1. 利用add_module方法

```python
net = nn.Sequential()
net.add_module("conv1",nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3))
net.add_module("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2))

print(net)
```



##### 6.1.2.2. 利用变长参数

​	这种方式构建时不能给每个层指定名称。

```python
net = nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
    nn.MaxPool2d(kernel_size = 2,stride = 2),
)

print(net)
```



##### 6.1.2.3. 利用OrderedDict

```python
from collections import OrderedDict

net = nn.Sequential(OrderedDict(
          [("conv1",nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)),
            ("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2))
          ])
        )

print(net)
```



#### 6.1.3. 继承nn.Module基类构建模型并辅助应用模型容器进行封装

​	当模型的结构比较复杂时，我们可以应用模型容器(nn.Sequential,nn.ModuleList,nn.ModuleDict)对模型的部分结构进行封装。

​	这样做会让模型整体更加有层次感，有时候也能减少代码量。

​	注意，在下面的范例中我们每次仅仅使用一种模型容器，但实际上这些模型容器的使用是非常灵活的，可以在一个模型中任意组合任意嵌套使用。

##### 6.1.3.1. nn.Sequential作为模型容器

```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,32),
        )
    def forward(self,x):
        x = self.conv(x)
        y = self.dense(x)
        return y 
```



---

##### 6.1.3.2. nn.ModuleList作为模型容器

​	注意下面中的ModuleList不能用Python中的列表代替。

```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Flatten(),
            nn.Linear(64,32)]
        )
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
net = Net()
print(net)
```



---

##### nn.ModuleDict作为模型容器

​	注意下面中的ModuleDict不能用Python中的字典代替。

```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.layers_dict = nn.ModuleDict({"conv1":nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
               "pool": nn.MaxPool2d(kernel_size = 2,stride = 2),
               "flatten": nn.Flatten(),
               "linear1": nn.Linear(64,32)
              })
    def forward(self,x):
        layers = ["conv1","pool","flatten","linear1"]
        for layer in layers:
            x = self.layers_dict[layer](x)
        return x
net = Net()
print(net)
```





---

### 6.2. 训练模型的3种方法

​	Pytorch通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。

​	有3类典型的训练循环代码风格：脚本形式训练循环，函数形式训练循环，类形式训练循环。

​	下面以minist数据集的分类模型的训练为例，演示这3种训练模型的风格。

#### 6.2.1. 准备数据

```python
transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="/home/kesci/input/data6936/data/minist/",train=True,download=True,transform=transform)
ds_valid = torchvision.datasets.MNIST(root="/home/kesci/input/data6936/data/minist/",train=False,download=True,transform=transform)

dl_train =  torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid =  torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_valid))
```



---

#### 6.2.2. 脚本风格

​	脚本风格的训练循环最为常见。

```python
def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true,y_pred_cls)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(),lr = 0.01)
metric_func = accuracy
metric_name = "accuracy"

epochs = 3
log_step_freq = 100

dfhistory = pd.DataFrame(columns = ["epoch","loss",metric_name,"val_loss","val_"+metric_name]) 
print("Start Training...")
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("=========="*8 + "%s"%nowtime)

for epoch in range(1,epochs+1):  

    # 1，训练循环-------------------------------------------------
    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1
    
    for step, (features,labels) in enumerate(dl_train, 1):
    
        # 梯度清零
        optimizer.zero_grad()

        # 正向传播求损失
        predictions = net(features)
        loss = loss_func(predictions,labels)
        metric = metric_func(predictions,labels)
        
        # 反向传播求梯度
        loss.backward()
        optimizer.step()

        # 打印batch级别日志
        loss_sum += loss.item()
        metric_sum += metric.item()
        if step%log_step_freq == 0:   
            print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                  (step, loss_sum/step, metric_sum/step))
            
    # 2，验证循环-------------------------------------------------
    net.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1

    for val_step, (features,labels) in enumerate(dl_valid, 1):
        with torch.no_grad():
            predictions = net(features)
            val_loss = loss_func(predictions,labels)
            val_metric = metric_func(predictions,labels)

        val_loss_sum += val_loss.item()
        val_metric_sum += val_metric.item()

    # 3，记录日志-------------------------------------------------
    info = (epoch, loss_sum/step, metric_sum/step, 
            val_loss_sum/val_step, val_metric_sum/val_step)
    dfhistory.loc[epoch-1] = info
    
    # 打印epoch级别日志
    print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
          "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") 
          %info)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
        
print('Finished Training...')
```



---

#### 6.2.3. 函数风格

​	该风格在脚本形式上作了简单的函数封装。

```python
model = net
model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
model.loss_func = nn.CrossEntropyLoss()
model.metric_func = accuracy
model.metric_name = "accuracy"


def train_step(model,features,labels):
    # 训练模式，dropout层发生作用
    model.train()
    
    # 梯度清零
    model.optimizer.zero_grad()
    
    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)

    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()

    return loss.item(),metric.item()

@torch.no_grad()
def valid_step(model,features,labels):
    # 预测模式，dropout层不发生作用
    model.eval()
    
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)
    
    return loss.item(), metric.item()


# 测试train_step效果
features,labels = next(iter(dl_train))
train_step(model,features,labels)

def train_model(model,epochs,dl_train,dl_valid,log_step_freq):
    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns = ["epoch","loss",metric_name,"val_loss","val_"+metric_name]) 
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)

    for epoch in range(1,epochs+1):  

        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features,labels) in enumerate(dl_train, 1):

            loss,metric = train_step(model,features,labels)

            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step%log_step_freq == 0:   
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                      (step, loss_sum/step, metric_sum/step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features,labels) in enumerate(dl_valid, 1):

            val_loss,val_metric = valid_step(model,features,labels)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum/step, metric_sum/step, 
                val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
              "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") 
              %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)

    print('Finished Training...')
    return dfhistory

epochs = 3
dfhistory = train_model(model,epochs,dl_train,dl_valid,log_step_freq = 100)
```



---

#### 6.2.4. 类风格

​	此处使用 `torchkeras` 中定义的模型接口构建模型，并调用 `compile` 方法和 `fit` 方法训练模型。

​	使用该形式训练模型非常简洁明了。推荐使用该形式。

```python
import torchkeras 
class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)]
        )
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
model = torchkeras.Model(CnnModel())
print(model)

from sklearn.metrics import accuracy_score

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.numpy(),y_pred_cls.numpy())

model.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy})

dfhistory = model.fit(3,dl_train = dl_train, dl_val=dl_valid, log_step_freq=100)
```



---

### 6.3. 使用GPU训练模型

​	深度学习的训练过程常常非常耗时，一个模型训练几个小时是家常便饭，训练几天也是常有的事情，有时候甚至要训练几十天。

​	训练过程的耗时主要来自于两个部分，一部分来自**数据准备，另一部分来自**参数迭代**。

​	当数据准备过程还是模型训练时间的主要瓶颈时，我们可以使用更多**进程**来准备数据。

​	当参数迭代过程成为训练时间的主要瓶颈时，我们通常的方法是应用**GPU**来进行加速。

​	Pytorch中使用GPU加速模型非常简单，只要将模型和数据移动到GPU上。核心代码只有以下几行。

```python
# 定义模型  
...   

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
model.to(device) # 移动模型到cuda  

# 训练模型  
...  

features = features.to(device) # 移动数据到cuda  
labels = labels.to(device) # 或者  labels = labels.cuda() if torch.cuda.is_available() else labels  
...
```

​	如果要使用多个 **GPU** 训练模型，也非常简单。只需要在将模型设置为数据并行风格模型。
​	则模型移动到 GPU 上之后，会在每一个 GPU 上拷贝一个副本，并把数据平分到各个 GPU 上进行训练。核心代码如下。

```python
# 定义模型  
...   

if torch.cuda.device_count() > 1:  
    model = nn.DataParallel(model) # 包装为并行风格模型  

# 训练模型  
...  
features = features.to(device) # 移动数据到cuda  
labels = labels.to(device) # 或者 labels = labels.cuda() if torch.cuda.is_available() else labels  
...
```

**以下是一些和GPU有关的基本操作汇总**



```python
import torch 
from torch import nn 

# 1，查看gpu信息
if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)

gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)

'''
if_cuda= True  
gpu_count= 1
'''
```



```python
# 2，将张量在gpu和cpu间移动
tensor = torch.rand((100,100))
tensor_gpu = tensor.to("cuda:0") # 或者 tensor_gpu = tensor.cuda()
print(tensor_gpu.device)
print(tensor_gpu.is_cuda)

tensor_cpu = tensor_gpu.to("cpu") # 或者 tensor_cpu = tensor_gpu.cpu() 
print(tensor_cpu.device)

''' 
cuda:0
True
cpu
'''
```



```python
# 3，将模型中的全部张量移动到gpu上
net = nn.Linear(2,1)
print(next(net.parameters()).is_cuda)
net.to("cuda:0") # 将模型中的全部参数张量依次到GPU上，注意，无需重新赋值为 net = net.to("cuda:0")
print(next(net.parameters()).is_cuda)
print(next(net.parameters()).device)

''' 把 weight 和 bias 移到 GPU 上
False  
True  
cuda:0
'''
```



```python
# 4，创建支持多个gpu数据并行的模型
linear = nn.Linear(2,1)
print(next(linear.parameters()).device)

model = nn.DataParallel(linear)
print(model.device_ids)
print(next(model.module.parameters()).device) 

#注意保存参数时要指定保存model.module的参数
torch.save(model.module.state_dict(), "./data/model_parameter.pkl") 

linear = nn.Linear(2,1)
linear.load_state_dict(torch.load("./data/model_parameter.pkl")) 

'''
cpu  
[0]  
cuda:0
'''
```



```python
# 5，清空cuda缓存

# 该方法在cuda超内存时十分有用
torch.cuda.empty_cache()
```



#### 6.3.1. 矩阵乘法范例

```python
# 使用cpu
a = torch.rand((10000,200))
b = torch.rand((200,10000))
c = torch.matmul(a,b)


# 使用gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.rand((10000,200),device = device) #可以指定在GPU上创建张量
b = torch.rand((200,10000)) #也可以在CPU上创建张量后移动到GPU上
b = b.to(device) #或者 b = b.cuda() if torch.cuda.is_available() else b 
c = torch.matmul(a,b)
```



---

#### 6.3.2. 线性回归范例

```python
# 准备数据
n = 1000000 #样本数量

X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0,-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0.t() + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动

# 定义模型
class LinearRegression(nn.Module): 
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))
    #正向传播
    def forward(self,x): 
        return x@self.w.t() + self.b
        
linear = LinearRegression() 
```



```python
# 准备数据
n = 1000000 #样本数量

X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0,-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0.t() + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动

# 移动到GPU上
print("torch.cuda.is_available() = ",torch.cuda.is_available())
X = X.cuda()
Y = Y.cuda()
print("X.device:",X.device)
print("Y.device:",Y.device)

# 定义模型
class LinearRegression(nn.Module): 
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))
    #正向传播
    def forward(self,x): 
        return x@self.w.t() + self.b
        
linear = LinearRegression() 

# 移动模型到GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
linear.to(device)

#查看模型是否已经移动到GPU上
print("if on cuda:",next(linear.parameters()).is_cuda)
```



---

#### 6.3.3. torchkeras使用单GPU范例

​	下面演示使用torchkeras来应用GPU训练模型的方法。

​	其对应的CPU训练模型代码参见《6-2,训练模型的3种方法》

​	本例仅需要在它的基础上增加一行代码，在model.compile时指定`device`即可。

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # 注意此处compile时指定了device
```



---

#### 6.3.4. torchkeras使用多GPU范例

​	注：以下范例需要在有多个GPU的机器上跑。如果在单GPU的机器上跑，也能跑通，但是实际上使用的是单个GPU。

```python
net = nn.DataParallel(CnnModule())  #Attention this line!!!
model = torchkeras.Model(net)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # 注意此处compile时指定了device
```









---











