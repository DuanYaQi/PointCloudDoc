# 20天吃掉那只Pytorch

## Start

如果是工程师，应该优先选TensorFlow2.

如果是学生或者研究人员，应该优先选择Pytorch.

如果时间足够，最好TensorFlow2和Pytorch都要学习掌握。

原因：

​	1，在工业界最重要的是模型落地，目前国内的大部分互联网企业只支持TensorFlow模型的在线部署，不支持Pytorch。 并且工业界更加注重的是模型的高可用性，许多时候使用的都是成熟的模型架构，调试需求并不大。
​	2，研究人员最重要的是快速迭代发表文章，需要尝试一些较新的模型架构。而Pytorch在易用性上相比TensorFlow2有一些优势，更加方便调试。 并且在2019年以来在学术界占领了大半壁江山，能够找到的相应最新研究成果更多。
​	3，TensorFlow2和Pytorch实际上整体风格已经非常相似了，学会了其中一个，学习另外一个将比较容易。两种框架都掌握的话，能够参考的开源模型案例更多，并且可以方便地在两种框架之间切换。



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

##### torch

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

##### pandas

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
#DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)

#dataloader本质是一个可迭代对象，使用iter()访问，不能使用next()访问；
dataset(Dataset): 传入的数据集
shuffle在每个epoch开始的时候，对数据进行重新排序
batch_size每个batch有多少个样本
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
i = torch.tensor(1);print(i,i.dtype)
x = torch.tensor(2.0);print(x,x.dtype)
b = torch.tensor(True);print(b,b.dtype)

# 指定数据类型dtype
i = torch.tensor(1,dtype = torch.int32);print(i,i.dtype)
x = torch.tensor(2.0,dtype = torch.double);print(x,x.dtype)

# 使用特定类型构造函数IntTensor BoolTensor Tensor
i = torch.IntTensor(1);print(i,i.dtype)
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
print(scalar.size())
print(scalar.shape)
print("------------------------------------------------------------")
vector = torch.tensor([1.0,2.0,3.0,4.0])
print(vector.size())
print(vector.shape)
print("------------------------------------------------------------")
matrix = torch.tensor([[1.0,2.0],[3.0,4.0]])
print(matrix.size())
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
```



```python
matrix26 = torch.arange(0,12).view(2,6)
print(matrix26)
print(matrix26.shape)

# 转置操作让张量存储结构扭曲
matrix62 = matrix26.t()
print(matrix62)
print(matrix62.is_contiguous())


# 直接使用view方法会失败，可以使用reshape方法
matrix34 = matrix62.view(3,4)    #error!!!!!!!!!!!!!
matrix34 = matrix62.reshape(3,4) #等价于matrix34 = matrix62.contiguous().view(3,4)
print(matrix34)
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
print(arr)          #[0,0,0]
print(tensor)		#[0,0,0]

print("\nafter add 1:")
np.add(arr,1, out = arr) #给 arr增加1，tensor也随之改变
print(arr)			#[1,1,1]
print(tensor)		#[1,1,1]
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
```



```python
x1 = torch.tensor(1.0,requires_grad = True) # x需要被求导
x2 = torch.tensor(2.0,requires_grad = True)

y1 = x1*x2
y2 = x1+x2


# 允许同时对多个自变量求导数
(dy1_dx1,dy1_dx2) = torch.autograd.grad(outputs=y1,inputs = [x1,x2],retain_graph = True)
print(dy1_dx1,dy1_dx2)

# 如果有多个因变量，相当于把多个因变量的梯度结果求和
(dy12_dx1,dy12_dx2) = torch.autograd.grad(outputs=[y1,y2],inputs = [x1,x2])
print(dy12_dx1,dy12_dx2)
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
Y_hat = X@w.t() + b  # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
loss = torch.mean(torch.pow(Y_hat-Y,2))

print(loss.data)
print(Y_hat.data)
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
```



---

#### 2.3.3. 计算图与反向传播



---

#### 2.3.4. 叶子节点和非叶子节点



---

#### 2.3.5. 计算图在TensorBoard中的可视化





------

## 3. Pytorch层次结构

​	Pytorch中5个不同的层次结构：即**硬件层，内核层，低阶API，中阶API，高阶API**【torchkeras】。并以线性回归和DNN二分类模型为例，直观对比展示在不同层级实现模型的特点。

​	Pytorch的层次结构从低到高可以分成如下五层。

​		第一层为硬件层，Pytorch支持**CPU**、**GPU**加入计算资源池。

​		第二层为**C++**实现的**内核**。

​		第三层为**Python**实现的**操作符**，提供了**封装C++内核的低级API**指令，主要包括各种**张量操作算子、自动微分、变量管理**。如**torch.tensor，torch.cat，torch.autograd.grad，nn.Module**。如果把模型比作一个房子，那么第三层API就是【模型之砖】。

​		第四层为**Python**实现的**模型组件**，对低级API进行了函数封装，主要包括各种**模型层，损失函数，优化器，数据管道**等等。如**torch.nn.Linear，torch.nn.BCE，torch.optim.Adam，torch.utils.data.DataLoader**。如果把模型比作一个房子，那么第四层API就是【模型之墙】。

​		第五层为**Python**实现的**模型接口**。Pytorch没有官方的高阶API。为了便于训练模型，作者仿照keras中的模型接口，使用了不到300行代码，封装了Pytorch的高阶模型接口**torchkeras.Model**。如果把模型比作一个房子，那么第五层API就是模型本身，即【模型之屋】。



------

## 4. Pytorch的低阶API

​	Pytorch的低阶API主要包括**张量操作，动态计算图和自动微分**。

​	在低阶API层次上，可以把Pytorch当做一个增强版的numpy来使用。Pytorch提供的方法比numpy更全面，运算速度更快，如果需要的话，还可以使用GPU进行加速。

​	前面几章我们对低阶API已经有了一个整体的认识，本章我们将重点详细介绍张量操作和动态计算图。

​	张量的操作主要包括张量的**结构操作**和张量的**数学运算**。

- 张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。
- 张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。

​	动态计算图我们将主要介绍动态计算图的特性，计算**图中的Function**，计算**图与反向传播**。



------

## 5. Pytorch的中阶API

我们将主要介绍Pytorch的如下中阶API

- 数据管道
- 模型层
- 损失函数
- TensorBoard可视化



------

## 6. Pytorch的高阶API

Pytorch没有官方的高阶API。一般通过nn.Module来构建模型并编写自定义训练循环。

为了更加方便地训练模型，作者编写了仿keras的Pytorch模型接口：torchkeras， 作为Pytorch的高阶API。

本章我们主要详细介绍Pytorch的高阶API如下相关的内容。

- 构建模型的3种方法（继承nn.Module基类，使用nn.Sequential，辅助应用模型容器）
- 训练模型的3种方法（脚本风格，函数风格，torchkeras.Model类风格）
- 使用GPU训练模型（单GPU训练，多GPU训练）

