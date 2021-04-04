# Debug

### 1. bool value of Tensor with more than one value is ambiguous

函数或者可调用对象使用时候没有加括号。



### 2. 注意：关于减少时间消耗

(1)只要是用到for循环都是在cpu上进行的，会消耗巨量的时间

(2)只要是用到生成矩阵这种操作都是在cpu上进行的，会很消耗时间。

(3)数据往cuda()上搬运会比较消耗时间，也就是说 .cuda() 会比较消耗时间，能去掉就去掉。

(4)在服务器上，如果可以在一块 gpu 上运行就不要采用 `net = nn.DataParallel(net)`，这种 gpu 并行方式比单个 gpu 要耗时。



### 3. DataLoader worker (pid 7413) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.

出现这个错误的情况是，在服务器上的docker中运行训练代码时，batch size设置得过大，shared memory不够（因为docker限制了shm）.解决方法是，将Dataloader的num_workers设置为0.



