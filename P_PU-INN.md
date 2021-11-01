# PU-INN





## Metric

```shell
git clone https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
git clone https://github.com/daerduoCarey/PyTorchEMD.git
```









## evaluate

### **Point Cloud Utils** (pcu)

- A Python library for common tasks on 3D point clouds. 

依赖: **numpyeigen、Embree、manifold**

```shell
git clone https://github.com/fwilliams/point-cloud-utils.git

python setup.py install
```





### Manifold

将任何三角形网格转换为 Watertight 流行





### Embree

光线跟踪内核库。高性能光线跟踪内核3.13.1 %英特尔公司



### **numpyeigen**

```shell
git clone https://github.com/fwilliams/numpyeigen.git

mkdir build
cd build
cmake ..
make 
make install
```



**numpyEigen** - Fast zero-overhead bindings between NumPy and Eigen

NumpyEigen可以轻松地将NumPy密集数组和SciPy稀疏矩阵透明地转换为Eigen，同时零复制开销，同时利用Eigen的表达式模板系统获得最大性能。

Eigen 是一个C++数值线性代数库。它使用表达式模板为给定的一组输入类型选择最快的数字算法。NumPy和SciPy是在Python中公开快速数值例程的库。 

由于Python中的类型信息只在运行时可用，所以要编写接受多个NumPy或SciPy类型、零复制开销并且可以利用Eigen中最快的数字内核的绑定并不容易。NumpyEigen透明地生成绑定，完成上述所有工作，在编译时向C++代码公开numpy类型信息。



NumpyEigen uses [pybind11](https://github.com/pybind/pybind11) under the hood which is included automatically by the cmake project.

**依赖 pybind11**





### **pybind 11**

```shell
git clone -b numpy-hacks https://github.com/fwilliams/pybind11.git
python setup.py install
```



c++ 11和Python之间的无缝可操作性

pybind11是一个轻量级的仅头库，它在Python中公开C++类型，反之亦然，主要用于创建现有C++代码的Python绑定。

它的目标和语法类似于优秀的Boost。大卫·亚伯拉罕斯的Python库:通过使用编译时内省推断类型信息来最小化传统扩展模块中的样板代码。 



**Boost.Python**

一个C++库，支持C++和Python编程语言之间的无缝互操作性。







---

## train

3 7 8 56    0.25附近   resflow.forward + interp + resflow.backward

67 0.4附近 resflow.forward





## validation_step()

```python
fowward
	compute_loss
    	network.forward
        	_logdetgrad
            	poisson
                	lamb=self.lamb.item()
                not self.exact_trace
                	basic_logdet_estimator
```





## 每次迭代显存增加2.2Mb

```python
96 * (8, 3, 256) float32

8*3*256 = 6144

6144*96 = 589 824

96 = 8 * 12
96 = 16 * 6

96 = 3 * 32 
   = 3 * 2 * 16 
   = 3 * 2 * 2 * 8
   = 3 * 2 * 2 * 2 * 4
   = 3 * 2 * 2 * 2 * 2 * 2
```





## Profiler 结果分析



```python
        with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
            outputs = self(xyz_sparse, upratio)
        print(prof.table())
```





```python
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            aten::transpose         0.00%      52.270us         0.00%      68.161us      68.161us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%      15.891us         0.00%      15.891us      15.891us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::zeros         0.00%      65.275us         0.00%     143.925us     143.925us      18.250us         0.00%      79.250us      79.250us             1  
                                aten::empty         0.00%      14.920us         0.00%      14.920us      14.920us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::zero_         0.00%      36.012us         0.00%      63.730us      63.730us      35.250us         0.00%      61.000us      61.000us             1  
                                aten::fill_         0.00%      27.718us         0.00%      27.718us      27.718us      25.750us         0.00%      25.750us      25.750us             1  
                                   aten::to         0.00%      42.735us         0.01%     424.923us     424.923us     374.500us         0.00%     425.500us     425.500us             1  
                        aten::empty_strided         0.00%      51.490us         0.00%      51.490us      51.490us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%     330.698us         0.00%     330.698us     330.698us      51.000us         0.00%      51.000us      51.000us             1  
                                aten::zeros         0.00%      28.294us         0.00%      70.929us      70.929us      37.000us         0.00%      69.750us      69.750us             1  
                                aten::empty         0.00%       9.133us         0.00%       9.133us       9.133us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::zero_         0.00%      20.113us         0.00%      33.502us      33.502us      19.250us         0.00%      32.750us      32.750us             1  
                                aten::fill_         0.00%      13.389us         0.00%      13.389us      13.389us      13.500us         0.00%      13.500us      13.500us             1  
                                   aten::to         0.00%      34.819us         0.00%     174.149us     174.149us      62.500us         0.00%     175.500us     175.500us             1  
                        aten::empty_strided         0.00%      27.730us         0.00%      27.730us      27.730us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%     111.600us         0.00%     111.600us     111.600us     113.000us         0.00%     113.000us     113.000us             1  
                                aten::zeros         0.00%      26.785us         0.00%      67.128us      67.128us      35.000us         0.00%      66.750us      66.750us             1  
                                aten::empty         0.00%       8.236us         0.00%       8.236us       8.236us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::zero_         0.00%      19.287us         0.00%      32.107us      32.107us      19.250us         0.00%      31.750us      31.750us             1  
                                aten::fill_         0.00%      12.820us         0.00%      12.820us      12.820us      12.500us         0.00%      12.500us      12.500us             1  
                                   aten::to         0.00%      27.560us         0.00%     119.552us     119.552us      51.250us         0.00%     117.750us     117.750us             1  
                        aten::empty_strided         0.00%      23.261us         0.00%      23.261us      23.261us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      68.731us         0.00%      68.731us      68.731us      66.500us         0.00%      66.500us      66.500us             1  
                                aten::zeros         0.00%      40.482us         0.00%      95.209us      95.209us      47.000us         0.00%      91.250us      91.250us             1  
                                aten::empty         0.00%       9.517us         0.00%       9.517us       9.517us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::zero_         0.00%      20.275us         0.00%      45.210us      45.210us      19.250us         0.00%      44.250us      44.250us             1  
                                aten::fill_         0.00%      24.935us         0.00%      24.935us      24.935us      25.000us         0.00%      25.000us      25.000us             1  
                                   aten::to         0.00%      27.212us         0.00%     119.590us     119.590us      51.250us         0.00%     119.750us     119.750us             1  
                        aten::empty_strided         0.00%      22.941us         0.00%      22.941us      22.941us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      69.437us         0.00%      69.437us      69.437us      68.500us         0.00%      68.500us      68.500us             1  
                                 aten::view         0.00%      25.287us         0.00%      25.287us      25.287us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::is_nonzero         0.00%      38.531us         0.00%     137.551us     137.551us       6.500us         0.00%      79.500us      79.500us             1  
                                 aten::item         0.00%      20.594us         0.00%      99.020us      99.020us       6.500us         0.00%      73.000us      73.000us             1  
                  aten::_local_scalar_dense         0.00%      78.426us         0.00%      78.426us      78.426us      66.500us         0.00%      66.500us      66.500us             1  
                            aten::transpose         0.00%      17.842us         0.00%      24.244us      24.244us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%       6.402us         0.00%       6.402us       6.402us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::contiguous         0.00%      27.235us         0.00%     168.694us     168.694us      53.000us         0.00%     164.500us     164.500us             1  
                           aten::empty_like         0.00%       9.497us         0.00%      29.071us      29.071us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::empty         0.00%      19.574us         0.00%      19.574us      19.574us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%     112.388us         0.00%     112.388us     112.388us     111.500us         0.00%     111.500us     111.500us             1  
                                 aten::view         0.00%      11.307us         0.00%      11.307us      11.307us       0.000us         0.00%       0.000us       0.000us             1  
                                 aten::mean         0.00%     141.131us         0.00%     168.363us     168.363us     356.500us         0.00%     356.500us     356.500us             1  
                                aten::empty         0.00%      16.793us         0.00%      16.793us      16.793us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%      10.439us         0.00%      10.439us      10.439us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::var         0.00%      81.389us         0.00%     108.387us     108.387us      18.500us         0.00%      18.500us      18.500us             1  
                                aten::empty         0.00%       4.695us         0.00%       4.695us       4.695us       0.000us         0.00%       0.000us       0.000us             1  
                              aten::resize_         0.00%      16.301us         0.00%      16.301us      16.301us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%       6.002us         0.00%       6.002us       6.002us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::empty         0.00%       7.860us         0.00%       7.860us       7.860us       0.000us         0.00%       0.000us       0.000us             1  
                                   aten::to         0.00%       7.668us         0.00%       7.668us       7.668us       1.500us         0.00%       1.500us       1.500us             1  
                              aten::detach_         0.00%      10.701us         0.00%      17.971us      17.971us       2.500us         0.00%       3.500us       3.500us             1  
                                    detach_         0.00%       7.270us         0.00%       7.270us       7.270us       1.000us         0.00%       1.000us       1.000us             1  
                                   aten::to         0.00%      17.117us         0.00%      91.064us      91.064us       6.000us         0.00%      51.000us      51.000us             1  
                        aten::empty_strided         0.00%      13.756us         0.00%      13.756us      13.756us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      60.191us         0.00%      60.191us      60.191us      45.000us         0.00%      45.000us      45.000us             1  
                                  aten::max         0.00%      21.698us         0.00%      96.545us      96.545us      21.000us         0.00%      95.500us      95.500us             1  
                              aten::maximum         0.00%      68.552us         0.00%      74.847us      74.847us      74.500us         0.00%      74.500us      74.500us             1  
                                aten::empty         0.00%       6.295us         0.00%       6.295us       6.295us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::neg         0.00%      35.640us         0.00%      92.150us      92.150us      35.500us         0.00%      89.000us      89.000us             1  
                                aten::empty         0.00%       3.760us         0.00%       3.760us       3.760us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::neg         0.00%      46.779us         0.00%      52.750us      52.750us      53.500us         0.00%      53.500us      53.500us             1  
                              aten::resize_         0.00%       5.971us         0.00%       5.971us       5.971us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      45.776us         0.00%      45.776us      45.776us      45.000us         0.00%      45.000us      45.000us             1  
                                  aten::log         0.00%      36.556us         0.00%      80.103us      80.103us      38.500us         0.00%      79.500us      79.500us             1  
                                aten::empty         0.00%       3.610us         0.00%       3.610us       3.610us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::log         0.00%      34.864us         0.00%      39.937us      39.937us      41.000us         0.00%      41.000us      41.000us             1  
                              aten::resize_         0.00%       5.073us         0.00%       5.073us       5.073us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::mul         0.00%      63.171us         0.00%      69.545us      69.545us      68.500us         0.00%      68.500us      68.500us             1  
                                aten::empty         0.00%       6.374us         0.00%       6.374us       6.374us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      41.357us         0.00%      41.357us      41.357us      41.000us         0.00%      41.000us      41.000us             1  
                                aten::fill_         0.00%      47.567us         0.00%      47.567us      47.567us      47.500us         0.00%      47.500us      47.500us             1  
                                 aten::view         0.00%      25.261us         0.00%      25.261us      25.261us       0.000us         0.00%       0.000us       0.000us             1  
                            aten::expand_as         0.00%      12.411us         0.00%      31.113us      31.113us      30.000us         0.00%      30.000us      30.000us             1  
                               aten::expand         0.00%      15.444us         0.00%      18.702us      18.702us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%       3.258us         0.00%       3.258us       3.258us       0.000us         0.00%       0.000us       0.000us             1  
                                 aten::view         0.00%      12.328us         0.00%      12.328us      12.328us       0.000us         0.00%       0.000us       0.000us             1  
                            aten::expand_as         0.00%       9.872us         0.00%      29.046us      29.046us      28.000us         0.00%      28.000us      28.000us             1  
                               aten::expand         0.00%      17.097us         0.00%      19.174us      19.174us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%       2.077us         0.00%       2.077us       2.077us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::add         0.00%      70.829us         0.00%      80.952us      80.952us      78.500us         0.00%      78.500us      78.500us             1  
                        aten::empty_strided         0.00%      10.123us         0.00%      10.123us      10.123us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::exp         0.00%      46.743us         0.00%     103.599us     103.599us      43.500us         0.00%     103.000us     103.000us             1  
                                aten::empty         0.00%       3.347us         0.00%       3.347us       3.347us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::exp         0.00%      47.609us         0.00%      53.509us      53.509us      59.500us         0.00%      59.500us      59.500us             1  
                              aten::resize_         0.00%       5.900us         0.00%       5.900us       5.900us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::mul         0.00%      82.692us         0.00%      91.554us      91.554us      92.000us         0.00%      92.000us      92.000us             1  
                        aten::empty_strided         0.00%       8.862us         0.00%       8.862us       8.862us       0.000us         0.00%       0.000us       0.000us             1  
                                 aten::view         0.00%      15.224us         0.00%      15.224us      15.224us       0.000us         0.00%       0.000us       0.000us             1  
                               aten::expand         0.00%      12.741us         0.00%      15.367us      15.367us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::as_strided         0.00%       2.626us         0.00%       2.626us       2.626us       0.000us         0.00%       0.000us       0.000us             1  
                           aten::contiguous         0.00%      23.662us         0.00%      88.583us      88.583us      37.500us         0.00%      86.000us      86.000us             1  
                           aten::empty_like         0.00%       6.043us         0.00%      16.424us      16.424us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::empty         0.00%      10.381us         0.00%      10.381us      10.381us       0.000us         0.00%       0.000us       0.000us             1  
                                aten::copy_         0.00%      48.497us         0.00%      48.497us      48.497us      48.500us         0.00%      48.500us      48.500us             1  
                                 aten::view         0.00%      12.953us         0.00%      12.953us      12.953us       0.000us         0.00%       0.000us       0.000us             1  
                                  aten::sum         0.00%      86.695us         0.00%      93.685us      93.685us      91.500us         0.00%      91.500us      91.500us             1  
                                aten::empty         0.00%       6.990us         0.00%       6.990us       6.990us       0.000us         0.00%       0.000us       0.000us             1  
                                 aten::rsub         0.00%      79.015us         0.00%      85.947us      85.947us       8.500us         0.00%       8.500us       8.500us             1  
                                aten::empty         0.00%       6.932us         0.00%       6.932us       6.932us       0.000us         0.00%       0.000us       0.000us             1  
                    aten::is_floating_point         0.00%       7.082us         0.00%       7.082us       7.082us       1.500us         0.00%       1.500us       1.500us             1  
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 7.867s
CUDA time total: 7.911s
```







```PYTHON
251.7 	648.4 	1045.2	1441.8
		396.7	396.8	396.6
    
251.7	660.8	1057.5	1466.4
		409.1	396.7	408.9

251.7	685.4	1082.1	1491.0
		433.7	396.7	408.9
    
251.7	685.4	1094.4	1540.2
		433.7	409		445.8
    
251.7	710.0	1131.3	1589.4
		458.3	421.3	458.1

```

