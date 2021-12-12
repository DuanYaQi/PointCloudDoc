# CUDA 4 DL

**setup.py**

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointops',
    ext_modules=[
        CUDAExtension('pointops_cuda', [
            'src/pointops_api.cpp',

            'src/ballquery/ballquery_cuda.cpp',
            'src/ballquery/ballquery_cuda_kernel.cu',
            'src/knnquery/knnquery_cuda.cpp',
            'src/knnquery/knnquery_cuda_kernel.cu',
            'src/grouping/grouping_cuda.cpp',
            'src/grouping/grouping_cuda_kernel.cu',
            'src/grouping_int/grouping_int_cuda.cpp',
            'src/grouping_int/grouping_int_cuda_kernel.cu',
            'src/interpolation/interpolation_cuda.cpp',
            'src/interpolation/interpolation_cuda_kernel.cu',
            'src/sampling/sampling_cuda.cpp',
            'src/sampling/sampling_cuda_kernel.cu',

            'src/labelstat/labelstat_cuda.cpp',
            'src/labelstat/labelstat_cuda_kernel.cu',

            'src/featuredistribute/featuredistribute_cuda.cpp',
            'src/featuredistribute/featuredistribute_cuda_kernel.cu'
        ],
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
```





**cuda_utils.h**

```h
#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cmath>

#define TOTAL_THREADS 1024

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


#endif
```

- 9.0 AT_CHECK > 10.0 TORCH_CHECK



**function**

```c++
cudaStream_t stream = THCState_getCurrentStream(state); //9.0
cudaStream_t stream = c10::cuda::getCurrentCUDAStream();//10.0
```