# PU-INN Code



# Experiment

## Ablation Study

### Interpolation

三个都比较，InterpolationModule结果最好，并且能收敛，

conv_1发散结果最差，

linear_1 收敛结果没有 InterpolationModule 好，差两个数量级

pugan 发散

```python
self.interp  = InterpolationModule(pc_channel=3, k=8)

self.interp_conv_1 = nn.Sequential(
nn.Conv1d(256, 1024, 1, bias=True),#kernel_size=1
nn.BatchNorm1d(1024),
nn.ReLU(inplace=True))

self.interp_conv_2 = nn.Sequential(
nn.Conv1d(3, 12, 1, bias=True),#kernel_size=1
nn.BatchNorm1d(12),
nn.ReLU(inplace=True))

self.interp_linear_1 = nn.Sequential(
nn.Linear(256, 1024),#kernel_size=1
nn.ReLU(inplace=True))

self.interp_pugan_updownup = up_projection_unit()
```



### ConditionalInjector

1. 加到每个stackediresblocks的第一层



2. 第一个stackediresblocks









## 网络架构

```python
PointResidualFlow(
  (transforms): ModuleList(
    (0): StackediResBlocks(
      (chain): ModuleList(
        (0): ActNormPC(3)
        (1): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (1): Swish()
            (2): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (3): Swish()
            (4): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (2): ActNormPC(3)
        (3): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (4): ActNormPC(3)
        (5): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (6): ActNormPC(3)
        (7): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (8): ActNormPC(3)
        (9): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (10): ActNormPC(3)
        (11): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (12): ActNormPC(3)
        (13): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (14): ActNormPC(3)
        (15): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (16): ActNormPC(3)
        (17): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (18): ActNormPC(3)
        (19): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (20): ActNormPC(3)
        (21): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (22): ActNormPC(3)
        (23): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (24): ActNormPC(3)
        (25): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (26): ActNormPC(3)
        (27): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (28): ActNormPC(3)
        (29): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (30): ActNormPC(3)
        (31): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (32): ActNormPC(3)
      )
    )
    (1): StackediResBlocks(
      (chain): ModuleList(
        (0): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (1): ActNormPC(3)
        (2): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (3): ActNormPC(3)
        (4): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (5): ActNormPC(3)
        (6): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (7): ActNormPC(3)
        (8): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (9): ActNormPC(3)
        (10): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (11): ActNormPC(3)
        (12): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (13): ActNormPC(3)
        (14): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (15): ActNormPC(3)
        (16): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (17): ActNormPC(3)
        (18): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (19): ActNormPC(3)
        (20): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (21): ActNormPC(3)
        (22): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (23): ActNormPC(3)
        (24): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (25): ActNormPC(3)
        (26): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (27): ActNormPC(3)
        (28): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (29): ActNormPC(3)
        (30): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (31): ActNormPC(3)
      )
    )
    (2): StackediResBlocks(
      (chain): ModuleList(
        (0): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (1): ActNormPC(3)
        (2): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (3): ActNormPC(3)
        (4): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (5): ActNormPC(3)
        (6): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (7): ActNormPC(3)
        (8): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (9): ActNormPC(3)
        (10): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (11): ActNormPC(3)
        (12): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (13): ActNormPC(3)
        (14): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (15): ActNormPC(3)
        (16): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (17): ActNormPC(3)
        (18): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (19): ActNormPC(3)
        (20): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (21): ActNormPC(3)
        (22): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (23): ActNormPC(3)
        (24): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (25): ActNormPC(3)
        (26): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (27): ActNormPC(3)
        (28): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (29): ActNormPC(3)
        (30): iResBlock(
          dist=poisson, n_samples=1, n_power_series=None, neumann_grad=True, exact_trace=False, brute_force=False
          (nnet): Sequential(
            (0): Swish()
            (1): InducedNormConv1d(3, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (2): Swish()
            (3): InducedNormConv1d(64, 64, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
            (4): Swish()
            (5): InducedNormConv1d(64, 3, kernel_size=1, stride=1, coeff=0.98, domain=2.00, codomain=2.00, n_iters=None, atol=0.001, rtol=0.001, learnable_ord=False)
          )
        )
        (31): ActNormPC(3)
      )
    )
  )
```

