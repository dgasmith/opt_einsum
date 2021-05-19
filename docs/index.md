# Overview

Optimized einsum can significantly reduce the overall execution time of einsum-like
expressions by optimizing the expression's contraction order and dispatching
many operations to canonical BLAS, cuBLAS, or other specialized routines.
Optimized einsum is agnostic to the backend and can handle NumPy, Dask,
PyTorch, Tensorflow, CuPy, Sparse, Theano, JAX, and Autograd arrays as well as
potentially any library which conforms to a standard API.

## Features

The algorithms found in this repository often power the `einsum` optimizations
in many of the above projects. For example, the optimization of `np.einsum`
has been passed upstream and most of the same features that can be found in
this repository can be enabled with `numpy.einsum(..., optimize=True)`. However,
this repository often has more up to date algorithms for complex contractions.
Several advanced features are as follows:

* Inspect [detailed information](paths/introduction.md) about the path chosen.
* Perform contractions with [numerous backends](getting_started/backends.md), including on the GPU and with libraries such as [TensorFlow](https://www.tensorflow.org) and [PyTorch](https://pytorch.org).
* Generate [reusable expressions](getting_started/reusing_paths.md), potentially with constant tensors, that can be compiled for greater performance.
* Use an arbitrary number of indices to find contractions for [hundreds or even thousands of tensors](examples/large_expr_with_greedy.md).
* Share [intermediate computations](getting_started/sharing_intermediates.md) among multiple contractions.
* Compute gradients of tensor contractions using [Autograd](https://github.com/HIPS/autograd) or [JAX](https://github.com/google/jax).

## Example

Take the following einsum-like expression:

$$
M_{pqrs} = C_{pi} C_{qj} I_{ijkl} C_{rk} C_{sl}
$$

and consider two different algorithms:

```python
import numpy as np

dim = 10
I = np.random.rand(dim, dim, dim, dim)
C = np.random.rand(dim, dim)

def naive(I, C):
    # N^8 scaling
    return np.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)

def optimized(I, C):
    # N^5 scaling
    K = np.einsum('pi,ijkl->pjkl', C, I)
    K = np.einsum('qj,pjkl->pqkl', C, K)
    K = np.einsum('rk,pqkl->pqrl', C, K)
    K = np.einsum('sl,pqrl->pqrs', C, K)
    return K
```

```python
>>> np.allclose(naive(I, C), optimized(I, C))
True
```

Most einsum functions do not consider building intermediate arrays;
therefore, helping einsum functions by creating these intermediate arrays can result
in considerable cost savings even for small N (N=10):

```python
%timeit naive(I, C)
1 loops, best of 3: 829 ms per loop

%timeit optimized(I, C)
1000 loops, best of 3: 445 µs per loop
```

The index transformation is a well-known contraction that leads to
straightforward intermediates. This contraction can be further
complicated by considering that the shape of the C matrices need not be
the same, in this case, the ordering in which the indices are transformed
matters significantly. Logic can be built that optimizes the order;
however, this is a lot of time and effort for a single expression.

The `opt_einsum` package is a typically a drop-in replacement for `einsum`
functions and can handle this logic and path finding for you:

```python
from opt_einsum import contract

dim = 30
I = np.random.rand(dim, dim, dim, dim)
C = np.random.rand(dim, dim)

%timeit optimized(I, C)
10 loops, best of 3: 65.8 ms per loop

%timeit contract('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
100 loops, best of 3: 16.2 ms per loop
```

The above will automatically find the optimal contraction order, in this case,
identical to that of the optimized function above, and compute the products
for you. Additionally, `contract` can use vendor BLAS with the `numpy.dot`
function under the hood to exploit additional parallelism and performance.

Details about the optimized contraction order can be explored:

```python
>>> import opt_einsum as oe

>>> path_info = oe.contract_path('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)

>>> print(path_info[0])
[(0, 2), (0, 3), (0, 2), (0, 1)]

>>> print(path_info[1])
  Complete contraction:  pi,qj,ijkl,rk,sl->pqrs
         Naive scaling:  8
     Optimized scaling:  5
      Naive FLOP count:  8.000e+08
  Optimized FLOP count:  8.000e+05
   Theoretical speedup:  1000.000
  Largest intermediate:  1.000e+04 elements
--------------------------------------------------------------------------------
scaling   BLAS                  current                                remaining
--------------------------------------------------------------------------------
   5      GEMM            ijkl,pi->jklp                      qj,rk,sl,jklp->pqrs
   5      GEMM            jklp,qj->klpq                         rk,sl,klpq->pqrs
   5      GEMM            klpq,rk->lpqr                            sl,lpqr->pqrs
   5      GEMM            lpqr,sl->pqrs                               pqrs->pqrs
```



## Citation

If this code has benefited your research, please support us by citing:

Daniel G. A. Smith and Johnnie Gray, opt_einsum - A Python package for optimizing contraction order for einsum-like expressions. **Journal of Open Source Software**, *2018*, 3(26), 753

DOI: https://doi.org/10.21105/joss.00753
