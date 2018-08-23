[![Build Status](https://travis-ci.org/dgasmith/opt_einsum.svg?branch=master)](https://travis-ci.org/dgasmith/opt_einsum) 
[![codecov](https://codecov.io/gh/dgasmith/opt_einsum/branch/master/graph/badge.svg)](https://codecov.io/gh/dgasmith/opt_einsum)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/opt_einsum/badges/version.svg)](https://anaconda.org/conda-forge/opt_einsum)
[![PyPI](https://img.shields.io/pypi/v/opt_einsum.svg)](https://pypi.org/project/opt-einsum/#description)
[![Documentation Status](https://readthedocs.org/projects/optimized-einsum/badge/?version=latest)](http://optimized-einsum.readthedocs.io/en/latest/?badge=latest)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00753/status.svg)](https://doi.org/10.21105/joss.00753)


##### News: Opt_einsum will be in NumPy 1.12 and BLAS features in NumPy 1.14! Call opt_einsum as `np.einsum(..., optimize=True)`. This repository contains more advanced features such as Dask or Tensorflow backends as well as a testing ground for newer features in this ecosystem. 

Optimized Einsum: A tensor contraction order optimizer
======================================================

Optimized einsum can greatly reduce the overall time [`np.einsum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html) takes by optimizing the expression's contraction order and dispatching many operations to canonical BLAS routines. See the [**documentation**](http://optimized-einsum.readthedocs.io) for more information.

As well as [`opt_einsum.contract`](https://optimized-einsum.readthedocs.io/en/latest/autosummary/opt_einsum.contract.html#opt-einsum-contract) acting as a drop-in replacement for `np.einsum`, the following capabilities are enabled by `opt_einsum`:

* Inspect [detailed information](http://optimized-einsum.readthedocs.io/en/latest/path_finding.html) about the path chosen.
* Perform contractions with [various backends](http://optimized-einsum.readthedocs.io/en/latest/backends.html), including on the GPU and with libraries such as [TensorFlow](https://www.tensorflow.org) and [PyTorch](https://pytorch.org).
* Generate [reusable expressions](http://optimized-einsum.readthedocs.io/en/latest/reusing_paths.html), potentially with [constant tensors](http://optimized-einsum.readthedocs.io/en/latest/reusing_paths.html#specifying-constants), that can be compiled for greater performance.
* Use an arbitrary number of indices to find contractions for [hundreds or even thousands of tensors](http://optimized-einsum.readthedocs.io/en/latest/ex_large_expr_with_greedy.html).
* Share [intermediate computations](http://optimized-einsum.readthedocs.io/en/latest/sharing_intermediates.html) among multiple contractions.

## Quick tutorial
Einsum is a powerful function for contracting tensors of arbitrary dimension and index.
However, it is only optimized to contract two terms at a time resulting in non-optimal scaling.

For example, let us examine the following index transformation:
`M_{pqrs} = C_{pi} C_{qj} I_{ijkl} C_{rk} C_{sl}`

We can then develop two seperate implementations that produce the same result:
```python
N = 10
C = np.random.rand(N, N)
I = np.random.rand(N, N, N, N)

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

The `np.einsum` function does not consider building intermediate arrays; therefore, helping einsum out by building these intermediate arrays can result in a considerable cost saving even for small N (N=10):

```python
np.allclose(naive(I, C), optimized(I, C))
True

%timeit naive(I, C)
1 loops, best of 3: 934 ms per loop

%timeit optimized(I, C)
1000 loops, best of 3: 527 us per loop
```

A 2000 fold speed up for 4 extra lines of code!
This contraction can be further complicated by considering that the shape of the C matrices need not be the same, in this case, the ordering in which the indices are transformed matters significantly.
Logic can be built that optimizes the ordering; however, this is a lot of time and effort for a single expression.

The opt_einsum package is a drop-in replacement for the np.einsum function and can handle all of this logic for you:

```python
from opt_einsum import contract

%timeit contract('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
1000 loops, best of 3: 324 us per loop
```

The above will automatically find the optimal contraction order, in this case, identical to that of the optimized function above, and compute the products for you. In this case, it even uses `np.dot` under the hood to exploit any vendor BLAS functionality that your NumPy build has!


Please see the [documentation](http://optimized-einsum.readthedocs.io/en/latest/?badge=latest) for more features!


## Installation

`opt_einsum` can either be installed via `pip install opt_einsum` or from conda `conda install opt_einsum -c conda-forge`. See the installation [documenation](http://optimized-einsum.readthedocs.io/en/latest/install.html) for further methods.

## Citation

If this code has benefited your research, please support us by citing:

Daniel G. A. Smith and Johnnie Gray, opt_einsum - A Python package for optimizing contraction order for einsum-like expressions. *Journal of Open Source Software*, **2018**, 3(26), 753

DOI: https://doi.org/10.21105/joss.00753


## Contributing

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the [contributing guide](https://github.com/dgasmith/opt_einsum/blob/master/.github/CONTRIBUTING.md).


