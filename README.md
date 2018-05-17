[![Build Status](https://travis-ci.org/dgasmith/opt_einsum.svg?branch=master)](https://travis-ci.org/dgasmith/opt_einsum) 
[![codecov](https://codecov.io/gh/dgasmith/opt_einsum/branch/master/graph/badge.svg)](https://codecov.io/gh/dgasmith/opt_einsum)
[![DOI](https://zenodo.org/badge/27930623.svg)](https://zenodo.org/badge/latestdoi/27930623)
[![Conda](https://anaconda.org/conda-forge/opt_einsum/badges/version.svg)](https://anaconda.org/conda-forge/opt_einsum)
[![PyPI](https://img.shields.io/pypi/v/opt_einsum.svg)](https://pypi.python.org/pypi/opt-einsum/1.0.1)
[![Documentation Status](https://readthedocs.org/projects/optimized-einsum/badge/?version=latest)](http://optimized-einsum.readthedocs.io/en/latest/?badge=latest)


##### News: Opt_einsum will be in NumPy 1.12 and BLAS features in NumPy 1.14! Call opt_einsum as `np.einsum(..., optimize=True)`. This repostiory contains more advanced features such as Dask or Tensorflow backends as well as continued testing ground for newer features in this ecosystem. 

Optimized Einsum: A tensor contraction order optimizer
======================================================

Optimized einsum can greatly reduce the overall time `np.einsum` takes by optimizing the expressions contraction order and dispatching many operations to canonical BLAS routines. See the [documention](http://optimized-einsum.readthedocs.io) for more information.

 - [Optimizing numpy's einsum function](https://github.com/dgasmith/opt_einsum/blob/master/README.md#optimizing-numpys-einsum-function)
 - [Obtaining the path expression](https://github.com/dgasmith/opt_einsum/blob/master/README.md#obtaining-the-path-expression)
 - [Reusing paths](https://github.com/dgasmith/opt_einsum/blob/master/README.md#reusing-paths-using-contract_expression)
 - [Installation](https://github.com/dgasmith/opt_einsum/blob/master/README.md#installation)

## Optimizing numpy's einsum function
Einsum is a very powerful function for contracting tensors of arbitrary dimension and index.
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

The `np.einsum` function does not consider building intermediate arrays; therefore, helping einsum out by building these intermediate arrays can result in a considerable cost savings even for small N (N=10):

```python
np.allclose(naive(I, C), optimized(I, C))
True

%timeit naive(I, C)
1 loops, best of 3: 934 ms per loop

%timeit optimized(I, C)
1000 loops, best of 3: 527 µs per loop
```

A 2000 fold speed up for 4 extra lines of code!
This contraction can be further complicated by considering that the shape of the C matrices need not be the same, in this case the ordering in which the indices are transformed matters greatly.
Logic can be built that optimizes the ordering; however, this is a lot of time and effort for a single expression.

The opt_einsum package is a drop in replacement for the np.einsum function and can handle all of this logic for you:

```python
from opt_einsum import contract

%timeit contract('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
1000 loops, best of 3: 324 µs per loop
```

The above will automatically find the optimal contraction order, in this case identical to that of the optimized function above, and compute the products for you. In this case, it even uses `np.dot` under the hood to exploit any vendor BLAS functionality that your NumPy build has!


Please see the [documentation](http://optimized-einsum.readthedocs.io/en/latest/?badge=latest) for more features!


## Installation

`opt_einsum` can either be installed via `pip install opt_einsum` or from conda `conda install opt_einsum -c conda-forge`. See the installation [documenation](http://optimized-einsum.readthedocs.io/en/latest/install.html) for further methods.
