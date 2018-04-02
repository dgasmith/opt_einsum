[![Build Status](https://travis-ci.org/dgasmith/opt_einsum.svg?branch=master)](https://travis-ci.org/dgasmith/opt_einsum) 
[![codecov](https://codecov.io/gh/dgasmith/opt_einsum/branch/master/graph/badge.svg)](https://codecov.io/gh/dgasmith/opt_einsum)
[![DOI](https://zenodo.org/badge/27930623.svg)](https://zenodo.org/badge/latestdoi/27930623)
[![Conda](https://anaconda.org/conda-forge/opt_einsum/badges/version.svg)](https://anaconda.org/conda-forge/opt_einsum)
[![PyPI](https://img.shields.io/pypi/v/opt_einsum.svg)](https://pypi.python.org/pypi/opt-einsum/1.0.1)
[![Documentation Status](https://readthedocs.org/projects/optimized-einsum/badge/?version=latest)](http://optimized-einsum.readthedocs.io/en/latest/?badge=latest)


##### News: Opt_einsum will be in NumPy 1.12 and BLAS features in NumPy 1.14! Call opt_einsum as `np.einsum(..., optimize=True)`. This repostiory will continue to provide a testing ground for new features. 

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

## Obtaining the path expression

Now, lets consider the following expression found in a perturbation theory (one of ~5,000 such expressions):
`bdik,acaj,ikab,ajac,ikbd`

At first, it would appear that this scales like N^7 as there are 7 unique indices; however, we can define a intermediate to reduce this scaling.

`a = bdik,ikab,ikbd` (N^5 scaling)

`result = acaj,ajac,a` (N^4 scaling)

This is a single possible path to the final answer (and notably, not the most optimal) out of many possible paths. Now, let opt_einsum compute the optimal path:

```python
import opt_einsum as oe

# Take a complex string
einsum_string = 'bdik,acaj,ikab,ajac,ikbd->'

# Build random views to represent this contraction
unique_inds = set(einsum_string.replace(',', ''))
index_size = [10, 17, 9, 10, 13, 16, 15, 14]
sizes_dict = {c : s for c, s in zip(set(einsum_string), index_size)}
views = oe.helpers.build_views(einsum_string, sizes_dict)

path_info = oe.contract_path(einsum_string, *views)
>>> print path_info[0]
[(1, 3), (0, 2), (0, 2), (0, 1)]

```
```
>>> print path_info[1]
  Complete contraction:  bdik,acaj,ikab,ajac,ikbd->
         Naive scaling:  7
     Optimized scaling:  4
      Naive FLOP count:  3.819e+08
  Optimized FLOP count:  8.000e+04
   Theoretical speedup:  4773.600
  Largest intermediate:  1.872e+03 elements
--------------------------------------------------------------------------------
scaling   BLAS                  current                                remaining
--------------------------------------------------------------------------------
   3     False             ajac,acaj->a                       bdik,ikab,ikbd,a->
   4     False           ikbd,bdik->bik                             ikab,a,bik->
   4     False              bik,ikab->a                                    a,a->
   1       DOT                    a,a->                                       ->
```
```python

einsum_result = np.einsum("bdik,acaj,ikab,ajac,ikbd->", *views)
contract_result = contract("bdik,acaj,ikab,ajac,ikbd->", *views)
>>> np.allclose(einsum_result, contract_result)
True
```

By contracting terms in the correct order we can see that this expression can be computed with N^4 scaling. Even with the overhead of finding the best order or 'path' and small dimensions, opt_einsum is roughly 900 times faster than pure einsum for this expression.


## Reusing paths using ``contract_expression``

If you expect to repeatedly use a particular contraction it can make things simpler and more efficient to not compute the path each time. Instead, supplying ``contract_expression`` with the contraction string and the shapes of the tensors generates a ``ContractExpression`` which can then be repeatedly called with any matching set of arrays. For example:

```python
>>> my_expr = oe.contract_expression("abc,cd,dbe->ea", (2, 3, 4), (4, 5), (5, 3, 6))
>>> print(my_expr)
<ContractExpression> for 'abc,cd,dbe->ea':
  1.  'dbe,cd->bce' [GEMM]
  2.  'bce,abc->ea' [GEMM]
```

Now we can call this expression with 3 arrays that match the original shapes without having to compute the path again:

```python
>>> x, y, z = (np.random.rand(*s) for s in [(2, 3, 4), (4, 5), (5, 3, 6)])
>>> my_expr(x, y, z)
array([[ 3.08331541,  4.13708916],
       [ 2.92793729,  4.57945185],
       [ 3.55679457,  5.56304115],
       [ 2.6208398 ,  4.39024187],
       [ 3.66736543,  5.41450334],
       [ 3.67772272,  5.46727192]])
```

Note that few checks are performed when calling the expression, and while it will work for a set of arrays with the same ranks as the original shapes but differing sizes, it might no longer be optimal.


## Installation

Thanks to [Nils Werner](https://github.com/nils-werner) `opt_einsum` can be installed with the line `pip install -e .[tests]`.
Test cases can then be run with `py.test -v`.

We are also now on PyPi: `pip install opt_einsum`
