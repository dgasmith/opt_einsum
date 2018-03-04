[![Build Status](https://travis-ci.org/dgasmith/opt_einsum.svg?branch=master)](https://travis-ci.org/dgasmith/opt_einsum) 
[![codecov](https://codecov.io/gh/dgasmith/opt_einsum/branch/master/graph/badge.svg)](https://codecov.io/gh/dgasmith/opt_einsum)
[![DOI](https://zenodo.org/badge/27930623.svg)](https://zenodo.org/badge/latestdoi/27930623)
[![Conda](https://anaconda.org/conda-forge/opt_einsum/badges/version.svg)](https://anaconda.org/conda-forge/opt_einsum)
[![PyPI](https://img.shields.io/pypi/v/opt_einsum.svg)](https://pypi.python.org/pypi/opt-einsum/1.0.1)
[![Documentation Status](https://readthedocs.org/projects/optimized-einsum/badge/?version=latest)](http://optimized-einsum.readthedocs.io/en/latest/?badge=latest)


##### News: Opt_einsum will be in NumPy 1.12 and BLAS features in NumPy 1.14! Call opt_einsum as `np.einsum(..., optimize=True)`. This repostiory will continue to provide a testing ground for new features. 

Optimized Einsum: A tensor contraction order optimizer
==========

 - [Optimizing numpy's einsum function](https://github.com/dgasmith/opt_einsum/blob/master/README.md#optimizing-numpys-einsum-function)
 - [Obtaining the path expression](https://github.com/dgasmith/opt_einsum/blob/master/README.md#obtaining-the-path-expression)
 - [Reusing paths](https://github.com/dgasmith/opt_einsum/blob/master/README.md#reusing-paths-using-contract_expression)
 - [More details on paths](https://github.com/dgasmith/opt_einsum/blob/master/README.md#more-details-on-paths)
 - [Finding the optimal path](https://github.com/dgasmith/opt_einsum/blob/master/README.md#finding-the-optimal-path)
 - [Finding the opportunistic path](https://github.com/dgasmith/opt_einsum/blob/master/README.md#finding-the-opportunistic-path)
 - [Testing](https://github.com/dgasmith/opt_einsum/blob/master/README.md#testing)
 - [Outstanding issues](https://github.com/dgasmith/opt_einsum/blob/master/README.md#outstanding-issues)
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

The einsum function does not consider building intermediate arrays; therefore, helping einsum out by building these intermediate arrays can result in a considerable cost savings even for small N (N=10):

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


## More details on paths

Finding the optimal order of contraction is not an easy problem and formally scales factorially with respect to the number of terms in the expression. First, lets discuss what a path looks like in opt_einsum:
```python
einsum_path = [(0, 1, 2, 3, 4)]
opt_path = [(1, 3), (0, 2), (0, 2), (0, 1)]
```
In opt_einsum each element of the list represents a single contraction.
For example the einsum_path would effectively compute the result in a way identical to that of einsum itself, while the
opt_path would perform four contractions that form an identical result.
This opt_path represents the path taken in our above example.
The first contraction (1,3) contracts the first and third terms together to produce a new term which is then appended to the list of terms, this is continued until all terms are contracted.
An example should illuminate this:

```
---------------------------------------------------------------------------------
scaling   GEMM                   current                                remaining
---------------------------------------------------------------------------------
terms = ['bdik', 'acaj', 'ikab', 'ajac', 'ikbd'] contraction = (1, 3)
  3     False              ajac,acaj->a                       bdik,ikab,ikbd,a->
terms = ['bdik', 'ikab', 'ikbd', 'a'] contraction = (0, 2)
  4     False            ikbd,bdik->bik                             ikab,a,bik->
terms = ['ikab', 'a', 'bik'] contraction = (0, 2)
  4     False              bik,ikab->a                                    a,a->
terms = ['a', 'a'] contraction = (0, 1)
  1       DOT                    a,a->                                       ->
```



## Finding the optimal path

The most optimal path can be found by searching through every possible way to contract the tensors together, this includes all combinations with the new intermediate tensors as well.
While this algorithm scales like N! and can often become more costly to compute than the unoptimized contraction itself, it provides an excellent benchmark.
The function that computes this path in opt_einsum is called ``optimal`` and works by iteratively finding every possible combination of pairs to contract in the current list of tensors.
This is iterated until all tensors are contracted together. The resulting paths are then sorted by total flop cost and the lowest one is chosen.
This algorithm runs in about 1 second for 7 terms, 15 seconds for 8 terms, and 480 seconds for 9 terms limiting its overall usefulness for a large number of terms.
By considering limited memory this can be sieved and can reduce the cost of computing the optimal function by an order of magnitude or more.

Lets look at an example:
```
Contraction:  abc,dc,ac->bd
```

Build a list with tuples that have the following form:
```python
iteration 0:
 "(cost, path,  list of input sets remaining)"
[ (0,    [],    [set(['a', 'c', 'b']), set(['d', 'c']), set(['a', 'c'])] ]
```

Since this is iteration zero, we have the initial list of input sets.
We can consider three possible combinations where we contract list positions (0, 1), (0, 2), or (1, 2) together:
```python
iteration 1:
[ (9504, [(0, 1)], [set(['a', 'c']), set(['a', 'c', 'b', 'd'])  ]),
  (1584, [(0, 2)], [set(['c', 'd']), set(['c', 'b'])            ]),
  (864,  [(1, 2)], [set(['a', 'c', 'b']), set(['a', 'c', 'd'])  ])]
```
We have now run through the three possible combinations, computed the cost of the contraction up to this point, and appended the resulting indices from the contraction to the list.
As all contractions only have two remaining input sets the only possible contraction is (0, 1):
```python
iteration 2:
[ (28512, [(0, 1), (0, 1)], [set(['b', 'd'])  ]),
  (3168,  [(0, 2), (0, 1)], [set(['b', 'd'])  ]),
  (19872, [(1, 2), (0, 1)], [set(['b', 'd'])  ])]
```
The final contraction cost is computed and we choose the second path from the list as the overall cost is the lowest.



## Finding the opportunistic path

Another way to find a path is to choose the best pair to contract at every iteration so that the formula scales like N^3. 
The "best" contraction pair is currently determined by the smallest of the tuple ``(-removed_size, cost)`` where ``removed_size`` is the size of the contracted tensors minus the size of the tensor created and ``cost`` is the cost of the contraction.
Basically, we want to remove the largest dimensions at the least cost.
To prevent large outer products the results are sieved by the amount of memory available.
Overall, this turns out to work extremely well and is only slower than the optimal path in several cases, and even then only by a factor of 2-4 while only taking 1 millisecond for terms of length 10.
To me, while still not perfect, it represents a "good enough" algorithm for general production.
It is fast enough that at worst case the overhead penalty is approximately 20 microseconds and is much faster for every other einsum test case I can build or generate randomly.

## Testing

Testing this function thoroughly is absolutely crucial; the testing scripts do required python pandas in addition to numpy. Testing is broken down into several tasks:

 - test_random: builds expressions of random term length where each term is of a random number of indices and contracts them together and finally compares to the einsum result.
 - test_set: runs through the current set of tests in test_helper.py.
 - test_singular: runs a single test from the test_helper set with debug and path printing. 
 - test_path: compares the optimal and opportunistic paths.
    
## Outstanding issues


 - path_optimal is poorly programmed. A dynamic programming approach would help greatly.
 - Comparing path_optimal and path_opportunistic shows that path_opportunistic can occasionally be faster. This is due to the fact that the input and output index order can have dramatic effects on performance for einsum.
 - The "improved" tensordot code is becoming fairly unwieldy. At this point only about ~40% of the dot-like expressions are handed off to tensordot.  
 - I make a lot of assumptions about tensordot as I am testing against vendor BLAS (intel MKL on haswell or opteron architecture).  
 - More memory options should be available. For example should we consider cumulative memory? (Feedback on the numpy mailing suggest this is not a great concern) 
 - Are we handling view dereferencing correctly? Views really should be garbage collected as soon as possible.

## Installation

Thanks to [Nils Werner](https://github.com/nils-werner) `opt_einsum` can be installed with the line `pip install -e .[tests]`.
Test cases can then be run with `py.test -v`.

We are also now on PyPi: `pip install opt_einsum`
