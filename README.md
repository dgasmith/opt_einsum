opt_einsum
==========

Optimizing numpy's einsum function.

Einsum is a very powerful function for contracting tensors of arbitrary dimension and index.
However, it is only optimized to contract two terms at a time resulting in non-optimal scaling.

For example, consider the following index transformation:
`M_{pqrs} = C_{pi} C_{qj} I_{ijkl} C_{rk} C_{sl}`

Consider two different algorithms:
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

The einsum function does not consider building intermediate arrays, therefore helping einsum out by building intermediate arrays can result in a considerable cost savings even for small N (N=10):

```
%timeit naive(I, C)
1 loops, best of 3: 1.18 s per loop

%timeit optimized(I, C)
1000 loops, best of 3: 612 µs per loop
```

The index transformation is a fairly simple contraction that leads to straightforward intermediates.
This can be further complicated by considering the shape of the C matrices need not be the same and then the ordering in which the intermediate contraction matters greatly.
Logic can be built that optimizes the ordering; however, this is a lot of time and effort for a single expression. 
Now lets consider the following expression found in perturbation theory (one of ~5,000 such expressions):
`bdik,acaj,ikab,ajac,ikbd`

At first, it would appear that this scales like N^7 as there are 7 unique indices; however, we can define a intermediate to reduce this scaling.

`a = bdik,ikab,ikbd` (N^6 scaling)

`result = acaj,ajac,a` (N^4 scaling)

this is a single possible path to the final answer (and notably, not the most optimal) out of many possible paths. Now lets let opt_einsum compute the optimal path.

```python
import test_helper as th
from opt_einsum import opt_einsum

sum_string = 'bdik,acaj,ikab,ajac,ikbd'
index_size = [10, 17, 9, 10, 13, 16, 15, 14, 11]]
views = th.build_views(sum_string, index_size) # Function that builds random arrays of the correct shape
ein_result = np.einsum(sum_string, *views)
opt_ein_result = opt_einsum(sum_string, *views, debug=1, path=path)

Complete contraction:  bdik,acaj,ikab,ajac,ikbd->
       Naive scaling:   7
--------------------------------------------------------------------------------
scaling   GEMM                   current                                remaining
--------------------------------------------------------------------------------
   3     False              ajac,acaj->a                       bdik,ikab,ikbd,a->
   4     False            ikbd,bdik->bik                             ikab,a,bik->
   4      True               bik,ikab->a                                    a,a->
   1      True                     a,a->                                      ,->
   
np.allclose(ein_result, opt_ein_result)
>>> True
   ```
By contracting terms in the correct order we can see that this expression can be computed with N^4 scaling. Even with the overhead of finding the best order or 'path' and the small dimensions opt_einsum is roughly 900 times faster than pure einsum.





