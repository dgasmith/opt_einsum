opt_einsum
==========

If this grows any further it might be a good idea to migrate to the wiki.

# TOC
 - [Optimizing numpy's einsum function](https://github.com/dgasmith/opt_einsum/blob/master/README.md#optimizing-numpys-einsum-function)
 - [More details on paths](https://github.com/dgasmith/opt_einsum/blob/master/README.md#more-details-on-paths)
 - [Finding the optimal path](https://github.com/dgasmith/opt_einsum/blob/master/README.md#finding-the-optimal-path)
 - [Finding the opportunistic path](https://github.com/dgasmith/opt_einsum/blob/master/README.md#finding-the-opportunistic-path)
 - [Testing](https://github.com/dgasmith/opt_einsum/blob/master/README.md#testing)
 - [Outstanding issues](https://github.com/dgasmith/opt_einsum/blob/master/README.md#outstanding-issues)

## Optimizing numpy's einsum function
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

The einsum function does not consider building intermediate arrays; therefore, helping einsum out by building intermediate arrays can result in a considerable cost savings even for small N (N=10):

```
%timeit naive(I, C)
1 loops, best of 3: 1.18 s per loop

%timeit optimized(I, C)
1000 loops, best of 3: 612 µs per loop
```

The index transformation is a fairly simple contraction that leads to straightforward intermediates.
This can be further complicated by considering that the shape of the C matrices need not be the same, in this case the ordering in which the indices are transformed matters greatly.
Logic can be built that optimizes the ordering; however, this is a lot of time and effort for a single expression. 
Now lets consider the following expression found in a perturbation theory (one of ~5,000 such expressions):
`bdik,acaj,ikab,ajac,ikbd`

At first, it would appear that this scales like N^7 as there are 7 unique indices; however, we can define a intermediate to reduce this scaling.

`a = bdik,ikab,ikbd` (N^6 scaling)

`result = acaj,ajac,a` (N^4 scaling)

this is a single possible path to the final answer (and notably, not the most optimal) out of many possible paths. Now lets let opt_einsum compute the optimal path:

```python
import test_helper as th
from opt_einsum import opt_einsum

sum_string = 'bdik,acaj,ikab,ajac,ikbd'
index_size = [10, 17, 9, 10, 13, 16, 15, 14, 11]]
views = th.build_views(sum_string, index_size) # Function that builds random arrays of the correct shape
ein_result = np.einsum(sum_string, *views)
opt_ein_result = opt_einsum(sum_string, *views, debug=1)

Complete contraction:  bdik,acaj,ikab,ajac,ikbd->
       Naive scaling:   7
---------------------------------------------------------------------------------
scaling   GEMM                   current                                remaining
---------------------------------------------------------------------------------
   3     False              ajac,acaj->a                       bdik,ikab,ikbd,a->
   4     False            ikbd,bdik->bik                             ikab,a,bik->
   4      True               bik,ikab->a                                    a,a->
   1      True                     a,a->                                      ,->
   
np.allclose(ein_result, opt_ein_result)
>>> True
   ```
By contracting terms in the correct order we can see that this expression can be computed with N^4 scaling. Even with the overhead of finding the best order or 'path' and small dimensions, opt_einsum is roughly 900 times faster than pure einsum for this expression.

## More details on paths

Finding the optimal order of contraction is not an easy problem and formally scales factorial with respect to the number of terms in the expression. First, lets discuss what a path looks like in opt_einsum:
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

```python
---------------------------------------------------------------------------------
scaling   GEMM                   current                                remaining
---------------------------------------------------------------------------------
terms = ['bdik', 'acaj', 'ikab', 'ajac', 'ikbd'] contraction = (1, 3)
  3     False              ajac,acaj->a                       bdik,ikab,ikbd,a->
terms = ['bdik', 'ikab', 'ikbd', 'a'] contraction = (0, 2)
  4     False            ikbd,bdik->bik                             ikab,a,bik->
terms = ['ikab', 'a', 'bik'] contraction = (0, 2)
  4      True               bik,ikab->a                                    a,a->
terms = ['a', 'a'] contraction = (0, 1)
  1      True                     a,a->                                      ,->
   ```



## Finding the optimal path

The most optimal path can be found by searching through every possible way to contract the tensors together, this includes all combinations with the new intermediate tensors as well.
While this algorithm scales like N! and can often become more costly to compute than the unoptimized contraction itself, it provides an excellent benchmark.
The function that computes this path in opt_einsum is called _path_optimal and works by iteratively finding every possible combination of pairs to contract in the current list of tensors.
This is iterated until all tensors are contracted together. The resulting paths are then sorted by total flop cost and the lowest one is chosen.
This algorithm runs in about 1 second for 7 terms, 15 seconds for 8 terms, and 480 seconds for 9 terms limiting its overall usefulness for a large number of terms.
By considering limited memory this can be sieved and can reduce the cost of the optimal function by an order of magnitude or more.

Lets look at an example:
```python
Contraction:  abc,dc,ac->bd

iteration 0:
Build a list with tuples that look the following:
   cost  path   list of input sets remaining
[ (0,    [],    [set(['a', 'c', 'b']), set(['d', 'c']), set(['a', 'c'])] ]

```
Since this is effectively iteration zero, we have the entire list of input sets.
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
The "best" contraction pair is currently determined by the smallest of the tuple (-removed_size, cost) where removed size represents the product of the size of indices removed from the overall contraction and cost is the cost of the contraction.
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

 - path_opportunistic probably can be improved.
 - path_optimal is terribly programmed and very slow. A dynamic programming approach would help greatly.
 - Both paths should consider if tensordot can be used. The downside is: figuring out if tensordot can be used is quite expensive.
 - Einsum can compute any Hadamard product of 3 or more tensors faster than building intermediates (e.g ``` np.einsum('ij,ij,ij->ij', ...) or np.einsum('jk,ijk,ij->ijk', ...) ```. Testing shows we must consider the usage of tensordot as well or everything slows down considerably.
 - I make a lot of assumptions about tensordot as I am testing against vendor BLAS (intel MKL on haswell or opteron architecture).
 - Often we can choose the order of output indices, choosing the correct order can have speeds ups of 2x or more.
 - More memory options should be available. For example should we consider cumulative memory? 
 - Are we handling view dereferencing correctly? Views really should be garbage collected as soon as possible.




