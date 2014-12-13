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

The einsum function does not consider building intermediate arrays, therefore helping out can result in considerable cost savings even for a small N (N=10):

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

At first, it would appear that this scales like N^7 as there are 7 unique indices; however, we can define a intermeidate to greatly reduce this scaling.

'a = bdik,ikab,ikbd' (N^6 scaling)
'result = acaj,ajac,a` (N^4 scaling)

this is a single possible path to the final answer (and notably, not the most optimal) out of many possible paths. 
Finding the best path for an arbitrary number of terms is not a easy problem.
At this point it is worth noting that we are aiming at only optimizing a single conctraction here.

** IN PROGRESS **

Run through an example:

```python
opt_einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)

# First
string = 'pi,qj,ijkl,rk,sl->pqrs'
views = [C1, C2, I, C3, C4]
c1 = ['pi,ijkl->pjkl', [2, 0]]

I1 = perform_contract(c1)
views = view.pop(2,0)
views += [I1]
string = string.pop(2,0)
string += [c1_result]

string = 'qj,rk,sl,pjkl->pqrs'
views = [C2, C3, C4, I1]
c2 = ['qj,pjkl->pqkl', [0, 3]]

I2 = perform_contract(c2)
...
string = 'rk,sl,pqkl->pqrs'
views = [C3, C4, I2]
c3 = ['rk,pqkl->pqrl', [0, 2]]

I3 = perform_contrce(c3)
...
string = 'sl,pqkl->pqrs'
views = [C4, I3]
c4 = ['sl,pqkl->pqrs', [0, 1]]


find_path(string, views):
    ...
    return [c1, c2, c3, c4]


done
```

Assumptions:
 - 'ijk,ijk,ijk' or 'ijk,ijk,jk' is an optimized expression.
 - Memory = max(output.shape, memory_set)


Opportunistic algorithm (order of importance):
 - Explore all possible 2 combinations
 - Run through assumptions
 - Importance:
   - Rank reduction, ordered by size
   - Term reduction
   - Memory reduction




