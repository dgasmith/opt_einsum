===============
The Greedy Path
===============

The ``greedy`` path iterates through the possible pair contractions and chooses the "best" contraction at every step until all contractions are considered.
The "best" contraction pair is determined by the smallest of the tuple ``(-removed_size, cost)`` where ``removed_size`` is the size of the contracted tensors minus the size of the tensor created and ``cost`` is the cost of the contraction.
Effectively, the algorithm chooses the best inner or dot product, Hadamard product, and then outer product at each iteration with a sieve to prevent large outer products.
This algorithm has proven to be quite successful for general production and only misses a few complex cases that make it slightly worse than the ``optimal`` algorithm.
Fortunately, these often only lead to increases in prefactor than missing the optimal scaling.

The ``greedy`` approach scales like N^2 rather than factorially, making ``greedy`` much more suitable for large numbers of contractions where the lower prefactor helps decrease latency.
As :mod:`opt_einsum` can handle an arbitrary number of indices the low scaling is necessary for extensive contraction networks.
The ``greedy`` functionality is provided by :func:`~opt_einsum.paths.greedy`.

Optimal Scaling Misses
----------------------
The greedy algorithm, while inexpensive, can occasionally miss optimal scaling in some circumstances as seen below. The ``greedy`` algorithm prioritizes expressions which remove the largest indices first, in this particular case this is the incorrect choice and it is difficult for any heuristic algorithm to "see ahead" as would be needed here.

It should be stressed these cases are quite rare and by default ``contract`` uses the ``optimal`` path for four and fewer inputs as the cost of evaluating the ``optimal`` path is similar to that of the ``greedy`` path.

.. code:: python

    >>> M = np.random.rand(35, 37, 59)
    >>> A = np.random.rand(35, 51, 59)
    >>> B = np.random.rand(37, 51, 51, 59)
    >>> C = np.random.rand(59, 27)

    >>> path, desc = oe.contract_path('xyf,xtf,ytpf,fr->tpr', M, A, B, C, path="greedy")
    >>> print(desc)
      Complete contraction:  xyf,xtf,ytpf,fr->tpr
             Naive scaling:  6
         Optimized scaling:  5
          Naive FLOP count:  2.146e+10
      Optimized FLOP count:  4.165e+08
       Theoretical speedup:  51.533
      Largest intermediate:  5.371e+06 elements
    --------------------------------------------------------------------------------
    scaling        BLAS                current                             remaining
    --------------------------------------------------------------------------------
       5          False         ytpf,xyf->tpfx                      xtf,fr,tpfx->tpr
       4          False          tpfx,xtf->tpf                           fr,tpf->tpr
       4           GEMM            tpf,fr->tpr                              tpr->tpr

    >>> path, desc = oe.contract_path('xyf,xtf,ytpf,fr->tpr', M, A, B, C, path="optimal")
    >>> print(desc)

      Complete contraction:  xyf,xtf,ytpf,fr->tpr
             Naive scaling:  6
         Optimized scaling:  4
          Naive FLOP count:  2.146e+10
      Optimized FLOP count:  2.744e+07
       Theoretical speedup:  782.283
      Largest intermediate:  1.535e+05 elements
    --------------------------------------------------------------------------------
    scaling        BLAS                current                             remaining
    --------------------------------------------------------------------------------
       4          False           xtf,xyf->tfy                      ytpf,fr,tfy->tpr
       4          False          tfy,ytpf->tfp                           fr,tfp->tpr
       4           TDOT            tfp,fr->tpr                              tpr->tpr