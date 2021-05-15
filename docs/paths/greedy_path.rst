===============
The Greedy Path
===============

The ``'greedy'`` approach provides a very efficient strategy for finding
contraction paths for expressions with large numbers of tensors.
It does this by eagerly choosing contractions in three stages:

  1. Eagerly compute any **Hadamard** products (in arbitrary order -- this is
     commutative).
  2. Greedily contract pairs of remaining tensors, at each step choosing the
     pair that maximizes ``reduced_size`` -- these are generally **inner**
     products.
  3. Greedily compute any pairwise **outer** products, at each step choosing
     the pair that minimizes ``sum(input_sizes)``.

The cost heuristic ``reduced_size`` is simply the size of the pair of potential
tensors to be contracted, minus the size of the resulting tensor.

The ``greedy`` algorithm has space and time complexity ``O(n * k)`` where ``n``
is the number of input tensors and ``k`` is the maximum number of tensors that
share any dimension (excluding dimensions that occur in the output or in every
tensor). As such, the algorithm scales well to very large sparse contractions
of low-rank tensors, and indeed, often finds the optimal, or close to optimal
path in such cases.

The ``greedy`` functionality is provided by :func:`~opt_einsum.paths.greedy`,
and is selected by the default ``optimize='auto'`` mode of ``opt_einsum`` for
expressions with many inputs. Expressions of up to a thousand tensors
should still take well less than a second to find paths for.


Optimal Scaling Misses
----------------------

The greedy algorithm, while inexpensive, can occasionally miss optimal scaling in some circumstances as seen below. The ``greedy`` algorithm prioritizes expressions which remove the largest indices first, in this particular case this is the incorrect choice and it is difficult for any heuristic algorithm to "see ahead" as would be needed here.

It should be stressed these cases are quite rare and by default ``contract`` uses the ``optimal`` path for four and fewer inputs as the cost of evaluating the ``optimal`` path is similar to that of the ``greedy`` path. Similarly, for 5-8 inputs, ``contract`` uses one of the
branching strategies which can find higher quality paths.

.. code:: python

    >>> M = np.random.rand(35, 37, 59)
    >>> A = np.random.rand(35, 51, 59)
    >>> B = np.random.rand(37, 51, 51, 59)
    >>> C = np.random.rand(59, 27)

    >>> path, desc = oe.contract_path('xyf,xtf,ytpf,fr->tpr', M, A, B, C, optimize="greedy")
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

    >>> path, desc = oe.contract_path('xyf,xtf,ytpf,fr->tpr', M, A, B, C, optimize="optimal")
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


So we can see that the ``greedy`` algorithm finds a path which is about 16
times slower than the ``optimal`` one. In such cases, it might be worth using
one of the more exhaustive optimization strategies: ``'optimal'``,
``'branch-all'`` or ``branch-2`` (all of which will find the optimal path in
this example).


Customizing the Greedy Path
---------------------------

The greedy path is a local optimizer in that it only ever assesses pairs of
tensors to contract, assigning each a heuristic 'cost' and then choosing the
'best' of these. Custom greedy approaches can be implemented by supplying
callables to the ``cost_fn`` and ``choose_fn`` arguments of
:func:`~opt_einsum.paths.greedy`.
