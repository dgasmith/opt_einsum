==========
opt_einsum
==========

:func:`~numpy.einsum` is a powerful function for contracting tensors of arbitrary
dimension and index. However, it is only optimized to contract two terms
at a time resulting in non-optimal scaling.

For example, consider the following index transformation:

.. math::

    M_{pqrs} = C_{pi} C_{qj} I_{ijkl} C_{rk} C_{sl}

Consider two different algorithms:

.. code:: python

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

The einsum function does not consider building intermediate arrays;
therefore, helping einsum out by creating these intermediate arrays can result
in considerable cost savings even for small N (N=10):

.. code:: python

    >> np.allclose(naive(I, C), optimized(I, C))
    True

    %timeit naive(I, C)
    1 loops, best of 3: 829 ms per loop

    %timeit optimized(I, C)
    1000 loops, best of 3: 445 µs per loop

The index transformation is a well-known contraction that leads to
straightforward intermediates. This contraction can be further
complicated by considering that the shape of the C matrices need not be
the same, in this case, the ordering in which the indices are transformed
matters significantly. Logic can be built that optimizes the order;
however, this is a lot of time and effort for a single expression.

The opt_einsum package is a drop-in replacement for the ``np.einsum`` function
and can handle all of the logic for you:

.. code:: python

    from opt_einsum import contract

    dim = 30
    I = np.random.rand(dim, dim, dim, dim)
    C = np.random.rand(dim, dim)

    %timeit optimized(I, C)
    10 loops, best of 3: 65.8 ms per loop

    %timeit contract('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
    100 loops, best of 3: 16.2 ms per loop

The above will automatically find the optimal contraction order, in this case, identical to that of the optimized function above, and compute the products for
you. In this case, it even uses `np.dot` under the hood to exploit any vendor
BLAS functionality that your NumPy build has!

We can then view more details about the optimized contraction order:

.. code:: python

    >>> from opt_einsum import contract_path

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

Citation
========

If this code has benefited your research, please support us by citing:

Daniel G. A. Smith and Johnnie Gray, opt_einsum - A Python package for optimizing contraction order for einsum-like expressions. **Journal of Open Source Software**, *2018*, 3(26), 753

DOI: https://doi.org/10.21105/joss.00753

========

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   input_format
   backends
   reusing_paths
   sharing_intermediates

.. toctree::
   :maxdepth: 1
   :caption: Path Information:

   path_finding
   optimal_path
   branching_path
   greedy_path

.. toctree::
   :maxdepth: 1
   :caption: Examples

   ex_large_expr_with_greedy
   ex_dask_reusing_intermediaries

.. toctree::
   :maxdepth: 1
   :caption: Help & Reference:

   api
   changelog
