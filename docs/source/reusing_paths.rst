=============
Reusing Paths
=============

If you expect to repeatedly use a particular contraction it can make things simpler and more efficient to not compute the path each time. Instead, supplying :func:`~opt_einsum.contract_expression` with the contraction string and the shapes of the tensors generates a :class:`~opt_einsum.contract.ContractExpression` which can then be repeatedly called with any matching set of arrays. For example:

.. code:: python

    >>> my_expr = oe.contract_expression("abc,cd,dbe->ea", (2, 3, 4), (4, 5), (5, 3, 6))
    >>> print(my_expr)
    <ContractExpression> for 'abc,cd,dbe->ea':
      1.  'dbe,cd->bce' [GEMM]
      2.  'bce,abc->ea' [GEMM]

The ``ContractExpression`` can be called with 3 arrays that match the original shapes without having to recompute the path:

.. code:: python

    >>> x, y, z = (np.random.rand(*s) for s in [(2, 3, 4), (4, 5), (5, 3, 6)])
    >>> my_expr(x, y, z)
    array([[ 3.08331541,  4.13708916],
           [ 2.92793729,  4.57945185],
           [ 3.55679457,  5.56304115],
           [ 2.6208398 ,  4.39024187],
           [ 3.66736543,  5.41450334],
           [ 3.67772272,  5.46727192]])

Note that few checks are performed when calling the expression, and while it will work for a set of arrays with the same ranks as the original shapes but differing sizes, it might no longer be optimal.
