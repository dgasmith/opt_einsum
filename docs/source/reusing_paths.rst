=============
Reusing Paths
=============

If you expect to repeatedly use a particular contraction it can make things simpler and more efficient to not compute the path each time. Instead, supplying :func:`~opt_einsum.contract_expression` with the contraction string and the shapes of the tensors generates a :class:`~opt_einsum.contract.ContractExpression` which can then be repeatedly called with any matching set of arrays. For example:

.. code:: python

    >>> my_expr = oe.contract_expression("abc,cd,dbe->ea", (2, 3, 4), (4, 5), (5, 3, 6))
    >>> print(my_expr)
    <ContractExpression('abc,cd,dbe->ea')>
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


====================
Specifying Constants
====================

Often one generates contraction expressions where some of the tensor arguments
will remain *constant* across many calls.
:func:`~opt_einsum.contract_expression` allows you to specify the indices of
these constant arguments, allowing ``opt_einsum`` to build and then reuse as
many constant contractions as possible. Take for example the equation:

.. code:: python

    >>> eq = "ij,jk,kl,lm,mn->ni"

where we know that only the first and last tensors will vary between calls.
We can specify this by marking the middle three as constant - we then need to
supply the actual arrays rather than just the shapes to
:func:`~opt_einsum.contract_expression`:

.. code:: python

    >>> # the shapes of the expression
    >>> args = [(9, 5), (5, 5), (5, 5), (5, 5), (5, 8)]

    >>> # now replace the constants with actual arrays
    >>> constants = [1, 2, 3]
    >>> for i in constant:
    ...     args[i] = np.random.rand(*args[i])

    >>> expr = oe.contract_expression(eq, *args, constants=constants)
    >>> expr
    <ContractExpression('ij,[jk,kl,lm],mn->ni', constants=[1, 2, 3])>

The expression now only takes the remaining two arrays as arguments (the
tensors with ``'ij'`` and ``'mn'`` indices), and will store as many resuable
constant contractions as possible.

.. code:: python

    >>> out1 = expr(np.random.rand(9, 5), np.random.rand(5, 8))
    >>> out1.shape
    (8, 9)

    >>> out2 = expr(np.random.rand(9, 5), np.random.rand(5, 8))
    >>> out2.shape
    (8, 9)

    >>> np.allclose(out1, out2)
    False

    >>> print(expr)
    <ContractExpression('ij,[jk,kl,lm],mn->ni', constants=[1, 2, 3])>
      1.  'jm,mn->jn' [GEMM]
      2.  'jn,ij->ni' [GEMM]

Where we can see that the expression now only has to perform
two contractions to compute the output.

.. note::

    The constant part of an expression is lazily generated upon first call,
    (with a particular backend) though it can be explicitly built with call to
    :meth:`ContractExpression.parse_constants`.

Even if there are no constant contractions to perform, it can be very
advantageous to specify constant tensors for particular backends.
For instance, if a GPU backend is used, the constant tensors will be kept on
the device rather than being transfered each time.
