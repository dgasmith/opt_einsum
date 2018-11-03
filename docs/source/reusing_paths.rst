=============
Reusing Paths
=============

If you expect to use a particular contraction repeatedly, it can make things simpler and more efficient not to compute the path each time. Instead, supplying :func:`~opt_einsum.contract_expression` with the contraction string and the shapes of the tensors generates a :class:`~opt_einsum.contract.ContractExpression` which can then be repeatedly called with any matching set of arrays. For example:

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


.. _constants-section:

Specifying Constants
====================

Often one generates contraction expressions where some of the tensor arguments
will remain *constant* across many calls.
:func:`~opt_einsum.contract_expression` allows you to specify the indices of
these constant arguments, allowing ``opt_einsum`` to build and then reuse as
many constant contractions as possible. Take for example the equation:

.. code:: python

    >>> eq = "ij,jk,kl,lm,mn->ni"

where we know that *only* the first and last tensors will vary between calls.
We can specify this by marking the middle three as constant - we then need to
supply the actual arrays rather than just the shapes to
:func:`~opt_einsum.contract_expression`:

.. code:: python

    >>> #           A       B       C       D       E
    >>> shapes = [(9, 5), (5, 5), (5, 5), (5, 5), (5, 8)]

    >>> # mark the middle three arrays as constant
    >>> constants = [1, 2, 3]

    >>> # generate the constant arrays
    >>> B, C, D = [np.random.randn(*shapes[i]) for i in constants]

    >>> # supplied ops are now mix of shapes and arrays
    >>> ops = (9, 5), B, C, D, (5, 8)

    >>> expr = oe.contract_expression(eq, *ops, constants=constants)
    >>> expr
    <ContractExpression('ij,[jk,kl,lm],mn->ni', constants=[1, 2, 3])>

The expression now only takes the remaining two arrays as arguments (the
tensors with ``'ij'`` and ``'mn'`` indices), and will store as many reusable
constant contractions as possible.

.. code:: python

    >>> A1, E1 = np.random.rand(*shapes[0]), np.random.rand(*shapes[-1])
    >>> out1 = expr(A1, E1)
    >>> out1.shap
    (8, 9)

    >>> A2, E2 = np.random.rand(*shapes[0]), np.random.rand(*shapes[-1])
    >>> out2 = expr(A2, E2)
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

    The constant part of an expression is lazily generated upon the first call
    (specific to each backend), though it can also be explicitly built by calling
    :meth:`~opt_einsum.contract.ContractExpression.evaluate_constants`.

We can confirm the advantage of using expressions and constants by timing the
following scenarios, first setting
``A = np.random.rand(*shapes[0])`` and ``E = np.random.rand(*shapes[-1])``.

- **contract from scratch:**

.. code:: python

    >>> %timeit oe.contract(eq, A, B, C, D, E)
    239 µs ± 5.06 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

- **contraction with an expression but no constants:**

.. code:: python

    >>> expr_no_consts = oe.contract_expression(eq, *shapes)
    >>> %timeit expr_no_consts(A, B, C, D, E)
    76.7 µs ± 2.47 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

- **contraction with an expression and constants marked:**

.. code:: python

    >>> %timeit expr(A, E)
    40.8 µs ± 1.22 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

Although this gives us a rough idea, of course the efficiency savings are
hugely dependent on the size of the contraction and number of possible constant
contractions.

We also note that even if there are *no* constant contractions to perform, it
can be very advantageous to specify constant tensors for particular backends.
For instance, if a GPU backend is used, the constant tensors will be kept on
the device rather than being transferred each time.
