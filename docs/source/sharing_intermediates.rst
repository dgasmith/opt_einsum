=====================
Sharing Intermediates
=====================

If you want to compute multiple similar contractions with common terms, you can embed them in a :func:`~opt_einsum.shared_intermediates` context. Computations of subexpressions in this context will be memoized, and will be garbage collected when the contexts exits.

For example, suppose we want to compute marginals at each point in a factor chain:

.. code:: python

    >>> inputs = 'ab,bc,cd,de,ef'
    >>> factors = [np.random.rand(4, 4) for _ in range(5)]
    >>> marginals = {output: contract('{}->{}'.format(inputs, output), *factors)
    >>>              for output in 'abcdef'}

To share this computation, we can perform all contractions in a shared context

.. code:: python

    >>> with shared_intermediates():
    >>>     marginals = {output: contract('{}->{}'.format(inputs, output), *factors)
    >>>                  for output in 'abcdef'}

If it is difficult to fit your code into a context, you can instead save the sharing cache for later reuse.

.. code:: python

    >>> with shared_intermediates() as cache:  # create a cache
    >>>     pass
    >>> marginals = {}
    >>> for output in 'abcdef':
    >>>     with shared_intermediates(cache):  # reuse a common cache
    >>>         marginals[output] = contract('{}->{}'.format(inputs, output), *factors)
    >>> del cache  # garbage collect intermediates

Note that sharing contexts can be nested, so it is safe to to use :func:`~opt_einsum.shared_intermediates` in library code without leaking intermediates into user caches.
