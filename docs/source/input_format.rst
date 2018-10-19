============
Input Format
============

The ``opt_einsum`` package is a drop-in replacement for the ``np.einsum`` function
and supports all input formats that ``np.einsum`` supports.

TBA: "Normal" inputs

In ``opt_einsum``, you can also put anything hashable and comparable such as `str` in the subscript list.
Note ``np.einsum`` currently does not support this syntax. A simple example of the syntax is:

.. code:: python

    >>> x, y, z = np.ones((1, 2)), np.ones((2, 2)), np.ones((2, 1))
    >>> oe.contract(x, ('left', 'bond1'), y, ('bond1', 'bond2'), z, ('bond2', 'right'), ('left', 'right'))
    array([[4.]])

The subscripts need to be hashable so that ``opt_einsum`` can efficiently process them, and
they should also be comparable because in this way we can keep a consistent behavior between integer 
subscripts and other types of subscripts. For example:

.. code:: python

    >>> x = np.array([[0, 1], [2, 0]])
    >>> oe.contract(x, (0, 1))  # original matrix
    array([[0, 1],
           [2, 0]])
    >>> oe.contract(x, (1, 0)) # the transpose
    array([[0, 2],
           [1, 0]])
    >>> oe.contract(x, ('a', 'b'))  # original matrix, consistent behavior
    array([[0, 1],
           [2, 0]])
    >>> oe.contract(x, ('b', 'a')) # the transpose, consistent behavior
    array([[0, 2],
           [1, 0]])
    >>> oe.contract(x, (0, 'a')) # relative sequence undefined, can't determine output
    TypeError: For this input type lists must contain either Ellipsis or hashable and comparable object (e.g. int, str)


