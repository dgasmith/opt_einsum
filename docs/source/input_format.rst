============
Input Format
============

The ``opt_einsum`` package was originally designed as a drop-in replacement for the ``np.einsum``
function and supports all input formats that ``np.einsum`` supports. There are
two styles of input accepted, a basic introduction to which can be found in the
documentation for :func:`numpy.einsum`. In addition to this, ``opt_einsum``
extends the allowed index labels to unicode or arbitrary hashable, comparable
objects in order to handle large contractions with many indices.


'Equation' Input
----------------

As with :func:`numpy.einsum`, here you specify an equation as a string,
followed by the array arguments:

.. code:: python

    >>> import opt_einsum as oe
    >>> eq = 'ijk,jkl->li'
    >>> x, y = np.random.rand(2, 3, 4), np.random.rand(3, 4, 5)
    >>> z = oe.contract(eq, x, y)
    >>> z.shape
    (5, 2)

However, in addition to the standard alphabet, ``opt_einsum`` also supports
unicode characters:

.. code:: python

    >>> eq = "αβγ,βγδ->δα"
    >>> oe.contract(eq, x, y).shape
    (5, 2)

This enables access to thousands of possible index labels. One way to access
these programmatically is through the function
:func:`~opt_einsum.parser.get_symbol`:

    >>> oe.get_symbol(805)
    'α'

which maps an ``int`` to a unicode characater. Note that as with
:func:`numpy.einsum` if the output is not specified with ``->`` it will default
to the sorted order of all indices appearing once:

.. code:: python

    >>> eq = "αβγ,βγδ"  # "->αδ" is implicit
    >>> oe.contract(eq, x, y).shape
    (2, 5)


'Interleaved' Input
-------------------

The other input format is to 'interleave' the array arguments with their index
labels ('subscripts') in pairs, optionally specifying the output indices as a
final argument. As with :func:`numpy.einsum`, integers are allowed as these
index labels:

.. code:: python

    >>> oe.contract(x, [1, 2, 3], y, [2, 3, 4], [4, 1]).shape
    >>> (5, 2)

with the default output order again specified by the sorted order of indices
appearing once. However, unlike :func:`numpy.einsum`, in ``opt_einsum`` you can
also put *anything* hashable and comparable such as `str` in the subscript list.
A simple example of this syntax is:

.. code:: python

    >>> x, y, z = np.ones((1, 2)), np.ones((2, 2)), np.ones((2, 1))
    >>> oe.contract(x, ('left', 'bond1'), y, ('bond1', 'bond2'), z, ('bond2', 'right'), ('left', 'right'))
    array([[4.]])

The subscripts need to be hashable so that ``opt_einsum`` can efficiently process them, and
they should also be comparable so as to allow a default sorted output. For example:

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


