======================
Backends & GPU Support
======================

A brief overview of libraries that have been tested:

    - `tensorflow <https://www.tensorflow.org/>`_: compiled tensor expressions
      that can run on GPU.
    - `theano <http://deeplearning.net/software/theano/>`_: compiled tensor
      expressions that can run on GPU.
    - `cupy <https://cupy.chainer.org/>`_: numpy-like api for GPU tensors.
    - `dask <https://dask.pydata.org/>`_: larger-than-memory tensor
      computations, distributed scheduling, and potential reuse of
      intermediaries.
    - `sparse <https://sparse.pydata.org/>`_: sparse tensors.

``opt_einsum`` is quite agnostic to the type of n-dimensional arrays (tensors)
it uses, since finding the contraction path only relies on getting the shape
attribute of each array supplied.
It can perform the underlying tensor contractions with various
libraries. In fact, any library that provides a :func:`~numpy.tensordot` and
:func:`~numpy.transpose` implementation can perform most normal contractions.
While more special functionality such as taking diagonals is reliant on a
:func:`~numpy.einsum` implementation.



General backend for any type of array
-------------------------------------

This 'duck-typing' support just requires specifying the correct ``backend``
argument for the type of arrays supplied when calling
:func:`~opt_einsum.contract`. For example with ``dask``:

.. code-block:: python

    >>> import opt_einsum as oe
    >>> import dask.array as da
    >>> shapes = (3, 200), (200, 300), (300, 4)
    >>> dxs = [da.random.normal(0, 1, shp, chunks=(100, 100)) for shp in shapes]
    >>> dxs
    [dask.array<da.random.normal, shape=(3, 200), dtype=float64, chunksize=(3, 100)>,
     dask.array<da.random.normal, shape=(200, 300), dtype=float64, chunksize=(100, 100)>,
     dask.array<da.random.normal, shape=(300, 4), dtype=float64, chunksize=(100, 4)>]


    >>> dy = oe.contract("ab,bc,cd", *dxs, backend='dask')
    >>> dy
    dask.array<transpose, shape=(3, 4), dtype=float64, chunksize=(3, 4)>

    >>> dy.compute()
    array([[ 470.71404665,    2.44931372,  -28.47577265,  424.37716615],
           [  64.38328345, -287.40753131,  144.46515642,  324.88169821],
           [-142.07153553, -180.41739259,  125.0973783 , -239.16754541]])


In this case, dask arrays in = dask array out, since dask arrays have a shape
attribute, and ``opt_einsum`` can find ``dask.array.tensordot`` and
``dask.array.transpose``.



Special Backends for numpy arrays - GPU etc.
----------------------------------------------

A special case is if you want to supply numpy arrays and get numpy arrays back,
but use a different backend, such as performing a contraction on a GPU.
Unless the specified backend works on numpy arrays this requires converting to
and from the backend array type. Currently ``opt_einsum`` can handle this
automatically for:

    - ``tensorflow``
    - ``theano``
    - ``cupy``

which all offer GPU support. Since ``tensorflow`` and ``theano`` both require
compiling the expression, this functionality is encapsulated in generating a
:class:`~opt_einsum.ContractExpression` using
:func:`~opt_einsum.contract_expression`, which can then be called using numpy
arrays whilst specifiying ``backend='tensorflow'`` etc.

For example with **theano**:

.. code-block:: python

    >>> import opt_einsum as oe
    >>> shapes = (3, 200), (200, 300), (300, 4)
    >>> expr = oe.contract_expression("ab,bc,cd", *shapes)
    >>> expr
    ContractExpression('ab,bc,cd')

    >>> import numpy as np
    >>> # GPU advantage mainly for low precision numbers
    >>> xs = xs = [np.random.randn(*shp).astype(np.float32) for shp in shapes]
    >>> expr(*xs, backend='theano')  # might see some fluff on first run
    ...
    array([[ 129.28352  , -128.00702  , -164.62917  , -335.11682  ],
           [-462.52344  , -121.12657  ,  -67.847626 ,  624.5457   ],
           [   5.2838974,   36.441578 ,   81.62851  ,  703.1576   ]],
          dtype=float32)

To run the expression with **tensorflow**, you need to register a default
session:

.. code-block:: python

    >>> import tensorflow as tf
    >>> sess = tf.Session()  # might see some fluff
    ...

    >>> with sess.as_default(): out = expr(*xs, backend='tensorflow')
    >>> out
    array([[ 129.28357  , -128.00684  , -164.62903  , -335.1167   ],
           [-462.52362  , -121.12659  ,  -67.84769  ,  624.5455   ],
           [   5.2839584,   36.44155  ,   81.62852  ,  703.15784  ]],
          dtype=float32)

Note that one could still supply this expression with, for example, a
``tensorflow.placeholder`` using ``backend='tensorflow'``, and then no
conversion would take place, instead you'd get a ``tensorflow.Tensor`` back.
