======================
Backends & GPU Support
======================

The following is a brief overview of libraries which have been tested with
``opt_einsum``:

    - `tensorflow <https://www.tensorflow.org/>`_: compiled tensor expressions
      that can run on GPU.
    - `theano <http://deeplearning.net/software/theano/>`_: compiled tensor
      expressions that can run on GPU.
    - `cupy <https://cupy.chainer.org/>`_: numpy-like api for GPU tensors.
    - `dask <https://dask.pydata.org/>`_: larger-than-memory tensor
      computations, distributed scheduling, and potential reuse of
      intermediaries.
    - `sparse <https://sparse.pydata.org/>`_: sparse tensors.
    - `pytorch <https://pytorch.org>`_: numpy-like api for GPU tensors.

``opt_einsum`` is quite agnostic to the type of n-dimensional arrays (tensors)
it uses, since finding the contraction path only relies on getting the shape
attribute of each array supplied.
It can perform the underlying tensor contractions with various
libraries. In fact, any library that provides a :func:`~numpy.tensordot` and
:func:`~numpy.transpose` implementation can perform most normal contractions.
While more special functionality such as axes reduction is reliant on a
:func:`~numpy.einsum` implementation.

.. note::

    For a contraction to be possible without using a backend einsum, it must
    satisfy the following rule: in the full expression (so *including* output
    indices) each index must appear twice. In other words, each dimension
    must be contracted with one other dimension, or left alone.


General backend for any ndarray
===============================

This 'duck-typing' support just requires specifying the correct ``backend``
argument for the type of arrays supplied when calling
:func:`~opt_einsum.contract` or letting it be automatically discovered based on
the first array (the default). For example, if you had a library installed
called ``'foo'`` which provided an :class:`~numpy.ndarray` like object with a
``.shape`` attribute as well as ``foo.tensordot`` and ``foo.transpose`` then
you could contract them with something like:

.. code-block:: python

    contract(einsum_str, *foo_arrays, backend='foo')

Behind the scenes :mod:`opt_einsum` will find the contraction path, perform
pairwise contractions using e.g. ``foo.tensordot`` and finally return whatever
type those functions return. In fact, if you don't want to be explicit you can
leave ``backend='auto'`` here and ``opt_einsum`` will infer ``'foo'`` by
itself.


Dask
----

`dask <https://dask.pydata.org/>`_ is an example of a library which satisfies
these requirements. For example:

.. code-block:: python

    >>> import opt_einsum as oe
    >>> import dask.array as da
    >>> shapes = (3, 200), (200, 300), (300, 4)
    >>> dxs = [da.random.normal(0, 1, shp, chunks=(100, 100)) for shp in shapes]
    >>> dxs
    [dask.array<da.random.normal, shape=(3, 200), dtype=float64, chunksize=(3, 100)>,
     dask.array<da.random.normal, shape=(200, 300), dtype=float64, chunksize=(100, 100)>,
     dask.array<da.random.normal, shape=(300, 4), dtype=float64, chunksize=(100, 4)>]


    >>> dy = oe.contract("ab,bc,cd", *dxs)  # will infer backend='dask'
    >>> dy
    dask.array<transpose, shape=(3, 4), dtype=float64, chunksize=(3, 4)>

    >>> dy.compute()
    array([[ 470.71404665,    2.44931372,  -28.47577265,  424.37716615],
           [  64.38328345, -287.40753131,  144.46515642,  324.88169821],
           [-142.07153553, -180.41739259,  125.0973783 , -239.16754541]])


In this case, dask arrays in = dask array out, since dask arrays have a shape
attribute, and ``opt_einsum`` can find ``dask.array.tensordot`` and
``dask.array.transpose``.


Sparse
------

The `sparse <https://sparse.pydata.org/>`_ library also fits the bill and is
supported. An example:

.. code-block:: python

    >>> import opt_einsum as oe
    >>> import sparse as sp
    >>> shapes = (3, 200), (200, 300), (300, 4)
    >>> sxs = [sp.random(shp) for shp in shapes]
    [<COO: shape=(3, 200), dtype=float64, nnz=6, sorted=False, duplicates=True>,
     <COO: shape=(200, 300), dtype=float64, nnz=600, sorted=False, duplicates=True>,
     <COO: shape=(300, 4), dtype=float64, nnz=12, sorted=False, duplicates=True>]

    >>> sy = oe.contract("ab,bc,cd", *sxs)
    <COO: shape=(3, 4), dtype=float64, nnz=0, sorted=False, duplicates=False>




Special (GPU) backends for numpy arrays
=======================================

A special case is if you want to supply numpy arrays and get numpy arrays back,
but use a different backend, such as performing a contraction on a GPU.
Unless the specified backend works on numpy arrays this requires converting to
and from the backend array type. Currently ``opt_einsum`` can handle this
automatically for:

    - `tensorflow <https://www.tensorflow.org/>`_
    - `theano <http://deeplearning.net/software/theano/>`_
    - `cupy <https://cupy.chainer.org/>`_
    - `pytorch <https://pytorch.org>`_

which all offer GPU support. Since ``tensorflow`` and ``theano`` both require
compiling the expression, this functionality is encapsulated in generating a
:class:`~opt_einsum.ContractExpression` using
:func:`~opt_einsum.contract_expression`, which can then be called using numpy
arrays whilst specifiying ``backend='tensorflow'`` etc.
Additionally, if arrays are marked as ``constant``
(see :ref:`constants-section`), then these arrays will be kept on the device
for optimal performance.


Theano
------

If ``theano`` is installed, using it as backend is as simple as specifiying
``backend='theano'``:

.. code-block:: python

    >>> import opt_einsum as oe
    >>> shapes = (3, 200), (200, 300), (300, 4)
    >>> expr = oe.contract_expression("ab,bc,cd", *shapes)
    >>> expr
    <ContractExpression('ab,bc,cd')>

    >>> import numpy as np
    >>> # GPU advantage mainly for low precision numbers
    >>> xs = [np.random.randn(*shp).astype(np.float32) for shp in shapes]
    >>> expr(*xs, backend='theano')  # might see some fluff on first run
    ...
    array([[ 129.28352  , -128.00702  , -164.62917  , -335.11682  ],
           [-462.52344  , -121.12657  ,  -67.847626 ,  624.5457   ],
           [   5.2838974,   36.441578 ,   81.62851  ,  703.1576   ]],
          dtype=float32)

Note that you can still supply ``theano.tensor.TensorType`` directly to
``opt_einsum`` (with ``backend='theano'``), and it will return the
relevant ``theano`` type.


Tensorflow
----------

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

Note that you can still supply this expression with, for example, a
``tensorflow.placeholder`` using ``backend='tensorflow'``, and then no
conversion would take place, instead you'd get a ``tensorflow.Tensor`` back.

Version 1.9 of tensorflow also added support for eager execution of
computations. If compilation of the contraction expression tensorflow graph is
taking a substantial amount of time up then it can be advantageous to use this,
especially since tensor contractions are quite compute-bound. This is achieved
by running the following snippet:


.. code-block:: python

  import tensorflow as tf
  tf.enable_eager_execution()

After which ``opt_einsum`` will automatically detect eager mode if
``backend='tensorflow'`` is supplied to a
:class:`~opt_einsum.ContractExpression`.


Pytorch & Cupy
--------------

Both `pytorch <https://pytorch.org>`_ and `cupy <https://cupy.chainer.org/>`_
offer numpy-like, GPU-enabled arrays which execute eagerly rather than
requiring any compilation. If they are installed, no steps are required to
utilize them other than specifiying the ``backend`` keyword:

.. code-block:: python

    >>> expr(*xs, backend='torch')
    array([[ 129.28357  , -128.00684  , -164.62903  , -335.1167   ],
           [-462.52362  , -121.12659  ,  -67.84769  ,  624.5455   ],
           [   5.2839584,   36.44155  ,   81.62852  ,  703.15784  ]],
          dtype=float32)

    >>> expr(*xs, backend='cupy')
    array([[ 129.28357  , -128.00684  , -164.62903  , -335.1167   ],
           [-462.52362  , -121.12659  ,  -67.84769  ,  624.5455   ],
           [   5.2839584,   36.44155  ,   81.62852  ,  703.15784  ]],
          dtype=float32)

And as with the other GPU backends, if raw ``cupy`` or ``pytorch`` arrays are
supplied the returned array will be of the same type, with no conversion
to or from ``numpy`` arrays.
