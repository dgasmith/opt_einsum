======================
Backends & GPU Support
======================

``opt_einsum`` is quite agnostic to the type of n-dimensional arrays (tensors)
it uses, since finding the contraction path only relies on getting the shape
attribute of each array supplied.
It can perform the underlying tensor contractions with various
libraries. In fact, any library that provides a :func:`~numpy.tensordot` and
:func:`~numpy.transpose` implementation can perform most normal contractions.
While more special functionality such as axes reduction is reliant on a
:func:`~numpy.einsum` implementation.
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
    - `autograd <https://github.com/HIPS/autograd>`_: automatic derivative
      computation for tensor expressions
    - `jax <https://github.com/google/jax>`_: compiled GPU tensor expressions
      including ``autograd``-like functionality

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

    >>> import sparse as sp
    >>> shapes = (3, 200), (200, 300), (300, 4)
    >>> sxs = [sp.random(shp) for shp in shapes]
    [<COO: shape=(3, 200), dtype=float64, nnz=6, sorted=False, duplicates=True>,
     <COO: shape=(200, 300), dtype=float64, nnz=600, sorted=False, duplicates=True>,
     <COO: shape=(300, 4), dtype=float64, nnz=12, sorted=False, duplicates=True>]

    >>> sy = oe.contract("ab,bc,cd", *sxs)
    <COO: shape=(3, 4), dtype=float64, nnz=0, sorted=False, duplicates=False>


Autograd
--------

The `autograd <https://github.com/HIPS/autograd>`_ library is a drop-in for
``numpy`` that can automatically compute the gradients of array expressions.
``opt_einsum`` automatically dispatches the ``autograd`` arrays correctly,
enabling a simple way to compute gradients of tensor contractions:

.. code-block:: python

    >>> import numpy as np
    >>> import autograd
    >>> shapes = [(2, 3), (3, 4), (4, 2)]
    >>> x, y, z = [np.random.rand(*s) for s in shapes]

    >>> # make single arg function as autograd takes derivative of first arg
    >>> def foo(xyz):
    ...    return oe.contract('ij,jk,ki->', *xyz)

    >>> foo([x, y, z])
    array(3.71251519)

    >>> # wrap foo with autograd to compute gradients instead
    >>> dfoo = autograd.grad(foo)
    >>> dx, dy, dz = dfoo(arrays)
    >>> dx, dy, dz
    (array([[1.30857301, 1.87061563, 0.93695379],
            [1.0665825 , 1.5318835 , 0.71335157]]),
     array([[0.52377733, 0.76218852, 0.18797777, 0.60494682],
            [0.23560851, 0.41939902, 0.05703505, 0.2871758 ],
            [0.81113451, 1.09167635, 0.32298711, 0.91939656]]),
     array([[0.5422258 , 1.04966071],
            [0.29421714, 0.55134473],
            [0.62781637, 0.93304068],
            [0.63050881, 0.92410755]]))

Jax
---

`jax <https://github.com/google/jax>`_ is itself a drop-in for ``autograd``,
that additionally uses  `XLA <https://www.tensorflow.org/xla>`_ to compile the
expressions, particularly for the GPU. Using it with ``opt_einsum`` is very
simple:

.. code-block:: python

    >>> import jax
    >>> # generate a compiled version of the above function
    >>> jit_foo = jax.jit(foo)
    >>> jit_foo([x, y, z])
    DeviceArray(3.7125154, dtype=float32)

    >>> # generate a compiled version of the gradient function
    >>> jit_dfoo = jax.jit(jax.grad(foo))
    >>> jit_dfoo([x, y, z])
    [DeviceArray([[1.1137383 , 1.14972878, 0.64056885],
                  [1.61149812, 1.46658325, 0.71612591]], dtype=float32),
     DeviceArray([[0.40407446, 0.69419581, 0.28825626, 0.39774451],
                  [0.3415361 , 0.87385571, 0.33857098, 0.50946677],
                  [0.70923012, 1.31801975, 0.53886849, 0.75821561]],
                 dtype=float32),
     DeviceArray([[0.54016989, 1.04018342],
                  [0.62524962, 1.11105669],
                  [0.50767434, 1.35125792],
                  [0.81190646, 1.3016566 ]], dtype=float32)]

.. note::

    ``jax`` defaults to converting all arrays to single precision. This
    behaviour can be changed by running
    ``from jax.config import config; config.update("jax_enable_x64", True)``
    **before** it has been imported and used at all.



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
    - `jax <https://github.com/google/jax>`_

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

Jax
---

`jax <https://github.com/google/jax>`_, as introduced above, can compile tensor
functions, in doing so often achieving better performance.
``opt_einsum`` expressions can handle this behind the scenes,
so again just the ``backend`` keyword needs to be supplied:

.. code-block:: python

    >>> expr(*xs, backend='jax')
    array([[ 129.28357  , -128.00684  , -164.62903  , -335.1167   ],
           [-462.52362  , -121.12659  ,  -67.84769  ,  624.5455   ],
           [   5.2839584,   36.44155  ,   81.62852  ,  703.15784  ]],
          dtype=float32)
