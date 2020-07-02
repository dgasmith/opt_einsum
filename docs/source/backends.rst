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

``opt_einsum`` is agnostic to the type of n-dimensional arrays (tensors)
it uses, since finding the contraction path only relies on getting the shape
attribute of each array supplied.
It can perform the underlying tensor contractions with various
libraries. In fact, any library that provides a :func:`~numpy.tensordot` and
:func:`~numpy.transpose` implementation can perform most normal contractions.
While more special functionality such as axes reduction is reliant on a
:func:`~numpy.einsum` implementation.

.. note::

    For a contraction to be possible without using a backend einsum, it must
    satisfy the following rule: in the full expression (*including* output
    indices) each index must appear twice. In other words, each dimension
    must be contracted with one other dimension, or left alone.


Backend agnostic contractions
=============================

The automatic backend detection will be detected based on the first supplied
array (default), this can be overridden by specifying the correct ``backend``
argument for the type of arrays supplied when calling
:func:`~opt_einsum.contract`. For example, if you had a library installed
called ``'foo'`` which provided an :class:`~numpy.ndarray` like object with a
``.shape`` attribute as well as ``foo.tensordot`` and ``foo.transpose`` then
you could contract them with something like:

.. code-block:: python

    contract(einsum_str, *foo_arrays, backend='foo')

Behind the scenes :mod:`opt_einsum` will find the contraction path, perform
pairwise contractions using e.g. ``foo.tensordot`` and finally return the canonical
type those functions return.

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

The `sparse <https://sparse.pydata.org/>`_ library also fits the requirements and is
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
    array(4.90422159)

    >>> # wrap foo with autograd to compute gradients instead
    >>> dfoo = autograd.grad(foo)
    >>> dx, dy, dz = dfoo(arrays)
    >>> dx, dy, dz
    (array([[1.10056194, 1.25078356, 1.48211494],
            [1.38945961, 1.5572077 , 1.65234003]]),
     array([[0.41710717, 0.63202881, 0.84573502, 0.95069975],
            [0.42706777, 0.73630994, 0.99328938, 0.77415267],
            [0.40773334, 0.61693475, 0.82545726, 0.93132302]]),
     array([[0.78747828, 1.28979012],
            [1.26051133, 1.48835538],
            [0.46896666, 0.55003072],
            [1.10840828, 1.16722494]]))

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
    DeviceArray(4.9042215, dtype=float32)

    >>> # generate a compiled version of the gradient function
    >>> jit_dfoo = jax.jit(jax.grad(foo))
    >>> jit_dfoo([x, y, z])
    [DeviceArray([[1.10056198, 1.25078356, 1.48211491],
                  [1.38945973, 1.5572077, 1.65234005]], dtype=float32),
     DeviceArray([[0.41710716, 0.63202882, 0.84573501, 0.95069975],
                  [0.42706776, 0.73630995, 0.99328935, 0.7741527 ],
                  [0.40773335, 0.61693472, 0.82545722, 0.93132305]],
                 dtype=float32),
     DeviceArray([[0.78747827, 1.28979015],
                  [1.2605114 , 1.4883554 ],
                  [0.46896666, 0.55003077],
                  [1.10840821, 1.16722488]], dtype=float32)]

.. note::

    ``jax`` defaults to converting all arrays to single precision. This
    behaviour can be changed by running
    ``from jax.config import config; config.update("jax_enable_x64", True)``
    **before** it has been imported and used at all.



Special (GPU) backends for numpy arrays
=======================================

A particular case is if numpy arrays are required for the input and output,
however, a more performant backend is required such as performing the contraction on a GPU.
Unless the specified backend works on numpy arrays, this requires converting to
and from the backend array type. Currently ``opt_einsum`` can handle this
automatically for:

    - `tensorflow <https://www.tensorflow.org/>`_
    - `theano <http://deeplearning.net/software/theano/>`_
    - `cupy <https://cupy.chainer.org/>`_
    - `pytorch <https://pytorch.org>`_
    - `jax <https://github.com/google/jax>`_

all of which offer GPU support. Since ``tensorflow`` and ``theano`` both require
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


Contracting arbitrary objects
=============================

There is one more explicit backend that can handle arbitrary arrays of objects,
so long the *objects themselves* just support multiplication and addition (
``__mul__`` and ``__add__`` dunder methods respectively). Use it by supplying
``backend='object'``.

For example, imagine we want to perform a contraction of arrays made up of
`sympy <www.sympy.org>`_ symbols:

.. code-block:: python

  >>> import opt_einsum as oe
  >>> import numpy as np
  >>> import sympy

  >>> # define the symbols
  >>> a, b, c, d, e, f, g, h, i, j, k, l = [sympy.symbols(oe.get_symbol(i)) for i in range(12)]
  >>> a * b + c * d
  𝑎𝑏+𝑐𝑑

  >>> # define the tensors (you might explicitly specify ``dtype=object``)
  >>> X = np.array([[a, b], [c, d]])
  >>> Y = np.array([[e, f], [g, h]])
  >>> Z = np.array([[i, j], [k, l]])

  >>> # contract the tensors!
  >>> oe.contract('uv,vw,wu->u', X, Y, Z, backend='object')
  array([i*(a*e + b*g) + k*(a*f + b*h), j*(c*e + d*g) + l*(c*f + d*h)],
        dtype=object)

There are a few things to note here:

* The returned array is a ``numpy.ndarray`` but since it has ``dtype=object``
  it can really hold *any* python objects
* We had to explicitly use ``backend='object'``, since :func:`numpy.einsum`
  would have otherwise been dispatched to, which can't handle ``dtype=object``
  (though :func:`numpy.tensordot` in fact can)
* Although an optimized pairwise contraction order is used, the looping in each
  single contraction is **performed in python so performance will be
  drastically lower than for numeric dtypes!**
