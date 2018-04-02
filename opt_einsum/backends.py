"""
Performing array conversions and operations with various backends.
"""

import importlib

import numpy

# known non top-level imports
_aliases = {'dask': 'dask.array', 'theano': 'theano.tensor'}


def _import_func(func, backend):
    """Try and import ``{backend}.{func}``, raise an error if library is
    installed but can't find that particular function.
    """
    try:
        lib = importlib.import_module(_aliases.get(backend, backend))
        return getattr(lib, func)
    except AttributeError:
        raise AttributeError("{} doesn't seem to provide the function {}".format(backend, func))


# manually cache functions as python2 doesn't support functools.lru_cache
#     other libs will be added to this if needed, but pre-populate with numpy
_cached_funcs = {
    ('tensordot', 'numpy'): numpy.tensordot,
    ('transpose', 'numpy'): numpy.transpose,
    ('einsum', 'numpy'): numpy.einsum,
}


def get_func(func, backend='numpy'):
    """Return ``{backend}.{func}``, e.g. ``numpy.einsum``, cache result.
    """
    try:
        return _cached_funcs[func, backend]
    except KeyError:
        fn = _import_func(func, backend)
        _cached_funcs[func, backend] = fn
        return fn


# mark libs with einsum, else try to use tensordot/tranpose as much as possible
_has_einsum = {}


def has_einsum(backend):
    """Check if ``{backend}.einsum`` exists, cache result for performance.
    """
    try:
        return _has_einsum[backend]
    except KeyError:
        try:
            get_func('einsum', backend)
            _has_einsum[backend] = True
        except AttributeError:
            _has_einsum[backend] = False

        return _has_einsum[backend]


# Tensorflow

def convert_arrays_to_tensorflow(arrays):
    """Convert numpy arrays to ``tensorflow.placeholder`` instances.
    """
    import tensorflow
    return [tensorflow.placeholder(x.dtype, x.shape) for x in arrays]


def build_tensorflow_expression(arrays, expr):
    """Build a tensorflow function based on ``arrays`` and ``expr``.
    """
    import tensorflow
    placeholders = convert_arrays_to_tensorflow(arrays)
    graph = expr._normal_contract(placeholders, backend='tensorflow')

    def tensorflow_contract(*arrays):
        session = tensorflow.get_default_session()
        return session.run(graph, feed_dict=dict(zip(placeholders, arrays)))

    return tensorflow_contract


# Theano

def convert_arrays_to_theano(arrays):
    """Convert numpy arrays to ``theano.tensor.TensorType`` instances.
    """
    import theano
    return [theano.tensor.TensorType(dtype=x.dtype, broadcastable=[False] * len(x.shape))() for x in arrays]


def build_theano_expression(arrays, expr):
    """Build a theano function based on ``arrays`` and ``expr``.
    """
    import theano
    in_vars = convert_arrays_to_theano(arrays)
    out_var = expr._normal_contract(in_vars, backend='theano')
    graph = theano.function(in_vars, out_var)

    def theano_contract(*arrays):
        return graph(*arrays)

    return theano_contract


# Cupy

def convert_arrays_to_cupy(arrays):  # pragma: no cover
    """Convert numpy arrays to ``cupy.ndarray`` instances.
    """
    import cupy
    return [cupy.asarray(x) for x in arrays]


def build_cupy_expression(_, expr):  # pragma: no cover
    """Build a cupy function based on ``arrays`` and ``expr``.
    """
    import cupy

    def cupy_contract(*arrays):
        cupy_arrays = convert_arrays_to_cupy(arrays)
        cupy_out = expr._normal_contract(cupy_arrays, backend='cupy')
        return cupy.asnumpy(cupy_out)

    return cupy_contract


# Dispatch to correct expression backend
#    these are the backends which support explicit to-and-from numpy conversion
CONVERT_BACKENDS = {
    'tensorflow': build_tensorflow_expression,
    'theano': build_theano_expression,
    'cupy': build_cupy_expression,
}


def build_expression(backend, arrays, expr):
    """Build an expression, based on ``expr`` and initial arrays ``arrays``,
    that evaluates using backend ``backend``.
    """
    return CONVERT_BACKENDS[backend](arrays, expr)
