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

def to_tensorflow(array, constant=False):
    """Convert a numpy array to a ``tensorflow.placeholder`` instance.
    """
    import tensorflow

    if isinstance(array, numpy.ndarray):
        if constant:
            return tensorflow.constant(array, array.dtype, array.shape)

        return tensorflow.placeholder(array.dtype, array.shape)

    return array


def build_tensorflow_expression(arrays, expr):
    """Build a tensorflow function based on ``arrays`` and ``expr``.
    """
    import tensorflow

    placeholders = [to_tensorflow(array) for array in arrays]
    graph = expr._contract(placeholders, backend='tensorflow')

    def tensorflow_contract(*arrays):
        session = tensorflow.get_default_session()
        # only want to feed placeholders - constant tensors already have values
        feed_dict = {p: a for p, a in zip(placeholders, arrays) if p.op.type == 'Placeholder'}
        return session.run(graph, feed_dict=feed_dict)

    return tensorflow_contract


def parse_constants_tensorflow(const_arrays, expr):
    """Convert constant arguments to tensorflow constants, and perform any
    possible constant contractions. Requires evaluating a tensorflow graph.
    """
    import tensorflow

    # compute the partial graph of new inputs
    const_arrays = [to_tensorflow(x, constant=True) for x in const_arrays]
    new_ops, new_contraction_list = expr(*const_arrays, backend='tensorflow', parse_constants=True)

    # evaluate the new inputs and convert to tensorflow constants
    session = tensorflow.get_default_session()
    new_ops = [None if x is None else to_tensorflow(session.run(x), constant=True) for x in new_ops]

    return new_ops, new_contraction_list


# Theano

def to_theano(array, constant=False):
    """Convert a numpy array to ``theano.tensor.TensorType`` instance.
    """
    import theano

    if isinstance(array, numpy.ndarray):
        if constant:
            return theano.tensor.constant(array)

        return theano.tensor.TensorType(dtype=array.dtype, broadcastable=[False] * len(array.shape))()

    return array


def build_theano_expression(arrays, expr):
    """Build a theano function based on ``arrays`` and ``expr``.
    """
    import theano

    in_vars = [to_theano(array) for array in arrays]
    out_var = expr._contract(in_vars, backend='theano')

    # don't supply constants to graph
    graph_ins = [x for x in in_vars if not isinstance(x, theano.tensor.TensorConstant)]
    graph = theano.function(graph_ins, out_var)

    def theano_contract(*arrays):
        return graph(*[x for x in arrays if not isinstance(x, theano.tensor.TensorConstant)])

    return theano_contract


def parse_constants_theano(const_arrays, expr):
    # compute the partial graph of new inputs
    const_arrays = [to_theano(x, constant=True) for x in const_arrays]
    new_ops, new_contraction_list = expr(*const_arrays, backend='theano', parse_constants=True)

    # evaluate the new inputs and convert to theano shared tensors
    new_ops = [None if x is None else to_theano(x.eval(), constant=True) for x in new_ops]

    return new_ops, new_contraction_list


# Cupy

def to_cupy(array):  # pragma: no cover
    import cupy

    if isinstance(array, numpy.ndarray):
        return cupy.asarray(array)

    return array


def build_cupy_expression(_, expr):  # pragma: no cover
    """Build a cupy function based on ``arrays`` and ``expr``.
    """

    def cupy_contract(*arrays):
        cupy_arrays = [to_cupy(x) for x in arrays]
        cupy_out = expr._contract(cupy_arrays, backend='cupy')
        return cupy_out.get()

    return cupy_contract


def parse_constants_cupy(const_arrays, expr):  # pragma: no cover
    """Convert constant arguments to cupy arrays, and perform any possible
    constant contractions.
    """
    const_arrays = [to_cupy(x) for x in const_arrays]
    return expr(*const_arrays, backend='cupy', parse_constants=True)


# Dispatch to correct expression backend
#    these are the backends which support explicit to-and-from numpy conversion
CONVERT_BACKENDS = {
    'tensorflow': build_tensorflow_expression,
    'theano': build_theano_expression,
    'cupy': build_cupy_expression,
}


PARSE_CONSTS_BACKENDS = {
    'tensorflow': parse_constants_tensorflow,
    'theano': parse_constants_theano,
    'cupy': parse_constants_cupy,
}


def build_expression(backend, arrays, expr):
    """Build an expression, based on ``expr`` and initial arrays ``arrays``,
    that evaluates using backend ``backend``.
    """
    return CONVERT_BACKENDS[backend](arrays, expr)


def parse_constants(backend, arrays, expr):
    """Convert constant arrays to the correct backend, and perform as much of
    the contraction of ``expr`` with these as possible.
    """
    return PARSE_CONSTS_BACKENDS[backend](arrays, expr)
