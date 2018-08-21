import contextlib
import functools
import numbers
from collections import Counter, OrderedDict

from .backends import cupy as _cupy
from .backends import torch as _torch
from .backends.dispatch import CONVERT_BACKENDS, build_expression, get_func
from .parser import get_symbol

_SHARING_STACK = []


def parse_equation(eq):
    """Parses an equation into a list of inputs and an output.
    """
    parts = eq.split('->')
    inputs = parts[0].split(',')
    output = parts[1] if len(parts) == 2 else ''
    return inputs, output


@contextlib.contextmanager
def shared_intermediates(cache=None):
    """Context in which contract intermediate results are shared.

    Note that intermediate computations will not be garbage collected until
    1. this context exits, and
    2. the yielded cache is garbage collected (if it was captured).

    Parameters
    ----------
    cache : dict
        If specified, a user-stored dict in which intermediate results will
        be stored. This can be used to interleave sharing contexts.

    Returns
    -------
    cache : dict
        A dictionary in which sharing results are stored. If ignored,
        sharing results will be garbage collected when this context is
        exited. This dict can be passed to another context to resume
        sharing.
    """
    if cache is None:
        cache = {}
    try:
        _SHARING_STACK.append(cache)
        yield cache
    finally:
        _SHARING_STACK.pop()


def count_cached_ops(cache):
    """Returns a counter of the types of each op in the cache.
    This is useful for profiling to increase sharing.
    """
    return Counter(key[0] for key in cache.keys())


def _alpha_canonicalize(equation):
    """Alpha convert in an order-independent canonical way.
    """
    rename = OrderedDict()
    for name in equation:
        if name in ',->':
            continue
        if name not in rename:
            rename[name] = get_symbol(len(rename))
    return ''.join(rename.get(x, x) for x in equation)


def _save_tensors(*tensors):
    """Save tensors in the cache to prevent their ids from being recycled.
    This is needed to prevent false cache lookups.
    """
    cache = _SHARING_STACK[-1]
    for tensor in tensors:
        cache['tensor', id(tensor)] = tensor


def _memoize(key, fn, *args):
    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]
    result = fn(*args)
    cache[key] = result
    return result


def transpose_cache_wrap(transpose, backend):

    @functools.wraps(transpose)
    def cached_transpose(a, axes):
        # hash by axes
        _save_tensors(a)
        axes = tuple(axes)
        key = 'transpose', backend, id(a), axes
        return _memoize(key, transpose, a, axes)

    return cached_transpose


def tensordot_cache_wrap(tensordot, backend):

    @functools.wraps(tensordot)
    def cached_tensordot(x, y, axes=2):
        # hash based on the (axes_x,axes_y) form of axes
        _save_tensors(x, y)
        if isinstance(axes, numbers.Number):
            axes = list(range(len(x.shape)))[len(x.shape) - axes:], list(range(len(y.shape)))[:axes]
        axes = tuple(axes[0]), tuple(axes[1])
        key = 'tensordot', backend, id(x), id(y), axes
        return _memoize(key, tensordot, x, y, axes)

    return cached_tensordot


def einsum_cache_wrap(einsum, backend):

    @functools.wraps(einsum)
    def cached_einsum(equation, *operands):
        # hash modulo commutativity by computing a canonical ordering and names
        _save_tensors(*operands)
        inputs, output = parse_equation(equation)
        canonical = sorted(zip(inputs, map(id, operands)), key=lambda x: x[1])
        canonical_ids = tuple(id_ for _, id_ in canonical)
        canonical_inputs = ','.join(input_ for input_, _ in canonical)
        canonical_equation = _alpha_canonicalize('{}->{}'.format(canonical_inputs, output))
        key = 'einsum', backend, canonical_equation, canonical_ids
        return _memoize(key, einsum, equation, *operands)

    return cached_einsum


_cache_wrap = {
    'transpose': transpose_cache_wrap,
    'tensordot': tensordot_cache_wrap,
    'einsum': einsum_cache_wrap,
}

_cached_funcs = {}


def get_func_shared(func, backend='numpy'):
    """Outside of any ``shared_intermediates`` context, this returns
    ``get_func(func, backend)``. Inside of a ``shared_intermediates`` context,
    this returns a cached version of that function.
    """
    if not _SHARING_STACK:
        return get_func(func, backend)
    try:
        return _cached_funcs[func, backend]
    except KeyError:
        fn = get_func(func, backend)
        cached_fn = _cache_wrap[func](fn, backend)
        _cached_funcs[func, backend] = cached_fn
        return cached_fn


def to_backend_cache_wrap(to_backend, backend):

    @functools.wraps(to_backend)
    def cached_to_backend(array):
        # hash by id
        key = 'to_backend', backend, id(array)
        return _memoize(key, to_backend, array)

    return cached_to_backend


_to_backend = {
    'torch': to_backend_cache_wrap(_torch.to_torch, 'torch'),
    'cupy': to_backend_cache_wrap(_cupy.to_cupy, 'cupy'),
}


def build_expression_shared(backend, arrays, expr):
    """Outside of any ``shared_intermediates`` context, this returns
    ``build_expression(backend, arrays, expr)``. Inside of a
    ``shared_intermediates`` context, this returns a version of that
    function that caches the ``numpy``-to-backend conversions.
    """
    if not _SHARING_STACK:
        return build_expression(backend, arrays, expr)

    try:
        to_backend = _to_backend[backend]
    except KeyError:
        raise NotImplementedError(
            'Sharing with the {} backend is only supported with manual conversions. '
            'Please convert your numpy arrays to {} format before contracting.'.format(backend))
    return CONVERT_BACKENDS[backend](arrays, expr, to_backend=to_backend)
