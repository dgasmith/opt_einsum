import contextlib
import functools
import numbers
from collections import OrderedDict

from .backends.dispatch import get_func
from .parser import get_symbol

_SHARING_STACK = []


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
        inputs, output = equation.split('->')
        inputs = inputs.split(',')
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


def get_shared_func(func, backend='numpy'):
    """Outside of a ``shared_intermediates`` context, this returns
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
