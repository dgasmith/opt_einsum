import contextlib
import functools
import numbers
from collections import Counter, OrderedDict

from .parser import get_symbol, parse_einsum_input

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


def _memoize(key, fn, *args, **kwargs):
    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]
    result = fn(*args, **kwargs)
    cache[key] = result
    return result


def transpose_cache_wrap(transpose):

    @functools.wraps(transpose)
    def cached_transpose(a, axes, backend='numpy'):
        if not _SHARING_STACK:
            return transpose(a, axes, backend=backend)

        # hash by axes
        _save_tensors(a)
        axes = tuple(axes)
        key = 'transpose', backend, id(a), axes
        return _memoize(key, transpose, a, axes, backend=backend)

    return cached_transpose


def tensordot_cache_wrap(tensordot):

    @functools.wraps(tensordot)
    def cached_tensordot(x, y, axes=2, backend='numpy'):
        if not _SHARING_STACK:
            return tensordot(x, y, axes, backend=backend)

        # hash based on the (axes_x,axes_y) form of axes
        _save_tensors(x, y)
        if isinstance(axes, numbers.Number):
            axes = list(range(len(x.shape)))[len(x.shape) - axes:], list(range(len(y.shape)))[:axes]
        axes = tuple(axes[0]), tuple(axes[1])
        key = 'tensordot', backend, id(x), id(y), axes
        return _memoize(key, tensordot, x, y, axes, backend=backend)

    return cached_tensordot


def einsum_cache_wrap(einsum):

    @functools.wraps(einsum)
    def cached_einsum(*args, **kwargs):
        if not _SHARING_STACK:
            return einsum(*args, **kwargs)

        # hash modulo commutativity by computing a canonical ordering and names
        backend = kwargs.pop('backend', 'numpy')
        equation = args[0]
        inputs, output, operands = parse_einsum_input(args)
        inputs = inputs.split(',')
        _save_tensors(*operands)
        canonical = sorted(zip(inputs, map(id, operands)), key=lambda x: x[1])
        canonical_ids = tuple(id_ for _, id_ in canonical)
        canonical_inputs = ','.join(input_ for input_, _ in canonical)
        canonical_equation = _alpha_canonicalize('{}->{}'.format(canonical_inputs, output))
        key = 'einsum', backend, canonical_equation, canonical_ids
        return _memoize(key, einsum, equation, *operands, backend=backend)

    return cached_einsum


def to_backend_cache_wrap(to_backend):

    @functools.wraps(to_backend)
    def cached_to_backend(array):
        if not _SHARING_STACK:
            return to_backend(array)

        # hash by id
        key = to_backend.__name__, id(array)
        return _memoize(key, to_backend, array)

    return cached_to_backend
