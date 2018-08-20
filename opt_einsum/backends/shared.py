import contextlib
import numbers
from collections import OrderedDict

from ..parser import get_symbol
from .dispatch import get_func

_SHARING_STACK = []
_CURRENT_BACKEND = None


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


@contextlib.contextmanager
def handle_sharing(backend):
    global _CURRENT_BACKEND
    if _SHARING_STACK and _CURRENT_BACKEND is None:
        try:
            _CURRENT_BACKEND = backend
            yield __name__
        finally:
            _CURRENT_BACKEND = None
    elif backend == __name__ and _CURRENT_BACKEND is None:
        raise ValueError('shared backend is available only via shared_intermediates')
    else:
        yield backend


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


def transpose(a, axes):
    _save_tensors(a)

    # hash by axes
    axes = tuple(axes)
    key = 'transpose', _CURRENT_BACKEND, id(a), axes

    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]

    result = get_func('transpose', _CURRENT_BACKEND)(a, axes)
    cache[key] = result
    return result


def tensordot(x, y, axes=2):
    _save_tensors(x, y)

    # hash based on the (axes_x,axes_y) form of axes
    if isinstance(axes, numbers.Number):
        axes = list(range(len(x.shape)))[len(x.shape) - axes:], list(range(len(y.shape)))[:axes]
    axes = tuple(axes[0]), tuple(axes[1])
    key = 'tensordot', _CURRENT_BACKEND, id(x), id(y), axes

    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]

    result = get_func('tensordot', _CURRENT_BACKEND)(x, y, axes)
    cache[key] = result
    return result


def einsum(equation, *operands):
    _save_tensors(*operands)

    # hash modulo commutativity by computing a canonical ordering and naming
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    canonical = sorted(zip(inputs, map(id, operands)), key=lambda x: x[1])
    canonical_ids = tuple(id_ for _, id_ in canonical)
    canonical_inputs = ','.join(input_ for input_, _ in canonical)
    canonical_equation = _alpha_canonicalize('{}->{}'.format(canonical_inputs, output))
    key = 'einsum', _CURRENT_BACKEND, canonical_equation, canonical_ids

    cache = _SHARING_STACK[-1]
    if key in cache:
        return cache[key]

    result = get_func('einsum', _CURRENT_BACKEND)(equation, *operands)
    cache[key] = result
    return result
