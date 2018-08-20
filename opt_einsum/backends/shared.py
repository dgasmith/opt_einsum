import contextlib
import numbers
from collections import OrderedDict

from ..parser import get_symbol
from .dispatch import get_func

_SHARING_STACK = []
_CURRENT_BACKEND = []


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
    _SHARING_STACK.append(cache)
    yield cache
    _SHARING_STACK.pop()


@contextlib.contextmanager
def handle_sharing(backend):
    if _SHARING_STACK and not _CURRENT_BACKEND:
        _CURRENT_BACKEND.append(backend)
        yield 'opt_einsum.backends.shared'
        _CURRENT_BACKEND.pop()
    if backend == 'opt_einsum.backends.shared' and not _CURRENT_BACKEND:
        raise ValueError('shared backend is available only via shared_intermediates')
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


def transpose(a, axes):
    backend = _CURRENT_BACKEND[0]
    cache = _SHARING_STACK[-1]
    cache['tensor', id(a)] = a

    axes = tuple(axes)
    key = 'transpose', backend, id(a), axes
    if key in cache:
        return cache[key]

    result = get_func('transpose', backend)(a, axes)
    cache[key] = result
    return result


def tensordot(x, y, axes=2):
    backend = _CURRENT_BACKEND[0]
    cache = _SHARING_STACK[-1]
    cache['tensor', id(x)] = x
    cache['tensor', id(y)] = y

    if isinstance(axes, numbers.Number):
        axes = list(range(len(x.shape)))[len(x.shape) - axes:], list(range(len(y.shape)))[:axes]
    axes = tuple(axes[0]), tuple(axes[1])
    key = 'tensordot', backend, id(x), id(y), axes
    if key in cache:
        return cache[key]

    result = get_func('tensordot', backend)(x, y, axes)
    cache[key] = result
    return result


def einsum(equation, *operands):
    backend = _CURRENT_BACKEND[0]
    cache = _SHARING_STACK[-1]
    for d in operands:
        cache['tensor', id(d)] = d

    # compute a canonical hash, modulo commutativity
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    canonical = sorted(zip(inputs, operands), key=lambda x: id(x[1]))
    canonical_inputs = ','.join(input_ for input_, _ in canonical)
    canonical_equation = _alpha_canonicalize('{}->{}'.format(canonical_inputs, output))
    canonical_operands = tuple(d for _, d in canonical)
    key = 'einsum', backend, canonical_equation, tuple(map(id, canonical_operands))
    if key in cache:
        return cache[key]

    result = get_func('einsum', backend)(equation, *operands)
    cache[key] = result
    return result
