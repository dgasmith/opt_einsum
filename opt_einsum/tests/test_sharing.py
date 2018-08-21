from collections import Counter

import numpy as np
import pytest

from opt_einsum import (contract_expression, contract_path, get_symbol,
                        helpers, shared_intermediates)
from opt_einsum.sharing import count_cached_ops

try:
    import cupy
    cupy_if_found = 'cupy'
except ImportError:
    cupy_if_found = pytest.param('cupy', marks=[pytest.mark.skip(reason="CuPy not installed.")])

try:
    import torch
    torch_if_found = 'torch'
except ImportError:
    torch_if_found = pytest.param('torch', marks=[pytest.mark.skip(reason="PyTorch not installed.")])

backends = ['numpy', torch_if_found, cupy_if_found]


@pytest.mark.parametrize('backend', backends)
def test_sharing_value(backend):
    eq = 'abc,bcd,cde,def->af'
    views = helpers.build_views(eq)
    shapes = [v.shape for v in views]
    expr = contract_expression(eq, *shapes)

    expected = expr(*views, backend=backend)
    with shared_intermediates():
        actual = expr(*views, backend=backend)

    assert (actual == expected).all()


@pytest.mark.parametrize('backend', backends)
def test_complete_sharing(backend):
    eq = 'ab,bc,cd->'
    views = helpers.build_views(eq)
    expr = contract_expression(eq, *(v.shape for v in views))

    print('-' * 40)
    print('Without sharing:')
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expected = count_cached_ops(cache)

    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expr(*views, backend=backend)
        actual = count_cached_ops(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(expected))
    print('With sharing: {} expressions'.format(actual))
    assert actual == expected


@pytest.mark.parametrize('backend', backends)
def test_partial_sharing(backend):
    eq = 'ab,bc,de->'
    x, y, z1 = helpers.build_views(eq)
    z2 = 2.0 * z1 - 1.0
    expr = contract_expression(eq, x.shape, y.shape, z1.shape)

    print('-' * 40)
    print('Without sharing:')
    num_exprs_nosharing = Counter()
    with shared_intermediates() as cache:
        expr(x, y, z1, backend=backend)
        num_exprs_nosharing.update(count_cached_ops(cache))
    with shared_intermediates() as cache:
        expr(x, y, z2, backend=backend)
        num_exprs_nosharing.update(count_cached_ops(cache))

    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        expr(x, y, z1, backend=backend)
        expr(x, y, z2, backend=backend)
        num_exprs_sharing = count_cached_ops(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(num_exprs_nosharing))
    print('With sharing: {} expressions'.format(num_exprs_sharing))
    assert num_exprs_nosharing['einsum'] > num_exprs_sharing['einsum']


def compute_cost(cache):
    return sum(1 for key in cache.keys() if key[0] in ('einsum', 'tensordot'))


@pytest.mark.parametrize('size', [3, 4, 5])
@pytest.mark.parametrize('backend', backends)
def test_chain(size, backend):
    xs = [np.random.rand(2, 2) for _ in range(size)]
    shapes = [x.shape for x in xs]
    alphabet = ''.join(get_symbol(i) for i in range(size + 1))
    names = [alphabet[i:i+2] for i in range(size)]
    inputs = ','.join(names)

    with shared_intermediates():
        print(inputs)
        for i in range(size + 1):
            target = alphabet[i]
            eq = '{}->{}'.format(inputs, target)
            path_info = contract_path(eq, *xs)
            print(path_info[1])
            expr = contract_expression(eq, *shapes)
            expr(*xs, backend=backend)
        print('-' * 40)


@pytest.mark.parametrize('size', [3, 4, 5, 10])
@pytest.mark.parametrize('backend', backends)
def test_chain_2(size, backend):
    xs = [np.random.rand(2, 2) for _ in range(size)]
    shapes = [x.shape for x in xs]
    alphabet = ''.join(get_symbol(i) for i in range(size + 1))
    names = [alphabet[i:i+2] for i in range(size)]
    inputs = ','.join(names)

    with shared_intermediates():
        print(inputs)
        for i in range(size):
            target = alphabet[i:i+2]
            eq = '{}->{}'.format(inputs, target)
            path_info = contract_path(eq, *xs)
            print(path_info[1])
            expr = contract_expression(eq, *shapes)
            expr(*xs, backend=backend)
        print('-' * 40)


@pytest.mark.parametrize('backend', backends)
def test_chain_2_growth(backend):
    sizes = list(range(1, 21))
    costs = []
    for size in sizes:
        xs = [np.random.rand(2, 2) for _ in range(size)]
        alphabet = ''.join(get_symbol(i) for i in range(size + 1))
        names = [alphabet[i:i+2] for i in range(size)]
        inputs = ','.join(names)

        with shared_intermediates() as cache:
            for i in range(size):
                target = alphabet[i:i+2]
                eq = '{}->{}'.format(inputs, target)
                expr = contract_expression(eq, *(x.shape for x in xs))
                expr(*xs, backend=backend)
            costs.append(compute_cost(cache))

    print('sizes = {}'.format(repr(sizes)))
    print('costs = {}'.format(repr(costs)))
    for size, cost in zip(sizes, costs):
        print('{}\t{}'.format(size, cost))


@pytest.mark.parametrize('size', [3, 4, 5])
@pytest.mark.parametrize('backend', backends)
def test_chain_sharing(size, backend):
    xs = [np.random.rand(2, 2) for _ in range(size)]
    alphabet = ''.join(get_symbol(i) for i in range(size + 1))
    names = [alphabet[i:i+2] for i in range(size)]
    inputs = ','.join(names)

    num_exprs_nosharing = 0
    for i in range(size + 1):
        with shared_intermediates() as cache:
            target = alphabet[i]
            eq = '{}->{}'.format(inputs, target)
            expr = contract_expression(eq, *(x.shape for x in xs))
            expr(*xs, backend=backend)
            num_exprs_nosharing += compute_cost(cache)

    with shared_intermediates() as cache:
        print(inputs)
        for i in range(size + 1):
            target = alphabet[i]
            eq = '{}->{}'.format(inputs, target)
            path_info = contract_path(eq, *xs)
            print(path_info[1])
            expr = contract_expression(eq, *(x.shape for x in xs))
            expr(*xs, backend=backend)
        num_exprs_sharing = compute_cost(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(num_exprs_nosharing))
    print('With sharing: {} expressions'.format(num_exprs_sharing))
    assert num_exprs_nosharing > num_exprs_sharing
