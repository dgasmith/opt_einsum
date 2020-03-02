"""
Tests the input parsing for opt_einsum. Duplicates the np.einsum input tests.
"""

import numpy as np
import pytest

from opt_einsum import contract, contract_path


def build_views(string):
    chars = 'abcdefghij'
    sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4])
    sizes = {c: s for c, s in zip(chars, sizes)}

    views = []

    string = string.replace('...', 'ij')

    terms = string.split('->')[0].split(',')
    for term in terms:
        dims = [sizes[x] for x in term]
        views.append(np.random.rand(*dims))
    return views


def test_type_errors():
    # subscripts must be a string
    with pytest.raises(TypeError):
        contract(0, 0)

    # out parameter must be an array
    with pytest.raises(TypeError):
        contract("", 0, out='test')

    # order parameter must be a valid order
    with pytest.raises(TypeError):
        contract("", 0, order='W')

    # casting parameter must be a valid casting
    with pytest.raises(ValueError):
        contract("", 0, casting='blah')

    # dtype parameter must be a valid dtype
    with pytest.raises(TypeError):
        contract("", 0, dtype='bad_data_type')

    # other keyword arguments are rejected
    with pytest.raises(TypeError):
        contract("", 0, bad_arg=0)

    # issue 4528 revealed a segfault with this call
    with pytest.raises(TypeError):
        contract(*(None, ) * 63)

    # Cannot have two ->
    with pytest.raises(ValueError):
        contract("->,->", 0, 5)

    # Undefined symbol lhs
    with pytest.raises(ValueError):
        contract("&,a->", 0, 5)

    # Undefined symbol rhs
    with pytest.raises(ValueError):
        contract("a,a->&", 0, 5)

    with pytest.raises(ValueError):
        contract("a,a->&", 0, 5)

    # Catch ellipsis errors
    string = '...a->...a'
    views = build_views(string)

    # Subscript list must contain Ellipsis or (hashable && comparable) object
    with pytest.raises(TypeError):
        contract(views[0], [Ellipsis, 0], [Ellipsis, ['a']])

    with pytest.raises(TypeError):
        contract(views[0], [Ellipsis, dict()], [Ellipsis, 'a'])


def test_value_errors():
    with pytest.raises(ValueError):
        contract("")

    # subscripts must be a string
    with pytest.raises(TypeError):
        contract(0, 0)

    # invalid subscript character
    with pytest.raises(ValueError):
        contract("i%...", [0, 0])
    with pytest.raises(ValueError):
        contract("...j$", [0, 0])
    with pytest.raises(ValueError):
        contract("i->&", [0, 0])

    with pytest.raises(ValueError):
        contract("")
    # number of operands must match count in subscripts string
    with pytest.raises(ValueError):
        contract("", 0, 0)
    with pytest.raises(ValueError):
        contract(",", 0, [0], [0])
    with pytest.raises(ValueError):
        contract(",", [0])

    # can't have more subscripts than dimensions in the operand
    with pytest.raises(ValueError):
        contract("i", 0)
    with pytest.raises(ValueError):
        contract("ij", [0, 0])
    with pytest.raises(ValueError):
        contract("...i", 0)
    with pytest.raises(ValueError):
        contract("i...j", [0, 0])
    with pytest.raises(ValueError):
        contract("i...", 0)
    with pytest.raises(ValueError):
        contract("ij...", [0, 0])

    # invalid ellipsis
    with pytest.raises(ValueError):
        contract("i..", [0, 0])
    with pytest.raises(ValueError):
        contract(".i...", [0, 0])
    with pytest.raises(ValueError):
        contract("j->..j", [0, 0])
    with pytest.raises(ValueError):
        contract("j->.j...", [0, 0])

    # invalid subscript character
    with pytest.raises(ValueError):
        contract("i%...", [0, 0])
    with pytest.raises(ValueError):
        contract("...j$", [0, 0])
    with pytest.raises(ValueError):
        contract("i->&", [0, 0])

    # output subscripts must appear in input
    with pytest.raises(ValueError):
        contract("i->ij", [0, 0])

    # output subscripts may only be specified once
    with pytest.raises(ValueError):
        contract("ij->jij", [[0, 0], [0, 0]])

    # dimensions much match when being collapsed
    with pytest.raises(ValueError):
        contract("ii", np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError):
        contract("ii->i", np.arange(6).reshape(2, 3))

    # broadcasting to new dimensions must be enabled explicitly
    with pytest.raises(ValueError):
        contract("i", np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError):
        contract("i->i", [[0, 1], [0, 1]], out=np.arange(4).reshape(2, 2))


def test_contract_inputs():

    with pytest.raises(TypeError):
        contract_path("i->i", [[0, 1], [0, 1]], bad_kwarg=True)

    with pytest.raises(ValueError):
        contract_path("i->i", [[0, 1], [0, 1]], memory_limit=-1)


@pytest.mark.parametrize(
    "string",
    [
        # Ellipse
        '...a->...',
        'a...->...',
        'a...a->...a',
        '...,...',
        'a,b',
        '...a,...b',
    ])
def test_compare(string):
    views = build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(string, *views)
    assert np.allclose(ein, opt)

    opt = contract(string, *views, optimize='optimal')
    assert np.allclose(ein, opt)


def test_ellipse_input1():
    string = '...a->...'
    views = build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(views[0], [Ellipsis, 0], [Ellipsis])
    assert np.allclose(ein, opt)


def test_ellipse_input2():
    string = '...a'
    views = build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(views[0], [Ellipsis, 0])
    assert np.allclose(ein, opt)


def test_ellipse_input3():
    string = '...a->...a'
    views = build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(views[0], [Ellipsis, 0], [Ellipsis, 0])
    assert np.allclose(ein, opt)


def test_ellipse_input4():
    string = '...b,...a->...'
    views = build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(views[0], [Ellipsis, 1], views[1], [Ellipsis, 0], [Ellipsis])
    assert np.allclose(ein, opt)


def test_singleton_dimension_broadcast():
    # singleton dimensions broadcast (gh-10343)
    p = np.ones((10, 2))
    q = np.ones((1, 2))

    ein = contract('ij,ij->j', p, q, optimize=False)
    opt = contract('ij,ij->j', p, q, optimize=True)
    assert np.allclose(ein, opt)
    assert np.allclose(opt, [10., 10.])

    p = np.ones((1, 5))
    q = np.ones((5, 5))

    for optimize in (True, False):
        res1 = contract("...ij,...jk->...ik", p, p, optimize=optimize),
        res2 = contract("...ij,...jk->...ik", p, q, optimize=optimize)
        assert np.allclose(res1, res2)
        assert np.allclose(res2, np.full((1, 5), 5))


def test_large_int_input_format():
    string = 'ab,bc,cd'
    x, y, z = build_views(string)
    string_output = contract(string, x, y, z)
    int_output = contract(x, (1000, 1001), y, (1001, 1002), z, (1002, 1003))
    assert np.allclose(string_output, int_output)
    for i in range(10):
        transpose_output = contract(x, (i + 1, i))
        assert np.allclose(transpose_output, x.T)


def test_hashable_object_input_format():
    string = 'ab,bc,cd'
    x, y, z = build_views(string)
    string_output = contract(string, x, y, z)
    hash_output1 = contract(x, ('left', 'bond1'), y, ('bond1', 'bond2'), z, ('bond2', 'right'))
    hash_output2 = contract(x, ('left', 'bond1'), y, ('bond1', 'bond2'), z, ('bond2', 'right'), ('left', 'right'))
    assert np.allclose(string_output, hash_output1)
    assert np.allclose(hash_output1, hash_output2)
    for i in range(1, 10):
        transpose_output = contract(x, ('b' * i, 'a' * i))
        assert np.allclose(transpose_output, x.T)
