from __future__ import division, absolute_import, print_function

import numpy as np
from opt_einsum import contract, contract_path
import pytest

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
        contract(*(None,)*63)

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

    with pytest.raises(TypeError):
        contract(views[0], [Ellipsis, 'a'], [Ellipsis, 0])

    with pytest.raises(TypeError):
        contract(views[0], [Ellipsis, 0], [Ellipsis, 'a'])


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


@pytest.mark.parametrize("string", [
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

