"""
Directly tests various parser utility functions.
"""

from multiprocessing.sharedctypes import Value
import numpy as np
import pytest
from opt_einsum.parser import get_symbol, parse_einsum_input, possibly_convert_to_numpy


def test_get_symbol():
    assert get_symbol(2) == "c"
    assert get_symbol(200000) == "\U00031540"
    # Ensure we skip surrogates '[\uD800-\uDFFF]'
    assert get_symbol(55295) == "\ud88b"
    assert get_symbol(55296) == "\ue000"
    assert get_symbol(57343) == "\ue7ff"


def test_parse_einsum_input():
    eq = "ab,bc,cd"
    ops = [np.random.rand(2, 3), np.random.rand(3, 4), np.random.rand(4, 5)]
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *ops])
    assert input_subscripts == eq
    assert output_subscript == "ad"
    assert operands == ops


def test_parse_einsum_input_shapes_error():
    eq = "ab,bc,cd"
    ops = [np.random.rand(2, 3), np.random.rand(3, 4), np.random.rand(4, 5)]

    with pytest.raises(ValueError):
        _ = parse_einsum_input([eq, *ops], shapes=True)


def test_parse_einsum_input_shapes():
    eq = "ab,bc,cd"
    shps = [(2, 3), (3, 4), (4, 5)]
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *shps], shapes=True)
    assert input_subscripts == eq
    assert output_subscript == "ad"
    assert np.allclose([possibly_convert_to_numpy(shp) for shp in shps], operands)
