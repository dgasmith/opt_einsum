"""
Directly tests various parser utility functions.
"""

import pytest

from opt_einsum.parser import get_symbol, parse_einsum_input, possibly_convert_to_numpy
from opt_einsum.testing import build_arrays_from_tuples, using_numpy


def test_get_symbol() -> None:
    assert get_symbol(2) == "c"
    assert get_symbol(200000) == "\U00031540"
    # Ensure we skip surrogates '[\uD800-\uDFFF]'
    assert get_symbol(55295) == "\ud88b"
    assert get_symbol(55296) == "\ue000"
    assert get_symbol(57343) == "\ue7ff"


def test_parse_einsum_input() -> None:
    input_subscripts, output_subscript, operands = parse_einsum_input(["ab,bc,cd", (2, 3), (3, 4), (4, 5)], shapes=True)
    assert input_subscripts == eq
    assert output_subscript == "ad"
    assert operands == ops


def test_parse_einsum_input_shapes_error() -> None:
    eq = "ab,bc,cd"
    ops = build_arrays_from_tuples([(2, 3), (3, 4), (4, 5)])

    with pytest.raises(ValueError):
        _ = parse_einsum_input([eq, *ops], shapes=True)


@using_numpy
def test_parse_einsum_input_shapes() -> None:
    import numpy as np

    eq = "ab,bc,cd"
    shapes = [(2, 3), (3, 4), (4, 5)]
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *shapes], shapes=True)
    assert input_subscripts == eq
    assert output_subscript == "ad"
    assert np.allclose([possibly_convert_to_numpy(shp) for shp in shps], operands)
