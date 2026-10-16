"""
Directly tests various parser utility functions.
"""

from typing import Any

import pytest

from opt_einsum.parser import find_output_shape, gen_unused_symbols, get_shape, get_symbol, parse_einsum_input
from opt_einsum.testing import build_arrays_from_tuples


def test_get_symbol() -> None:
    assert get_symbol(2) == "c"
    assert get_symbol(200000) == "\U000315cc"
    # Ensure we skip surrogates '[\uD800-\uDFFF]'
    assert get_symbol(55155) == "\ud7ff"
    assert get_symbol(55156) == "\ue000"
    assert get_symbol(55295) == "\ue08b"
    assert get_symbol(55296) == "\ue08c"
    assert get_symbol(57343) == "\ue88b"


@pytest.mark.parametrize("n", [52, 55156, 55157, 60000])
def test_symbols_utf8(n: int) -> None:
    symbols = "".join(get_symbol(i) for i in range(n))
    assert len(set(symbols)) == n
    assert symbols.encode("utf-8").decode("utf-8") == symbols


def test_unused_symbols_utf8() -> None:
    used = "ab\ud7ff\ue000"
    symbols = "".join(gen_unused_symbols(used, 60000))
    assert not set(symbols).intersection(used)
    assert len(set(symbols)) == 60000
    assert symbols.encode("utf-8").decode("utf-8") == symbols


def test_parse_einsum_input() -> None:
    eq = "ab,bc,cd"
    ops = build_arrays_from_tuples([(2, 3), (3, 4), (4, 5)])
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *ops])
    assert input_subscripts == eq
    assert output_subscript == "ad"
    assert operands == ops


def test_parse_einsum_input_shapes_error() -> None:
    eq = "ab,bc,cd"
    ops = build_arrays_from_tuples([(2, 3), (3, 4), (4, 5)])

    with pytest.raises(ValueError):
        _ = parse_einsum_input([eq, *ops], shapes=True)


def test_parse_einsum_input_shapes() -> None:
    eq = "ab,bc,cd"
    shapes = [(2, 3), (3, 4), (4, 5)]
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *shapes], shapes=True)
    assert input_subscripts == eq
    assert output_subscript == "ad"
    assert shapes == operands


def test_parse_with_ellisis() -> None:
    eq = "...a,ab"
    shapes = [(2, 3), (3, 4)]
    input_subscripts, output_subscript, operands = parse_einsum_input([eq, *shapes], shapes=True)
    assert input_subscripts == "da,ab"
    assert output_subscript == "db"
    assert shapes == operands


@pytest.mark.parametrize(
    "array, shape",
    [
        [[5], (1,)],
        [[5, 5], (2,)],
        [(5, 5), (2,)],
        [[[[[[5, 2]]]]], (1, 1, 1, 1, 2)],
        [[[[[["abcdef", "b"]]]]], (1, 1, 1, 1, 2)],
        ["A", ()],
        [b"A", ()],
        [True, ()],
        [5, ()],
        [5.0, ()],
        [5.0 + 0j, ()],
    ],
)
def test_get_shapes(array: Any, shape: tuple[int]) -> None:
    assert get_shape(array) == shape


def test_find_output_shape_with_zero_length_broadcast() -> None:
    assert find_output_shape(["i", "i"], [(0,), (1,)], "i") == (0,)
