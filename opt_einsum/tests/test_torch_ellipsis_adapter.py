"""Regression coverage for backend symbol conversion with ellipses."""

import pytest

from opt_einsum.parser import convert_to_valid_einsum_chars


@pytest.mark.parametrize(
    ("equation", "expected"),
    [
        ("...i,i", "...a,a"),
        ("...i,i->...", "...a,a->..."),
        ("...\u03bb,\u03bb->...", "...a,a->..."),
        ("ij,j->i", "ab,b->a"),
    ],
)
def test_convert_symbols_preserves_ellipsis(equation, expected):
    assert convert_to_valid_einsum_chars(equation) == expected


@pytest.mark.parametrize(
    ("equation", "reference", "shape"),
    [
        ("...i,i", "...i,i", (2, 2, 2, 3)),
        ("...i,i->...", "...i,i->...", (2, 2, 2, 3)),
        ("...i,i->...", "...i,i->...", (3,)),
        ("...\u03bb,\u03bb->...", "...i,i->...", (2, 2, 2, 3)),
        ("ij,j->i", "ij,j->i", (2, 3)),
    ],
)
def test_torch_adapter_matches_torch_with_ellipsis(equation, reference, shape):
    torch = pytest.importorskip("torch")
    from opt_einsum.backends import torch as adapter

    tensor = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape)
    weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    expected = torch.einsum(reference, tensor, weights)
    actual = adapter.einsum(equation, tensor, weights)
    torch.testing.assert_close(actual, expected)


def test_torch_adapter_batched_matmul_with_ellipsis():
    torch = pytest.importorskip("torch")
    from opt_einsum.backends import torch as adapter

    left = torch.arange(24.0).reshape(2, 3, 4)
    right = torch.arange(40.0).reshape(2, 4, 5)
    equation = "...ij,...jk->...ik"
    torch.testing.assert_close(adapter.einsum(equation, left, right), torch.einsum(equation, left, right))


@pytest.mark.parametrize("equation", [".i,i->i", "..i,i->i", "....i,i->i"])
def test_torch_adapter_does_not_turn_invalid_dots_into_indices(equation):
    torch = pytest.importorskip("torch")
    from opt_einsum.backends import torch as adapter

    matrix = torch.arange(6.0).reshape(2, 3)
    weights = torch.ones(3)
    with pytest.raises((RuntimeError, ValueError)):
        adapter.einsum(equation, matrix, weights)


@pytest.mark.parametrize("number_of_labels", [26, 27, 52])
@pytest.mark.parametrize("has_ellipsis", [False, True])
def test_torch_adapter_preserves_implicit_output_order(number_of_labels, has_ellipsis):
    torch = pytest.importorskip("torch")
    from opt_einsum.backends import torch as adapter

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[:number_of_labels]
    equation = ("..." if has_ellipsis else "") + labels
    shape = ((2,) if has_ellipsis else ()) + (1,) * (number_of_labels - 1) + (3,)
    values = torch.arange(2 * 3 if has_ellipsis else 3, dtype=torch.float64).reshape(shape)
    torch.testing.assert_close(adapter.einsum(equation, values), torch.einsum(equation, values))
