"""Regressions for the reported #220 workload and its discussion."""

import pytest

from opt_einsum import contract, contract_expression, contract_path


@pytest.mark.parametrize("optimize", ["optimal", "greedy", "dp"])
def test_broadcast_planning_issue220(optimize):
    # Original shape-only report: a singleton batch on the last operand should
    # not force construction of the 10000 x 32 x 32 outer product.
    shapes = [(32, 32, 32), (10000, 32), (1, 32)]
    path, info = contract_path("ijk,bj,bk->bi", *shapes, shapes=True, optimize=optimize)
    assert path == [(0, 2), (0, 1)]
    assert info.largest_intermediate == 320000
    assert info.opt_cost == 20545536


@pytest.mark.parametrize("constant", [False, True])
def test_broadcast_expression_changed_shape_issue220(constant):
    np = pytest.importorskip("numpy")
    # ContractExpression promises same-rank inputs may change dimension sizes.
    # Reuse the report's equation at a smaller batch size, then replace the
    # broadcast batch by a real batch after the first (constant-caching) call.
    rng = np.random.default_rng(42)
    a = rng.random((32, 32, 32))
    b = rng.random((8, 32))
    c = rng.random((1, 32))
    shapes = [a if constant else a.shape, b.shape, c.shape]
    expr = contract_expression("ijk,bj,bk->bi", *shapes, constants=[0] if constant else [])
    args = [b, c] if constant else [a, b, c]
    np.testing.assert_allclose(expr(*args), np.einsum("ijk,bj,bk->bi", a, b, c))
    c = rng.random((8, 32))
    args = [b, c] if constant else [a, b, c]
    out = np.empty((8, 32))
    assert expr(*args, out=out) is out
    np.testing.assert_allclose(out, np.einsum("ijk,bj,bk->bi", a, b, c))


def test_broadcast_planning_preserves_singleton_output_issue220():
    np = pytest.importorskip("numpy")
    # nedlrichards' reported output-shape concern: all occurrences may be 1.
    rng = np.random.default_rng(42)
    arrays = [rng.random(shape) for shape in [(32, 32, 32), (1, 32), (1, 32)]]
    actual = contract("ijk,bj,bk->bi", *arrays)
    assert actual.shape == (1, 32)
    np.testing.assert_allclose(actual, np.einsum("ijk,bj,bk->bi", *arrays))


def test_broadcast_planning_rejects_invalid_diagonal_issue220():
    np = pytest.importorskip("numpy")
    # The exact invalid repeated-label example supplied in the issue discussion
    # must not become valid when a singleton occurrence is omitted from planning.
    with pytest.raises(ValueError, match="Repeated label"):
        contract("ii,i->i", np.ones((1, 2)), np.ones(2))


@pytest.mark.parametrize(
    "equation, expected_cost",
    [
        ("ab,c,a->abc", 30),
        ("ab,c,a->bc", 22),
    ],
)
def test_broadcast_projected_cost_matches_optimizer(equation, expected_cost):
    np = pytest.importorskip("numpy")
    # An omitted singleton axis does not introduce reduction arithmetic when
    # it is absent from the projected contraction's index union.
    arrays = [
        np.arange(1.0, 4.0).reshape(1, 3),
        np.arange(1.0, 3.0),
        np.arange(1.0, 5.0),
    ]
    _, info = contract_path(equation, *arrays, optimize="optimal")
    assert info.opt_cost == expected_cost
    np.testing.assert_array_equal(
        contract(equation, *arrays, optimize="optimal"),
        np.einsum(equation, *arrays),
    )
