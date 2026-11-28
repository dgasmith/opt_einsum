"""Tests for execution options on BLAS-eligible contractions."""

import importlib
from typing import Any

import pytest

from opt_einsum import contract, contract_expression, shared_intermediates


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("expression", [False, True])
def test_blas_dtype_option_preserves_widened_result(sharing: bool, expression: bool) -> None:
    np = pytest.importorskip("numpy")
    a = np.array([100, 100], dtype=np.int8)
    b = np.array([1, 1], dtype=np.int8)
    expected = np.einsum("i,i->", a, b, dtype=np.int64)

    def run() -> Any:
        if expression:
            expr = contract_expression("i,i->", a.shape, b.shape, dtype=np.int64)
            return expr(a, b)
        return contract("i,i->", a, b, dtype=np.int64)

    if sharing:
        with shared_intermediates():
            result = run()
    else:
        result = run()
    assert result == expected == 200
    assert result.dtype == expected.dtype


def test_blas_options_do_not_reuse_different_dtypes_in_shared_cache() -> None:
    np = pytest.importorskip("numpy")
    a = np.array([100, 100], dtype=np.int8)
    b = np.array([1, 1], dtype=np.int8)
    with shared_intermediates():
        int_result = contract("i,i->", a, b, dtype=np.int64)
        float_result = contract("i,i->", a, b, dtype=np.float64)
    assert int_result == float_result == 200
    assert int_result.dtype == np.dtype("int64")
    assert float_result.dtype == np.dtype("float64")


def test_blas_casting_option_and_out() -> None:
    np = pytest.importorskip("numpy")
    a = np.array([[1.5, 2.0], [3.0, 4.0]])
    out = np.empty((2, 2), dtype=np.int64)
    expected = np.empty_like(out)
    np.einsum("ij,jk->ik", a, a, out=expected, casting="unsafe")
    result = contract("ij,jk->ik", a, a, out=out, casting="unsafe")
    assert result is out
    np.testing.assert_array_equal(out, expected)


def test_blas_order_option() -> None:
    np = pytest.importorskip("numpy")
    a = np.arange(9.0).reshape(3, 3)
    result = contract("ij,jk->ik", a, a, order="F")
    np.testing.assert_allclose(result, np.einsum("ij,jk->ik", a, a, order="F"))
    assert result.flags.f_contiguous


def test_blas_path_is_preserved_without_execution_options(monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip("numpy")
    module = importlib.import_module("opt_einsum.contract")
    calls = []
    original = module._tensordot

    def observed(*args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_tensordot", observed)
    a = np.eye(2)
    np.testing.assert_allclose(contract("ij,jk->ik", a, a), a)
    assert calls


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize("expression", [False, True])
def test_mixed_einsum_and_blas_steps_honor_dtype(sharing: bool, expression: bool) -> None:
    np = pytest.importorskip("numpy")
    a = np.array([[100, 0], [0, 100]], dtype=np.int8)
    b = np.ones((2, 2), dtype=np.int8)
    c = np.ones((2, 2), dtype=np.int8)
    equation = "ii,ij,jk->k"
    path = [(0, 1), (0, 1)]
    expected = np.einsum(equation, a, b, c, dtype=np.int64)

    def run() -> Any:
        if expression:
            expr = contract_expression(equation, a.shape, b.shape, c.shape, optimize=path, dtype=np.int64)
            return expr(a, b, c)
        return contract(equation, a, b, c, optimize=path, dtype=np.int64)

    if sharing:
        with shared_intermediates():
            result = run()
    else:
        result = run()
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype


def test_expression_with_constants_preserves_dtype() -> None:
    np = pytest.importorskip("numpy")
    a = np.full((2, 2), 100, dtype=np.int8)
    b = np.ones((2, 2), dtype=np.int8)
    c = np.ones((2, 2), dtype=np.int8)
    equation = "ij,jk,kl->il"
    expr = contract_expression(equation, a, b, c.shape, constants=[0, 1], optimize=[(0, 1), (0, 1)], dtype=np.int64)
    result = expr(c)
    expected = np.einsum(equation, a, b, c, dtype=np.int64)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype


def test_backend_without_einsum_keeps_tensordot_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType

    np = pytest.importorskip("numpy")
    from opt_einsum.backends import dispatch

    backend: Any = ModuleType("test_no_einsum_backend")
    backend.tensordot = np.tensordot
    backend.transpose = np.transpose
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setattr(dispatch, "_cached_funcs", dispatch._cached_funcs.copy())
    monkeypatch.setattr(dispatch, "_has_tensordot", dispatch._has_tensordot.copy())
    monkeypatch.setattr(dispatch, "_has_einsum", dispatch._has_einsum.copy())
    a = np.array([1.0, 2.0])
    assert contract("i,i->", a, a, backend=backend.__name__) == 5.0
    with pytest.raises(TypeError):
        contract("i,i->", a, a, dtype=np.float64, backend=backend.__name__)


def test_shared_expression_with_evaluated_blas_constants_writes_out() -> None:
    np = pytest.importorskip("numpy")
    a = np.array([[1, 2], [3, 4]], dtype=np.int8)
    b = np.eye(2, dtype=np.int8)
    c = np.eye(2, dtype=np.int8)
    equation = "ij,jk,kk->i"
    expr = contract_expression(equation, a, b, c.shape, constants=[0, 1], optimize=[(0, 1), (0, 1)], dtype=np.int64)
    out = np.full(2, -7, dtype=np.int64)
    expected = np.einsum(equation, a, b, c, dtype=np.int64)
    with shared_intermediates():
        result = expr(c, out=out)
    assert result is out
    np.testing.assert_array_equal(out, expected)


def test_blas_safe_casting_rejects_before_writing_out() -> None:
    np = pytest.importorskip("numpy")
    a = np.array([[1.5, 2.0], [3.0, 4.0]])
    out = np.full((2, 2), -7, dtype=np.int64)
    with pytest.raises(TypeError):
        np.einsum("ij,jk->ik", a, a, out=np.empty_like(out), casting="safe")
    with pytest.raises(TypeError):
        contract("ij,jk->ik", a, a, out=out, casting="safe")
    np.testing.assert_array_equal(out, -7)


def test_blas_dtype_with_interleaved_subscripts() -> None:
    np = pytest.importorskip("numpy")
    a = np.array([100, 100], dtype=np.int8)
    b = np.array([1, 1], dtype=np.int8)
    result = contract(a, [0], b, [0], [], dtype=np.int64)
    assert result == 200
    assert result.dtype == np.dtype("int64")


def test_non_numpy_einsum_backend_does_not_ignore_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType

    np = pytest.importorskip("numpy")
    from opt_einsum.backends import dispatch

    backend: Any = ModuleType("test_ignoring_options_backend")
    backend.tensordot = np.tensordot
    backend.einsum = lambda eq, *arrays, **kwargs: np.einsum(eq, *arrays)
    backend.transpose = np.transpose
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setattr(dispatch, "_cached_funcs", dispatch._cached_funcs.copy())
    monkeypatch.setattr(dispatch, "_has_tensordot", dispatch._has_tensordot.copy())
    monkeypatch.setattr(dispatch, "_has_einsum", dispatch._has_einsum.copy())
    a = np.array([100, 100], dtype=np.int8)
    b = np.ones(2, dtype=np.int8)
    out = np.full((), -7, dtype=np.int64)
    with pytest.raises(TypeError):
        contract("i,i->", a, b, out=out, dtype=np.int64, backend=backend.__name__)
    assert out == -7


@pytest.mark.parametrize("with_out", [False, True])
def test_torch_blas_options_keep_existing_error(with_out: bool) -> None:
    torch = pytest.importorskip("torch")
    a = torch.tensor([100, 100], dtype=torch.int8)
    b = torch.ones(2, dtype=torch.int8)
    out = torch.full((), -7, dtype=torch.int64) if with_out else None
    with pytest.raises(TypeError):
        contract("i,i->", a, b, out=out, dtype=torch.int64, backend="torch")
    if out is not None:
        assert out.item() == -7


@pytest.mark.parametrize("sharing", [False, True])
def test_unrecognized_options_do_not_collide_with_internal_path(sharing: bool) -> None:
    from contextlib import nullcontext

    np = pytest.importorskip("numpy")
    a = np.array([100, 100], dtype=np.int8)
    b = np.ones(2, dtype=np.int8)
    context = shared_intermediates() if sharing else nullcontext()
    with context, pytest.raises(TypeError):
        contract("i,i->", a, b, dtype=np.int64, _full_contraction_list=[])


@pytest.mark.parametrize("option", ["evaluate_constants", "operands_"])
def test_expression_reserved_options_keep_original_error_on_torch(option: str) -> None:
    torch = pytest.importorskip("torch")
    options: dict[str, Any] = {option: True}
    expr = contract_expression("i,i->i", (2,), (2,), **options)
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    with pytest.raises(TypeError, match=f"multiple values for (keyword )?argument '{option}'"):
        expr(a, b, backend="torch")


@pytest.mark.parametrize("option", ["operands_", "contraction_list", "backend", "evaluate_constants", "out"])
def test_expression_mutated_reserved_options_keep_original_error(option: str) -> None:
    np = pytest.importorskip("numpy")
    expr = contract_expression("i,i->i", (2,), (2,))
    expr.kwargs[option] = True
    a = np.array([1.0, 2.0])
    with pytest.raises(TypeError, match=f"multiple values for (keyword )?argument '{option}'"):
        expr(a, a)
