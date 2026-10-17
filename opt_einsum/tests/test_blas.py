"""
Tests the BLAS capability for the opt_einsum module.
"""

from typing import Any

import pytest

from opt_einsum import blas, contract, contract_expression

blas_tests = [
    # DOT
    ((["k", "k"], "", set("k")), "DOT"),  # DDOT
    ((["ijk", "ijk"], "", set("ijk")), "DOT"),  # DDOT
    # GEMV?
    # GEMM
    ((["ij", "jk"], "ik", set("j")), "GEMM"),  # GEMM N N
    ((["ijl", "jlk"], "ik", set("jl")), "GEMM"),  # GEMM N N Tensor
    ((["ij", "kj"], "ik", set("j")), "GEMM"),  # GEMM N T
    ((["ijl", "kjl"], "ik", set("jl")), "GEMM"),  # GEMM N T Tensor
    ((["ji", "jk"], "ik", set("j")), "GEMM"),  # GEMM T N
    ((["jli", "jlk"], "ik", set("jl")), "GEMM"),  # GEMM T N Tensor
    ((["ji", "kj"], "ik", set("j")), "GEMM"),  # GEMM T T
    ((["jli", "kjl"], "ik", set("jl")), "GEMM"),  # GEMM T T Tensor
    # GEMM with final transpose
    ((["ij", "jk"], "ki", set("j")), "GEMM"),  # GEMM N N
    ((["ijl", "jlk"], "ki", set("jl")), "GEMM"),  # GEMM N N Tensor
    ((["ij", "kj"], "ki", set("j")), "GEMM"),  # GEMM N T
    ((["ijl", "kjl"], "ki", set("jl")), "GEMM"),  # GEMM N T Tensor
    ((["ji", "jk"], "ki", set("j")), "GEMM"),  # GEMM T N
    ((["jli", "jlk"], "ki", set("jl")), "GEMM"),  # GEMM T N Tensor
    ((["ji", "kj"], "ki", set("j")), "GEMM"),  # GEMM T T
    ((["jli", "kjl"], "ki", set("jl")), "GEMM"),  # GEMM T T Tensor
    # Tensor Dot (requires copy), lets not deal with this for now
    ((["ilj", "jlk"], "ik", set("jl")), "TDOT"),  # FT GEMM N N Tensor
    ((["ijl", "ljk"], "ik", set("jl")), "TDOT"),  # ST GEMM N N Tensor
    ((["ilj", "kjl"], "ik", set("jl")), "TDOT"),  # FT GEMM N T Tensor
    ((["ijl", "klj"], "ik", set("jl")), "TDOT"),  # ST GEMM N T Tensor
    ((["lji", "jlk"], "ik", set("jl")), "TDOT"),  # FT GEMM T N Tensor
    ((["jli", "ljk"], "ik", set("jl")), "TDOT"),  # ST GEMM T N Tensor
    ((["lji", "jlk"], "ik", set("jl")), "TDOT"),  # FT GEMM T N Tensor
    ((["jli", "ljk"], "ik", set("jl")), "TDOT"),  # ST GEMM T N Tensor
    # Tensor Dot (requires copy), lets not deal with this for now with transpose
    ((["ilj", "jlk"], "ik", set("lj")), "TDOT"),  # FT GEMM N N Tensor
    ((["ijl", "ljk"], "ik", set("lj")), "TDOT"),  # ST GEMM N N Tensor
    ((["ilj", "kjl"], "ik", set("lj")), "TDOT"),  # FT GEMM N T Tensor
    ((["ijl", "klj"], "ik", set("lj")), "TDOT"),  # ST GEMM N T Tensor
    ((["lji", "jlk"], "ik", set("lj")), "TDOT"),  # FT GEMM T N Tensor
    ((["jli", "ljk"], "ik", set("lj")), "TDOT"),  # ST GEMM T N Tensor
    ((["lji", "jlk"], "ik", set("lj")), "TDOT"),  # FT GEMM T N Tensor
    ((["jli", "ljk"], "ik", set("lj")), "TDOT"),  # ST GEMM T N Tensor
    # Other
    ((["ijk", "ikj"], "", set("ijk")), "DOT/EINSUM"),  # Transpose DOT
    ((["i", "j"], "ij", set()), "OUTER/EINSUM"),  # Outer
    ((["ijk", "ik"], "j", set("ik")), "GEMV/EINSUM"),  # Matrix-vector
    ((["ijj", "jk"], "ik", set("j")), False),  # Double index
    ((["ijk", "j"], "ij", set()), False),  # Index sum 1
    ((["ij", "ij"], "ij", set()), False),  # Index sum 2
]


@pytest.mark.parametrize("inp,benchmark", blas_tests)
def test_can_blas(inp: Any, benchmark: bool) -> None:
    result = blas.can_blas(*inp)
    assert result == benchmark


def test_blas_out() -> None:
    np = pytest.importorskip("numpy")

    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    c = np.random.rand(4, 4)
    d = np.empty((4, 4))

    contract("ij,jk->ik", a, b, out=d)
    np.testing.assert_allclose(d, np.dot(a, b))
    assert np.allclose(d, np.dot(a, b))

    contract("ij,jk,kl->il", a, b, c, out=d)
    np.testing.assert_allclose(d, np.dot(a, b).dot(c))


@pytest.mark.parametrize(
    "string,shapes",
    [
        ("i,i->", [(4,), (4,)]),
        ("ij,ij->", [(3, 4), (3, 4)]),
        ("ij,jk,ik->", [(2, 3), (3, 4), (2, 4)]),
    ],
)
@pytest.mark.parametrize("use_blas", [False, True])
@pytest.mark.parametrize("expression", [False, True])
def test_scalar_out(string: str, shapes: list[tuple[int, ...]], use_blas: bool, expression: bool) -> None:
    np = pytest.importorskip("numpy")
    views = [np.arange(np.prod(shape), dtype=np.float64).reshape(shape) for shape in shapes]
    expected = np.einsum(string, *views)
    out = np.full((), np.nan)

    if expression:
        expr = contract_expression(string, *shapes, use_blas=use_blas)
        result = expr(*views, out=out)
    else:
        result = contract(string, *views, use_blas=use_blas, out=out)

    assert result is out
    assert out.shape == ()
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("out", [0, 1])
@pytest.mark.parametrize("use_blas", [False, True])
@pytest.mark.parametrize("expression", [False, True])
def test_scalar_out_python_int(out: int, use_blas: bool, expression: bool) -> None:
    np = pytest.importorskip("numpy")
    views = [np.arange(4, dtype=np.float64), np.arange(4, dtype=np.float64)]

    with pytest.raises(TypeError):
        if expression:
            expr = contract_expression("i,i->", (4,), (4,), use_blas=use_blas)
            expr(*views, out=out)
        else:
            contract("i,i->", *views, use_blas=use_blas, out=out)
