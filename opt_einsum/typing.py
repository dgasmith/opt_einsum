"""Types used in the opt_einsum package."""

from collections import namedtuple
from collections.abc import Callable, Collection
from typing import Any, Literal

TensorShapeType = tuple[int, ...]
PathType = Collection[TensorShapeType]

ArrayType = Any

ArrayIndexType = frozenset[str]
ArrayShaped = namedtuple("ArrayShaped", ["shape"])

ContractionListType = list[tuple[Any, ArrayIndexType, str, tuple[str, ...] | None, str | bool]]
PathSearchFunctionType = Callable[[list[ArrayIndexType], ArrayIndexType, dict[str, int], int | None], PathType]

# Contract kwargs
OptimizeKind = (
    None
    | int
    | Literal["optimal"]
    | Literal["dp"]
    | Literal["greedy"]
    | Literal["random-greedy"]
    | Literal["random-greedy-128"]
    | Literal["branch-all"]
    | Literal["branch-2"]
    | Literal["auto"]
    | Literal["auto-hq"]
    | PathSearchFunctionType
    | PathType
)
BackendType = Literal["auto", "object", "autograd", "cupy", "dask", "jax", "theano", "tensorflow", "torch", "libjax"]
