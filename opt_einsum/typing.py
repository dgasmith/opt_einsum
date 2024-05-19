"""
Types used in the opt_einsum package
"""

from typing import Any, Callable, Collection, Dict, FrozenSet, List, Literal, Optional, Tuple, Union

PathType = Collection[Tuple[int, ...]]
TensorShapeType = Tuple[int, ...]

ArrayType = Any  # TODO
ArrayIndexType = FrozenSet[str]

ContractionListType = List[Tuple[Any, ArrayIndexType, str, Optional[Tuple[str, ...]], Union[str, bool]]]
PathSearchFunctionType = Callable[[List[ArrayIndexType], ArrayIndexType, Dict[str, int], Optional[int]], PathType]

# Contract kwargs
OptimizeKind = Union[
    None,
    bool,
    Literal[
        "optimal", "dp", "greedy", "random-greedy", "random-greedy-128", "branch-all", "branch-2", "auto", "auto-hq"
    ],
    PathType,
    PathSearchFunctionType,
]
