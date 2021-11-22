"""
Types used in the opt_einsum package
"""

from typing import Collection, FrozenSet, Tuple

PathType = Collection[Tuple[int, ...]]
TensorIndexType = FrozenSet[str]
TensorShapeType = Tuple[int, ...]
