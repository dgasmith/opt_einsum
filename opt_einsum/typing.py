"""
Types used in the opt_einsum package
"""

from typing import Collection, Set, Tuple

PathType = Collection[Tuple[int, ...]]
TensorIndexType = Set[str]
TensorShapeType = Tuple[int, ...]
