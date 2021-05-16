"""
Types used in the opt_einsum package
"""

from typing import List, Tuple, Set

PathType = List[Tuple[int, ...]]
TensorIndexType = Set[str]
TensorShapeType = Tuple[int, ...]
