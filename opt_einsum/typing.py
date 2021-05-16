"""
Types used in the opt_einsum package
"""

from typing import List, Tuple, Set, Collection

PathType = Collection[Tuple[int, ...]]
TensorIndexType = Set[str]
TensorShapeType = Tuple[int, ...]
