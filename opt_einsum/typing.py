"""
Types used in the opt_einsum package
"""

from typing import Any, Collection, FrozenSet, List, Optional, Tuple, Union

PathType = Collection[Tuple[int, ...]]
ArrayType = Any  # TODO
ArrayIndexType = FrozenSet[str]
TensorShapeType = Tuple[int, ...]
ContractionListType = List[Tuple[Any, ArrayIndexType, str, Optional[Tuple[str, ...]], Union[str, bool]]]
