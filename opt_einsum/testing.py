"""
Testing routines for opt_einsum.
"""

from importlib import import_module
from importlib.util import find_spec
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import pytest

from opt_einsum.parser import get_symbol
from opt_einsum.typing import GenericArrayType, PathType

_valid_chars = "abcdefghijklmopqABC"
_sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4]
_default_dim_dict = {c: s for c, s in zip(_valid_chars, _sizes)}

HAS_NUMPY = find_spec("numpy") is not None

using_numpy = pytest.mark.skipif(
    not HAS_NUMPY,
    reason="Numpy not detected.",
)


def import_numpy_or_skip() -> Any:
    if not HAS_NUMPY:
        pytest.skip("Numpy not detected.")
    else:
        return import_module("numpy")


def build_views(string: str, dimension_dict: Optional[Dict[str, int]] = None) -> List[GenericArrayType]:
    """
    Builds random numpy arrays for testing.

    Parameters:
        string: List of tensor strings to build
        dimension_dict: Dictionary of index _sizes

    Returns
        The resulting views.

    Examples:
        ```python
        >>> view = build_views('abbc', {'a': 2, 'b':3, 'c':5})
        >>> view[0].shape
        (2, 3, 3, 5)
        ```

    """
    np = import_numpy_or_skip()

    if dimension_dict is None:
        dimension_dict = _default_dim_dict

    views = []
    terms = string.split("->")[0].split(",")
    for term in terms:
        dims = [dimension_dict[x] for x in term]
        views.append(np.random.rand(*dims))
    return views


@overload
def rand_equation(
    n: int,
    regularity: int,
    n_out: int = ...,
    d_min: int = ...,
    d_max: int = ...,
    seed: Optional[int] = ...,
    global_dim: bool = ...,
    *,
    return_size_dict: Literal[True],
) -> Tuple[str, PathType, Dict[str, int]]: ...


@overload
def rand_equation(
    n: int,
    regularity: int,
    n_out: int = ...,
    d_min: int = ...,
    d_max: int = ...,
    seed: Optional[int] = ...,
    global_dim: bool = ...,
    return_size_dict: Literal[False] = ...,
) -> Tuple[str, PathType]: ...


def rand_equation(
    n: int,
    regularity: int,
    n_out: int = 0,
    d_min: int = 2,
    d_max: int = 9,
    seed: Optional[int] = None,
    global_dim: bool = False,
    return_size_dict: bool = False,
) -> Union[Tuple[str, PathType, Dict[str, int]], Tuple[str, PathType]]:
    """Generate a random contraction and shapes.

    Parameters:
        n: Number of array arguments.
        regularity: 'Regularity' of the contraction graph. This essentially determines how
            many indices each tensor shares with others on average.
        n_out: Number of output indices (i.e. the number of non-contracted indices).
            Defaults to 0, i.e., a contraction resulting in a scalar.
        d_min: Minimum dimension size.
        d_max: Maximum dimension size.
        seed: If not None, seed numpy's random generator with this.
        global_dim: Add a global, 'broadcast', dimension to every operand.
        return_size_dict: Return the mapping of indices to sizes.

    Returns:
        eq: The equation string.
        shapes: The array shapes.
        size_dict: The dict of index sizes, only returned if ``return_size_dict=True``.

    Examples:
        ```python
        >>> eq, shapes = rand_equation(n=10, regularity=4, n_out=5, seed=42)
        >>> eq
        'oyeqn,tmaq,skpo,vg,hxui,n,fwxmr,hitplcj,kudlgfv,rywjsb->cebda'

        >>> shapes
        [(9, 5, 4, 5, 4),
        (4, 4, 8, 5),
        (9, 4, 6, 9),
        (6, 6),
        (6, 9, 7, 8),
        (4,),
        (9, 3, 9, 4, 9),
        (6, 8, 4, 6, 8, 6, 3),
        (4, 7, 8, 8, 6, 9, 6),
        (9, 5, 3, 3, 9, 5)]
        ```
    """

    np = import_numpy_or_skip()
    if seed is not None:
        np.random.seed(seed)

    # total number of indices
    num_inds = n * reg // 2 + n_out
    inputs = ["" for _ in range(n)]
    output = []

    size_dict = {get_symbol(i): np.random.randint(d_min, d_max + 1) for i in range(num_inds)}

    # generate a list of indices to place either once or twice
    def gen():
        for i, ix in enumerate(size_dict):
            # generate an outer index
            if i < n_out:
                output.append(ix)
                yield ix
            # generate a bond
            else:
                yield ix
                yield ix

    # add the indices randomly to the inputs
    for i, ix in enumerate(np.random.permutation(list(gen()))):
        # make sure all inputs have at least one index
        if i < n:
            inputs[i] += ix
        else:
            # don't add any traces on same op
            where = np.random.randint(0, n)
            while ix in inputs[where]:
                where = np.random.randint(0, n)

            inputs[where] += ix

    # possibly add the same global dim to every arg
    if global_dim:
        gdim = get_symbol(num_inds)
        size_dict[gdim] = np.random.randint(d_min, d_max + 1)
        for i in range(n):
            inputs[i] += gdim
        output += gdim

    # randomly transpose the output indices and form equation
    output = "".join(np.random.permutation(output))  # type: ignore
    eq = "{}->{}".format(",".join(inputs), output)

    # make the shapes
    shapes = [tuple(size_dict[ix] for ix in op) for op in inputs]

    ret = (eq, shapes)

    if return_size_dict:
        return ret + (size_dict,)
    else:
        return ret


def build_arrays_from_tuples(path: PathType) -> List[Any]:
    """Build random numpy arrays from a path.
    
    Parameters:
        path: The path to build arrays from.
    
    Returns:
        The resulting arrays."""
    np = import_numpy_or_skip()

    return [np.random.rand(*x) for x in path]
