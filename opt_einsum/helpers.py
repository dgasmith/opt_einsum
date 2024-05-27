"""
Contains helper functions for opt_einsum testing scripts
"""

from typing import Any, Collection, Dict, FrozenSet, Iterable, List, Tuple, overload
from typing import Any, Collection, Dict, FrozenSet, Iterable, List, Literal, Optional, Tuple, Union, overload

from opt_einsum.typing import ArrayIndexType, ArrayType

from opt_einsum.parser import get_symbol
from opt_einsum.typing import ArrayIndexType, PathType

__all__ = ["build_views", "compute_size_by_dict", "find_contraction", "flop_count"]

_valid_chars = "abcdefghijklmopqABC"
_sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4]
_default_dim_dict = {c: s for c, s in zip(_valid_chars, _sizes)}


def build_views(string: str, dimension_dict: Optional[Dict[str, int]] = None) -> List[np.ndarray]:
    """
    Builds random numpy arrays for testing.

    Parameters
    ----------
    string : str
        List of tensor strings to build
    dimension_dict : dictionary
        Dictionary of index _sizes

    Returns
    -------
    ret : list of np.ndarry's
        The resulting views.

    Examples
    --------
    >>> view = build_views('abbc', {'a': 2, 'b':3, 'c':5})
    >>> view[0].shape
    (2, 3, 3, 5)

    """

    if dimension_dict is None:
        dimension_dict = _default_dim_dict

    views = []
    terms = string.split("->")[0].split(",")
    for term in terms:
        dims = [dimension_dict[x] for x in term]
        views.append(np.random.rand(*dims))
    return views


@overload
def compute_size_by_dict(indices: Iterable[int], idx_dict: List[int]) -> int: ...


@overload
def compute_size_by_dict(indices: Collection[str], idx_dict: Dict[str, int]) -> int: ...


def compute_size_by_dict(indices: Any, idx_dict: Any) -> int:
    """
    Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index _sizes

    Returns
    -------
    ret : int
        The resulting product.

    Examples
    --------
    >>> compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
    90

    """
    ret = 1
    for i in indices:  # lgtm [py/iteration-string-and-sequence]
        ret *= idx_dict[i]
    return ret


def find_contraction(
    positions: Collection[int],
    input_sets: List[ArrayIndexType],
    output_set: ArrayIndexType,
) -> Tuple[FrozenSet[str], List[ArrayIndexType], ArrayIndexType, ArrayIndexType]:
    """
    Finds the contraction for a given set of input and output sets.

    Parameters
    ----------
    positions : iterable
        Integer positions of terms used in the contraction.
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript

    Returns
    -------
    new_result : set
        The indices of the resulting contraction
    remaining : list
        List of sets that have not been contracted, the new set is appended to
        the end of this list
    idx_removed : set
        Indices removed from the entire contraction
    idx_contraction : set
        The indices used in the current contraction

    Examples
    --------

    # A simple dot product test case
    >>> pos = (0, 1)
    >>> isets = [set('ab'), set('bc')]
    >>> oset = set('ac')
    >>> find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})

    # A more complex case with additional terms in the contraction
    >>> pos = (0, 2)
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('ac')
    >>> find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})
    """

    remaining = list(input_sets)
    inputs = (remaining.pop(i) for i in sorted(positions, reverse=True))
    idx_contract = frozenset.union(*inputs)
    idx_remain = output_set.union(*remaining)

    new_result = idx_remain & idx_contract
    idx_removed = idx_contract - new_result
    remaining.append(new_result)

    return new_result, remaining, idx_removed, idx_contract


def flop_count(
    idx_contraction: Collection[str],
    inner: bool,
    num_terms: int,
    size_dictionary: Dict[str, int],
) -> int:
    """
    Computes the number of FLOPS in the contraction.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction

    Returns
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples
    --------

    >>> flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
    30

    >>> flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
    60

    """

    overall_size = compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor


def has_array_interface(array: ArrayType) -> ArrayType:
    if hasattr(array, "__array_interface__"):
        return True
    else:
        return False

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

    if seed is not None:
        np.random.seed(seed)

    # total number of indices
    num_inds = n * regularity // 2 + n_out
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

    if return_size_dict:
        return (
            eq,
            shapes,
            size_dict,
        )
    else:
        return (eq, shapes)
