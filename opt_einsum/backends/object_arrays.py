"""Functions for performing contractions with array elements which are objects."""

import functools
import operator

from opt_einsum.typing import ArrayType


def object_einsum(eq: str, *arrays: ArrayType) -> ArrayType:
    """A ``einsum`` implementation for ``numpy`` arrays with object dtype.
    The loop is performed in python, meaning the objects themselves need
    only to implement ``__mul__`` and ``__add__`` for the contraction to be
    computed. This may be useful when, for example, computing expressions of
    tensors with symbolic elements, but note it will be very slow when compared
    to ``numpy.einsum`` and numeric data types!

    Parameters
    ----------
    eq : str
        The contraction string, should specify output.
    arrays : sequence of arrays
        These can be any indexable arrays as long as addition and
        multiplication is defined on the elements.

    Returns:
    -------
    out : numpy.ndarray
        The output tensor, with ``dtype=object``.
    """
    import numpy as np  # type: ignore

    # when called by ``opt_einsum`` we will always be given a full eq
    lhs, output = eq.split("->")
    inputs = lhs.split(",")

    sizes: dict[str, int] = {}
    for term, array in zip(inputs, arrays):
        for k, d in zip(term, array.shape):
            if k not in sizes or sizes[k] == 1:
                sizes[k] = d

    # Broadcast views up front so the inner loop can use ordinary coordinates.
    broadcasted_arrays = []
    for term, array in zip(inputs, arrays):
        target_shape = tuple(sizes[k] for k in term)
        broadcasted_arrays.append(np.broadcast_to(array, target_shape))
    arrays = tuple(broadcasted_arrays)

    out_size = tuple(sizes[k] for k in output)
    out = np.empty(out_size, dtype=object)

    inner = tuple(k for k in sizes if k not in output)
    inner_size = tuple(sizes[k] for k in inner)

    for coo_o in np.ndindex(*out_size):
        coord = dict(zip(output, coo_o))

        def gen_inner_sum():
            for coo_i in np.ndindex(*inner_size):
                coord.update(dict(zip(inner, coo_i)))
                locs = (tuple(coord[k] for k in term) for term in inputs)
                elements = (array[loc] for array, loc in zip(arrays, locs))
                yield functools.reduce(operator.mul, elements)

        terms = gen_inner_sum()
        # Empty reductions are zero; nonempty ones need only object addition.
        out[coo_o] = functools.reduce(operator.add, terms, next(terms, 0))

    return out
