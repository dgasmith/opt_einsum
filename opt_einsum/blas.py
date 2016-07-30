import numpy as np
from . import paths


def can_blas(inputs, result, idx_removed):
    """
    Checks if we can use a BLAS call.

    Parameters
    ----------
    inputs : list of str
        Specifies the subscripts for summation.
    result : str
        Resulting summation.
    idx_removed : set
        Indices that are removed in the summation


    Returns
    -------
    type : str or bool
        The type of BLAS call to be used or False if none.

    Notes
    -----
    We assume several operations are not efficient such as a transposed
    DDOT, therefore 'ijk,jki->' would return False.

    Examples
    --------
    >>> _can_blas(['ij', 'jk'], 'ik', set('j'))
    'GEMM'

    >>> _can_blas(['ijj', 'jk'], 'ik', set('j'))
    False

    """

    # Gotta remove indices
    if len(idx_removed) == 0:
        return False

    # Can only do two
    if len(inputs) != 2:
        return False

    # Build a few temporaries
    sets = [set(x) for x in inputs]
    keep_left = sets[0] - idx_removed
    keep_right = sets[1] - idx_removed
    input_left = inputs[0]
    input_right = inputs[1]
    rs = len(idx_removed)

    if any(len(l) != len(s) for l, s in zip(inputs, sets)):
        return False

    # Cannot handle partial inner
    if len(keep_left & keep_right):
        return False

    # DDOT
    elif inputs[0] == inputs[1]:
        return 'DOT'

    # DDOT doesnt make sense if you have to tranpose
    elif sets[0] == sets[1]:
        return False

    # GEMM no transpose
    elif input_left[-rs:] == input_right[:rs]:
        return 'GEMM'

    # GEMM transpose both
    elif input_left[:rs] == input_right[-rs:]:
        return 'GEMM'

    # GEMM transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        return 'GEMM'

    # GEMM tranpose left
    elif input_left[:rs] == input_right[:rs]:
        return 'GEMM'

    # Einsum is faster than vectordot if we have to copy
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        return False

    # Conventional tensordot
    else:
        return 'TDOT'


def tensor_blas(view_left, input_left, view_right, input_right, index_result, idx_removed):
    """
    Computes the DOT product between two tensors

    Parameters
    ----------


    Returns
    -------
    type : array
        The resulting BLAS operation.

    Notes
    -----
    Currently does little checks to ensure the accuracy of results

    Examples
    --------
    >>> _can_blas(['ij', 'jk'], 'ik', set('j'))
    'GEMM'

    >>> _can_blas(['ijj', 'jk'], 'ik', set('j'))
    False

    """

    idx_removed = set(idx_removed)
    keep_left = set(input_left) - idx_removed
    keep_right = set(input_right) - idx_removed

    # We trust this must be called correctly
    dimension_dict = {}
    for i, s in zip(input_left, view_left.shape):
        dimension_dict[i] = s
    for i, s in zip(input_right, view_right.shape):
        dimension_dict[i] = s

    # Do we want to be able to do this?

    # Check for duplicate indices, cannot do einsum('iij,jkk->ik') operations here
    # if (len(set(input_left)) != len(input_left)):
    #     new_inds = ''.join(keep_left) + ''.join(idx_removed)
    #     view_left = np.einsum(input_left + '->' + new_inds, view_left, order='C')
    #     input_left = new_inds

    # if (len(set(input_right)) != len(input_right)):
    #     new_inds = ''.join(idx_removed) + ''.join(keep_right)
    #     view_right = np.einsum(input_right + '->' + new_inds, view_right, order='C')
    #     input_right = new_inds

    # Tensordot guarantees a copy for ndim > 2, should avoid skip if possible
    rs = len(idx_removed)
    dim_left = paths.compute_size_by_dict(keep_left, dimension_dict)
    dim_right = paths.compute_size_by_dict(keep_right, dimension_dict)
    dim_removed = paths.compute_size_by_dict(idx_removed, dimension_dict)
    tensor_result = input_left + input_right
    for s in idx_removed:
        tensor_result = tensor_result.replace(s, "")

    # This is ugly, but can vastly speed up certain operations
    # Vectordot
    if input_left == input_right:
        new_view = np.dot(view_left.ravel(), view_right.ravel())

    # Matrix multiply
    # No transpose needed
    elif input_left[-rs:] == input_right[:rs]:
        new_view = np.dot(view_left.reshape(dim_left, dim_removed),
                          view_right.reshape(dim_removed, dim_right))

    # Transpose both
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(view_left.reshape(dim_removed, dim_left).T,
                          view_right.reshape(dim_right, dim_removed).T)

    # Transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(view_left.reshape(dim_left, dim_removed),
                          view_right.reshape(dim_right, dim_removed).T)

    # Tranpose left
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(view_left.reshape(dim_removed, dim_left).T,
                          view_right.reshape(dim_removed, dim_right))

    # Conventional tensordot
    else:
        # Find indices to contract over
        left_pos, right_pos = (), ()
        for s in idx_removed:
            left_pos += (input_left.find(s),)
            right_pos += (input_right.find(s),)
        new_view = np.tensordot(view_left, view_right, axes=(left_pos, right_pos))

    # Make sure the resulting shape is correct
    tensor_shape = tuple(dimension_dict[x] for x in tensor_result)
    if (new_view.shape != tensor_shape):
        if (len(tensor_result) > 0):
            new_view.shape = tensor_shape
        else:
            new_view = np.squeeze(new_view)

    if tensor_result != index_result:
        new_view = np.einsum(tensor_result + '->' + index_result, new_view)

    return new_view
