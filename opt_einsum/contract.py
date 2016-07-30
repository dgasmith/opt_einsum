from . import paths
from . import parser
from . import blas
import numpy as np

def contract_path(*operands, **kwargs):
    """
    Evaluates the lowest cost contraction order for a given set of contraints.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    path : bool or list, optional (default: ``greedy``)
        Choose the type of path.

        - if a list is given uses this as the path.
        - 'greedy' A N^3 algorithm that chooses the best pair contraction
            at each step.
        - 'optimal' means a N! algorithm that tries all possible ways of
            contracting the listed tensors.

    memory_limit : int, optional (default: largest input or output array size)
        Maximum number of elements allowed in an intermediate array.

    Returns
    -------
    path : list
        The einsum path
    string_representation : str
        A printable representation of the path

    Notes
    -----
    A path is a list of tuples where the each tuple represents a single
    contraction. For each tuple, the operands involved in the contraction are popped
    and the array resulting from the contraction is appended to the end of the
    operand list.

    Examples
    --------
    >>> I = np.random.rand(10, 10, 10, 10)
    >>> C = np.random.rand(10, 10)
    >>> opt_path = np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C, return_path=True)

    >>> print(opt_path[0])
    [(2, 0), (3, 0), (2, 0), (1, 0)]
    >>> print(opt_path[1])
      Complete contraction:  ea,fb,abcd,gc,hd->efgh
             Naive scaling:   8
          Naive FLOP count:  1.000e+09
      Optimized FLOP count:  8.000e+05
       Theoretical speedup:  1250.000
      Largest intermediate:  1.000e+04 elements
    --------------------------------------------------------------------------------
    scaling   BLAS                  current                                remaining
    --------------------------------------------------------------------------------
       5     False            abcd,ea->bcde                      fb,gc,hd,bcde->efgh
       5     False            bcde,fb->cdef                         gc,hd,cdef->efgh
       5     False            cdef,gc->defg                            hd,defg->efgh
       5     False            defg,hd->efgh                               efgh->efgh
    """

    path_arg = kwargs.pop("path", "greedy")
    memory_limit = kwargs.pop('memory_limit', None)
    tensordot = kwargs.pop('tensordot', True)

    # Hidden option, only einsum should call this
    einsum_call_arg = kwargs.pop("einsum_call", False)

    # Python side parsing
    input_subscripts, output_subscript, operands = parser.parse_einsum_input(operands)
    subscripts = input_subscripts + '->' + output_subscript

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))

    # Get length of each unique dimension and ensure all dimension are correct
    dimension_dict = {}
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript %s does not contain the "
                             "correct number of indices for operand %d.",
                             input_subscripts[tnum], tnum)
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            if char in dimension_dict.keys():
                if dimension_dict[char] != dim:
                    raise ValueError("Size of label '%s' for operand %d does "
                                     "not match previous terms.", char, tnum)
            else:
                dimension_dict[char] = dim

    # Compute size of each input array plus the output array
    if memory_limit is None:
        size_list = []
        for term in input_list + [output_subscript]:
            size_list.append(paths.compute_size_by_dict(term, dimension_dict))
        out_size = max(size_list)
        memory_arg = out_size
    else:
        memory_arg = int(memory_limit)

    # Compute naive cost
    # This isnt quite right, need to look into exactly how einsum does this
    naive_cost = paths.compute_size_by_dict(indices, dimension_dict)
    indices_in_input = input_subscripts.replace(',', '')
    mult = max(len(input_list) - 1, 1)
    if (len(indices_in_input) - len(set(indices_in_input))):
        mult *= 2
    naive_cost *= mult

    # Compute path
    if not isinstance(path_arg, str):
        path = path_arg
    elif len(input_list) == 1:
        path = [(0,)]
    elif len(input_list) == 2:
        path = [(0, 1)]
    elif (indices == output_set):
        # If no rank reduction leave it to einsum
        path = [tuple(range(len(input_list)))]
    elif (path_arg in ["greedy", "opportunistic"]):
        # Maximum memory should be at most out_size for this algorithm
        memory_arg = min(memory_arg, out_size)
        path = paths.greedy(input_sets, output_set, dimension_dict, memory_arg)
    elif path_arg == "optimal":
        path = paths.optimal(input_sets, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError("Path name %s not found", path_arg)

    cost_list, scale_list, size_list = [], [], []
    contraction_list = []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))

        contract = paths.find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract

        cost = paths.compute_size_by_dict(idx_contract, dimension_dict)
        if idx_removed:
            cost *= 2
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(paths.compute_size_by_dict(out_inds, dimension_dict))

        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = "".join([x[1] for x in sorted(sort_result)])

        input_list.append(idx_result)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        if tensordot:
            can_gemm = blas.can_blas(tmp_inputs, idx_result, idx_removed)
            # Dont want to deal with this quite yet
            if can_gemm == 'TDOT':
                can_gemm = False
        else:
            can_gemm = False

        contraction = (contract_inds, idx_removed, can_gemm, einsum_str, input_list[:])
        contraction_list.append(contraction)

    opt_cost = sum(cost_list)

    if einsum_call_arg:
        return (operands, contraction_list)

    # Return the path along with a nice string representation
    overall_contraction = input_subscripts + "->" + output_subscript
    header = ("scaling", "BLAS", "current", "remaining")

    path_print  = "  Complete contraction:  %s\n" % overall_contraction
    path_print += "         Naive scaling:  %2d\n" % len(indices)
    path_print += "      Naive FLOP count:  %.3e\n" % naive_cost
    path_print += "  Optimized FLOP count:  %.3e\n" % opt_cost
    path_print += "   Theoretical speedup:  %3.3f\n" % (naive_cost / float(opt_cost))
    path_print += "  Largest intermediate:  %.3e elements\n" % max(size_list)
    path_print += "-" * 80 + "\n"
    path_print += "%6s %6s %24s %40s\n" % header
    path_print += "-" * 80

    for n, contraction in enumerate(contraction_list):
        inds, idx_rm, gemm, einsum_str, remaining = contraction
        remaining_str = ",".join(remaining) + "->" + output_subscript
        path_run = (scale_list[n], gemm, einsum_str, remaining_str)
        path_print += "\n%4d    %6s %24s %40s" % path_run

    return (path, path_print)


# Rewrite einsum to handle different cases
def contract(*operands, **kwargs):
    """
    Evaluates the Einstein summation convention based on the operands,
    differs from einsum by utilizing intermediate arrays to
    reduce overall computational time.

    Produces results identical to that of the einsum function; however,
    the contract function expands on the einsum function by building
    intermediate arrays to reduce the computational scaling and utilizes
    BLAS calls when possible.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    tensordot : bool, optional (default: True)
        If true use tensordot where possible.
    path : bool or list, optional (default: ``greedy``)
        Choose the type of path.

        - if a list is given uses this as the path.
        - 'greedy' means a N^3 algorithm that greedily
            chooses the best algorithm.
        - 'optimal' means a N! algorithm that tries all possible ways of
            contracting the listed tensors.

    memory_limit : int, optional (default: largest input or output array size)
        Maximum number of elements allowed in an intermediate array.


    Returns
    -------
    output : ndarray
        The results based on Einstein summation convention.

    See Also
    --------
    einsum, tensordot, dot

    Notes
    -----
    Subscript labels follow the same convention as einsum with the exception
    that integer indexing and ellipses are not currently supported.

    The amount of extra memory used by this function depends greatly on the
    einsum expression and BLAS usage.  Without BLAS the maximum memory used is:
    ``(number_of_terms / 2) * memory_limit``.  With BLAS the maximum memory used
    is: ``max((number_of_terms / 2), 2) * memory_limit``.  For most operations
    the memory usage is approximately equivalent to the memory_limit.

    Note: BLAS is not yet implemented in this branch.
    One operand operations are supported by calling ``np.einsum``.  Two operand
    operations are first checked to see if a BLAS call can be utilized then
    defaulted to einsum.  For example ``np.contract('ab,bc->', a, b)`` and
    ``np.contract('ab,cb->', a, b)`` are prototypical matrix matrix multiplication
    examples.  Higher dimensional matrix matrix multiplicaitons are also considered
    such as ``np.contract('abcd,cdef', a, b)`` and ``np.contract('abcd,cefd', a,
    b)``.  For the former, GEMM can be called without copying data; however, the
    latter requires a copy of the second operand.  For all matrix matrix
    multiplication examples it is beneficial to copy the data and call GEMM;
    however, for matrix vector multiplication it is not beneficial to do so.  For
    example ``np.contract('abcd,cd', a, b)`` will call GEMV while
    ``np.contract('abcd,ad', a, b)`` will call einsum as copying the first operand
    then calling GEMV does not provide a speed up compared to calling einsum.

    For three or more operands contract computes the optimal order of two and
    one operand operations.  The ``optimal`` path scales like N! where N is the
    number of terms and is found by calculating the cost of every possible path and
    choosing the lowest cost.  This path can be more costly to compute than the
    contraction itself for a large number of terms (~N>7).  The ``greedy``
    path scales like N^3 and first tries to do any matrix matrix multiplications,
    then inner products, and finally outer products.  This path usually takes a
    trivial amount of time to compute unless the number of terms is extremely large
    (~N>20).  The greedy path typically computes the most optimal path, but
    is not guaranteed to do so.  Both of these algorithms are sieved by the
    variable memory to prevent the formation of very large tensors.

    Examples
    --------
    A index transformation example, the optimized version runs ~2000 times faster than
    conventional einsum even for this small example.

    >>> I = np.random.rand(10, 10, 10, 10)
    >>> C = np.random.rand(10, 10)
    >>> opt_result = np.einsum('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C, optimize=True)
    >>> ein_result = np.einsum('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C)
    >>> np.allclose(ein_result, opt_result)
    True
    """

    # Grab non-einsum kwargs
    optimize_arg = kwargs.pop('optimize', True)
    if optimize_arg is True:
        optimize_arg = 'greedy'

    valid_einsum_kwargs = ['out', 'dtype', 'order', 'casting']
    einsum_kwargs = {k: v for (k, v) in kwargs.items() if k in valid_einsum_kwargs}

    # If no optimization, run pure einsum
    if (optimize_arg is False):
        return np.einsum(*operands, **einsum_kwargs)

    # Make sure all keywords are valid
    valid_contract_kwargs = ['tensordot', 'path', 'memory_limit'] + valid_einsum_kwargs
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_contract_kwargs]
    if len(unknown_kwargs):
        raise TypeError("Contract: Did not understand the following kwargs: %s" % unknown_kwargs)

    use_tensordot = kwargs.pop('tensordot', True)

    # Special handeling if out is specified
    specified_out = False
    out_array = kwargs.pop('out', None)
    if out_array is not None:
        specified_out = True

    # Build the contraction list and operand
    memory_limit = kwargs.pop('memory_limit', None)
    operands, contraction_list = contract_path(*operands, path=optimize_arg,
                                               memory_limit=memory_limit,
                                               einsum_call=True,
                                               tensordot=use_tensordot)

    # Start contraction loop
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, gemm, einsum_str, remaining = contraction
        tmp_operands = []
        for x in inds:
            tmp_operands.append(operands.pop(x))

        # If out was specified
        if specified_out and ((num + 1) == len(contraction_list)):
            kwargs["out"] = out_array

        # Do the contraction
        if gemm is False:
            new_view = np.einsum(einsum_str, *tmp_operands, **einsum_kwargs)
        else:
            inputs, result = einsum_str.split('->')
            inds1, inds2 = inputs.split(',')
            new_view = blas.tensor_blas(tmp_operands[0], inds1,
                                        tmp_operands[1], inds2,
                                        result, idx_rm)

            # Poor way to handle this for now
            if 'out' in kwargs:
                kwargs['out'][:] = new_view

        # Append new items and derefernce what we can
        operands.append(new_view)
        del tmp_operands, new_view

    if specified_out:
        return out_array
    else:
        return operands[0]
