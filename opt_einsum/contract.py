"""
Contains the primary optimization and contraction routines
"""

import numpy as np

from . import blas
from . import helpers
from . import paths
from . import parser


def contract_path(*operands, **kwargs):
    """
    Evaluates the lowest cost einsum-like contraction order.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    path : bool or list, optional (default: ``greedy``)
        Choose the type of path.

        - if a list is given uses this as the path.
        - 'greedy' An algorithm that chooses the best pair contraction
            at each step. Scales cubically with the number of terms in the
            contraction.
        - 'optimal' An algorithm that tries all possible ways of
            contracting the listed tensors. Scales exponentially with
            the number of terms in the contraction.
    use_blas : bool
        Use BLAS functions or not

    memory_limit : int, optional (default: largest input or output array size)
        Maximum number of elements allowed in intermediate arrays.

    Returns
    -------
    path : list of tuples
        The einsum path
    string_repr : str
        A printable representation of the path

    Notes
    -----
    The resulting path indicates which terms of the input contraction should be
    contracted first, the result of this contraction is then appended to the end of
    the contraction list.

    Examples
    --------

    We can begin with a chain dot example. In this case it is optimal to
    contract the b and c tensors reprsented by the first element of the path (1,
    2). The resulting tensor is added to the end of the contraction and the
    remaining contraction (0, 1) is then completed.

    >>> a = np.random.rand(2, 2)
    >>> b = np.random.rand(2, 5)
    >>> c = np.random.rand(5, 2)
    >>> path_info = opt_einsum.contract_path('ij,jk,kl->il', a, b, c)
    >>> print(path_info[0])
    [(1, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ij,jk,kl->il
             Naive scaling:  4
         Optimized scaling:  3
          Naive FLOP count:  1.600e+02
      Optimized FLOP count:  5.600e+01
       Theoretical speedup:  2.857
      Largest intermediate:  4.000e+00 elements
    -------------------------------------------------------------------------
    scaling                  current                                remaining
    -------------------------------------------------------------------------
       3                   kl,jk->jl                                ij,jl->il
       3                   jl,ij->il                                   il->il


    A more complex index transformation example.

    >>> I = np.random.rand(10, 10, 10, 10)
    >>> C = np.random.rand(10, 10)
    >>> path_info = oe.contract_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C)

    >>> print(path_info[0])
    [(0, 2), (0, 3), (0, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ea,fb,abcd,gc,hd->efgh
             Naive scaling:  8
         Optimized scaling:  5
          Naive FLOP count:  8.000e+08
      Optimized FLOP count:  8.000e+05
       Theoretical speedup:  1000.000
      Largest intermediate:  1.000e+04 elements
    --------------------------------------------------------------------------
    scaling                  current                                remaining
    --------------------------------------------------------------------------
       5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
       5               bcde,fb->cdef                         gc,hd,cdef->efgh
       5               cdef,gc->defg                            hd,defg->efgh
       5               defg,hd->efgh                               efgh->efgh
    """

    # Make sure all keywords are valid
    valid_contract_kwargs = ['path', 'memory_limit', 'einsum_call', 'use_blas']
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_contract_kwargs]
    if len(unknown_kwargs):
        raise TypeError("einsum_path: Did not understand the following kwargs: %s" % unknown_kwargs)

    path_type = kwargs.pop('path', 'greedy')
    memory_limit = kwargs.pop('memory_limit', None)

    # Hidden option, only einsum should call this
    einsum_call_arg = kwargs.pop("einsum_call", False)
    use_blas = kwargs.pop('use_blas', True)

    # Python side parsing
    input_subscripts, output_subscript, operands = parser.parse_einsum_input(operands)
    subscripts = input_subscripts + '->' + output_subscript

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape

        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript %s does not contain the "
                             "correct number of indices for operand %d.", input_subscripts[tnum], tnum)
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            if char in dimension_dict.keys():
                if dimension_dict[char] != dim:
                    raise ValueError("Size of label '%s' for operand %d does " "not match previous terms.", char, tnum)
            else:
                dimension_dict[char] = dim

    # Compute size of each input array plus the output array
    size_list = []
    for term in input_list + [output_subscript]:
        size_list.append(helpers.compute_size_by_dict(term, dimension_dict))
    out_size = max(size_list)

    if memory_limit is None:
        memory_arg = out_size
    else:
        if memory_limit < 1:
            if memory_limit == -1:
                memory_arg = int(1e20)
            else:
                raise ValidationError("Memory limit must be larger than 0, or -1")
        else:
            memory_arg = int(memory_limit)

    # Compute naive cost
    # This isnt quite right, need to look into exactly how einsum does this
    # indices_in_input = input_subscripts.replace(',', '')
    # inne
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = helpers.flop_count(indices, inner_product, len(input_list), dimension_dict)

    # Compute the path
    if not isinstance(path_type, str):
        path = path_type
    elif len(input_list) == 1:
        # Nothing to be optimized
        path = [(0, )]
    elif len(input_list) == 2:
        # Nothing to be optimized
        path = [(0, 1)]
    elif indices == output_set:
        # If no rank reduction leave it to einsum
        path = [tuple(range(len(input_list)))]
    elif path_type in ["greedy", "opportunistic"]:
        path = paths.greedy(input_sets, output_set, dimension_dict, memory_arg)
    elif path_type == "optimal":
        path = paths.optimal(input_sets, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError("Path name %s not found", path_type)

    cost_list = []
    scale_list = []
    size_list = []
    contraction_list = []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))

        contract = helpers.find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract

        # Compute cost, scale, and size
        cost = helpers.flop_count(idx_contract, idx_removed, len(contract_inds), dimension_dict)
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(helpers.compute_size_by_dict(out_inds, dimension_dict))

        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))

        if use_blas:
            do_blas = blas.can_blas(tmp_inputs, out_inds, idx_removed)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = "".join([x[1] for x in sorted(sort_result)])

        input_list.append(idx_result)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        contraction = (contract_inds, idx_removed, einsum_str, input_list[:], do_blas)
        contraction_list.append(contraction)

    opt_cost = sum(cost_list)

    if einsum_call_arg:
        return operands, contraction_list

    # Return the path along with a nice string representation
    overall_contraction = input_subscripts + "->" + output_subscript
    header = ("scaling", "BLAS", "current", "remaining")

    path_print = "  Complete contraction:  %s\n" % overall_contraction
    path_print += "         Naive scaling:  %d\n" % len(indices)
    path_print += "     Optimized scaling:  %d\n" % max(scale_list)
    path_print += "      Naive FLOP count:  %.3e\n" % naive_cost
    path_print += "  Optimized FLOP count:  %.3e\n" % opt_cost
    path_print += "   Theoretical speedup:  %3.3f\n" % (naive_cost / float(opt_cost))
    path_print += "  Largest intermediate:  %.3e elements\n" % max(size_list)
    path_print += "-" * 80 + "\n"
    path_print += "%6s %6s %24s %40s\n" % header
    path_print += "-" * 80

    for n, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, do_blas = contraction
        remaining_str = ",".join(remaining) + "->" + output_subscript
        path_run = (scale_list[n], do_blas, einsum_str, remaining_str)
        path_print += "\n%4d %9s %24s %40s" % path_run

    return path, path_print


# Rewrite einsum to handle different cases
def contract(*operands, **kwargs):
    """
    contract(subscripts, *operands, out=None, dtype=None, order='K',
           casting='safe', use_blas=False, optimize=True, memory_limit=None)

    Evaluates the Einstein summation convention on the operands. A drop in
    replacment for NumPy's einsum function that optimizes the order of contraction
    to reduce overall scaling at the cost of several intermediate arrays.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    out : array_like
        A output array in which set the resulting output.
    dtype : str
        The dtype of the given contraction, see np.einsum.
    order : str
        The order of the resulting contraction, see np.einsum.
    casting : str
        The casting procedure for operations of different dtype, see np.einsum.
    use_blas : bool
        Do you use BLAS for valid operations, may use extra memory for more intermediates.
    optimize : bool, str, or list, optional (default: ``greedy``)
        Choose the type of path.

        - if a list is given uses this as the path.
        - 'greedy' An algorithm that chooses the best pair contraction
            at each step. Scales cubically with the number of terms in the
            contraction.
        - 'optimal' An algorithm that tries all possible ways of
            contracting the listed tensors. Scales exponentially with
            the number of terms in the contraction.

    memory_limit : int or None (default : None)
        The upper limit of the size of tensor created, by default this will be
        Give the upper bound of the largest intermediate tensor contract will build.
        By default (None) will size the ``memory_limit`` as the largest input tensor.
        Users can also specify ``-1`` to allow arbitrarily large tensors to be built.

    Returns
    -------
    out : array_like
        The result of the einsum expression.

    Notes
    -----
    This function should produce result identical to that of NumPy's einsum
    function. The primary difference is `contract` will attempt to form
    intermediates which reduce the overall scaling of the given einsum contraction.
    By default the worst intermediate formed will be equal to that of the largest
    input array. For large einsum expressions with many input arrays this can
    provide arbitrarily large (1000 fold+) speed improvements.

    For contractions with just two tensors this function will attempt to use
    NumPy's built in BLAS functionality to ensure that the given operation is
    preformed in an optimal manner. When NumPy is linked to a threaded BLAS, potenital
    speedsups are on the order of 20-100 for a six core machine.

    Examples
    --------

    See opt_einsum.contract_path or numpy.einsum

    """

    # Grab non-einsum kwargs
    optimize_arg = kwargs.pop('optimize', True)
    if optimize_arg is True:
        optimize_arg = 'greedy'

    use_blas = kwargs.pop('use_blas', True)

    valid_einsum_kwargs = ['out', 'dtype', 'order', 'casting']
    einsum_kwargs = {k: v for (k, v) in kwargs.items() if k in valid_einsum_kwargs}

    # If no optimization, run pure einsum
    if optimize_arg is False:
        return np.einsum(*operands, **einsum_kwargs)

    # Make sure all keywords are valid
    valid_contract_kwargs = ['memory_limit', 'use_blas'] + valid_einsum_kwargs
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_contract_kwargs]
    if len(unknown_kwargs):
        raise TypeError("Did not understand the following kwargs: %s" % unknown_kwargs)

    # Special handeling if out is specified
    specified_out = False
    out_array = einsum_kwargs.pop('out', None)
    if out_array is not None:
        specified_out = True

    # Build the contraction list and operand
    memory_limit = kwargs.pop('memory_limit', None)

    operands, contraction_list = contract_path(
        *operands, path=optimize_arg, memory_limit=memory_limit, einsum_call=True, use_blas=use_blas)

    handle_out = False

    # Start contraction loop
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        tmp_operands = []
        for x in inds:
            tmp_operands.append(operands.pop(x))

        # Do we need to deal with the output?
        if specified_out and ((num + 1) == len(contraction_list)):
            handle_out = True

        # Call tensordot
        if blas:

            # Checks have already been handled
            input_str, results_index = einsum_str.split('->')
            input_left, input_right = input_str.split(',')

            tensor_result = input_left + input_right
            for s in idx_rm:
                tensor_result = tensor_result.replace(s, "")

            # Find indices to contract over
            left_pos, right_pos = [], []
            for s in idx_rm:
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))

            # Contract!
            new_view = np.tensordot(*tmp_operands, axes=(tuple(left_pos), tuple(right_pos)))

            # Build a new view if needed
            if (tensor_result != results_index) or handle_out:
                if handle_out:
                    einsum_kwargs["out"] = out_array
                new_view = np.einsum(tensor_result + '->' + results_index, new_view, **einsum_kwargs)

        # Call einsum
        else:
            # If out was specified
            if handle_out:
                einsum_kwargs["out"] = out_array

            # Do the contraction
            new_view = np.einsum(einsum_str, *tmp_operands, **einsum_kwargs)

        # Append new items and derefernce what we can
        operands.append(new_view)
        del tmp_operands, new_view

    if specified_out:
        return out_array
    else:
        return operands[0]
