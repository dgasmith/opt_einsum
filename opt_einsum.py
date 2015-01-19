import numpy as np


def _compute_size_by_dict(inds, ind_dict):
    """
    Computes the product of the elements in ind based on the
    dictionary ind_dict.
    """
    ret = 1
    for i in inds:
        ret *= ind_dict[i]
    return ret


def _find_contraction(positions, input_sets, output_set):
    """
    Finds the contraction for a given set of input and output sets
    positions - positions of the input_sets that are contracted
    input_sets - list of sets in the input
    output_set - output idx set
    returns:
      new_result - the indices of the resulting contraction
      remaining - list of sets that have not been contracted
      idx_removed - indices removed from the entire contraction
      idx_contract - the indices that are used in the contraction
    """

    idx_contract = set()
    idx_remain = output_set.copy()
    remaining = []
    for ind, value in enumerate(input_sets):
        if ind in positions:
            idx_contract |= value
        else:
            remaining.append(value)
            idx_remain |= value

    new_result = idx_remain & idx_contract
    idx_removed = (idx_contract - new_result)
    remaining.append(new_result)
    return (new_result, remaining, idx_removed, idx_contract)


def _path_optimal(input_sets, output_set, ind_dict, memory):
    """
    Computes all possible ways to contract the tensors
    input_sets - list of sets for input_setsut indices
    output_set - set of output_setput indices
    ind_dict - dictionary for the size of each idx
    memory - largest allowed number of elements in a new array
    returns path
    """

    current = [(0, [], input_sets)]
    for iteration in range(len(input_sets) - 1):
        new = []
        # Grab all unique pairs
        comb_iter = zip(*np.triu_indices(len(input_sets) - iteration, 1))
        for curr in current:
            cost, positions, remaining = curr
            for con in comb_iter:

                contract = _find_contraction(con, remaining, output_set)
                new_result, new_input_sets, idx_removed, idx_contract = contract

                # Sieve the results based on memory
                if _compute_size_by_dict(new_result, ind_dict) > memory:
                    continue

                # Find cost
                new_cost = _compute_size_by_dict(idx_contract, ind_dict)
                if len(idx_removed) > 0:
                    new_cost *= 2

                # Build (total_cost, positions, indices_remaining)
                new_cost += cost
                new_pos = positions + [con]
                new.append((new_cost, new_pos, new_input_sets))

        # Update list to iterate over
        current = new

    # If we have not found anything return single einsum contraction
    if len(new) == 0:
        return [tuple(range(len(input_sets)))]

    new.sort()
    path = new[0][1]
    return path


def _path_opportunistic(input_sets, output_set, ind_dict, memory):
    """
    Finds best path by choosing the best pair contraction
    Best pair is determined by the sorted tuple (-idx_removed, cost)
    input_sets - list of sets for input_setsut indices
    output_set - set of output_setput indices
    ind_dict - dictionary for the size of each idx
    memory - largest allowed number of elements in a new array
    returns path
    """

    path = []
    for iteration in range(len(input_sets) - 1):
        iteration_results = []
        comb_iter = zip(*np.triu_indices(len(input_sets), 1))
        for positions in comb_iter:

            contract = _find_contraction(positions, input_sets, output_set)
            idx_result, new_input_sets, idx_removed, idx_contract = contract

            # Sieve the results based on memory
            if _compute_size_by_dict(idx_result, ind_dict) > memory:
                continue

            # Build sort tuple
            removed_size = _compute_size_by_dict(idx_removed, ind_dict)
            cost = _compute_size_by_dict(idx_contract, ind_dict)
            sort = (-removed_size, cost)

            # Add contraction to possible choices
            iteration_results.append([sort, positions, new_input_sets])

        # If we did not find a new contraction contract remaining
        if len(iteration_results) == 0:
            path.append(tuple(range(len(input_sets))))
            break

        # Sort based on first idx
        iteration_results.sort()
        best = iteration_results[0]
        path.append(best[1])
        input_sets = best[2]

    return path


# Rewrite einsum to handle different cases
def contract(subscripts, *operands, **kwargs):
    """
    Attempts to contract tensors in an optimal order using both
    np.einsum and np.tensordot. Primarily aims at reducing the
    overall rank of the contration by building intermediates."

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    debug : bool, (default: False)
        Level of printing.
    tensordot : bool, optional (default: True)
        If true use tensordot where possible.
    path : bool or list, optional (default: `opportunistic`)
        Choose the type of path.

        - if a list is given uses this as the path.
        - 'opportunistic' means a N^3 algorithm that opportunistically
            chooses the best algorithm.
        - 'optimal' means a N! algorithm that tries all possible ways of
            contracting the listed tensors.

    memory : int, optional (default: largest input or output array size)
        Maximum number of elements in an intermediate array.

    Returns
    -------
    output : ndarray
        The results based on Einstein summation convention.

    See Also
    --------
    einsum, tensordot, dot

    """

    # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # 'abcdefghijklmnopqrstuvwxyz'

    # Split into input and output subscripts
    if '->' in subscripts:
        input_subscripts, output_subscript = subscripts.split('->')
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(',', '')
        output_subscript = ''
        for s in sorted(set(tmp_subscripts)):
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    if ('.' in input_subscripts) or ('.' in output_subscript):
        raise ValueError("Ellipsis are not currenly supported by contract.")

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    input_sets = map(set, input_list)
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))

    # Make sure number operands is equivalent to the number of terms
    if len(input_list) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the \
                          number of operands.")

    # Get length of each unique dimension and ensure all dimension are correct
    dimension_dict = {}
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError("einstein sum subscript %s does not contain the \
              correct number of indices for operand %d.", operands[tnum], tnum)
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            if char in dimension_dict.keys():
                if dimension_dict[char] != dim:
                    raise ValueError("Size of label '%s' for operand %d does \
                                      not match previous terms.", char, tnum)
            else:
                dimension_dict[char] = dim

    # TODO Should probably be cast up to double precision
    arr_dtype = np.result_type(*operands)
    operands = [np.asanyarray(v, dtype=arr_dtype) for v in operands]

    # Compute size of each input array plus the output array
    size_list = []
    for term in input_list + [output_subscript]:
        size_list.append(_compute_size_by_dict(term, dimension_dict))
    out_size = max(size_list)

    # Grab a few kwargs
    tdot_arg = kwargs.get("tensordot", True)
    path_arg = kwargs.get("path", "opportunistic")
    memory_arg = kwargs.get("memory", out_size)
    return_path_arg = kwargs.get("return_path", False)

    # If total flops is very small just avoid the overhead altogether
    total_flops = _compute_size_by_dict(indices, dimension_dict)
    # if (total_flops < 1e6) and not return_path_arg:
    #     return np.einsum(subscripts, *operands)

    # If no rank reduction leave it to einsum
    if (indices == output_set) and not return_path_arg:
        return np.einsum(subscripts, *operands)

    # Compute path
    if not isinstance(path_arg, str):
        path = path_arg
    elif len(input_list) == 1:
        path = [(0)]
    elif len(input_list) == 2:
        path = [(0, 1)]
    elif path_arg == "opportunistic":
        # Maximum memory should be at most out_size for this algorithm
        memory_arg = min(memory_arg, out_size)
        path = _path_opportunistic(input_sets, output_set, dimension_dict, memory_arg)
    elif path_arg == "optimal":
        path = _path_optimal(input_sets, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError("Path name %s not found", path_arg)

    contraction_list = []
    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = sorted(list(contract_inds), reverse=True)

        contract = _find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract

        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:    
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = ''.join([x[1] for x in sorted(sort_result)])

        input_list.append(idx_result)
        einsum_str = ','.join(tmp_inputs) + '->' + idx_result
        contraction = (contract_inds, False, einsum_str, input_list[:])
        contraction_list.append(contraction)

    # Return the path along with a nice string representation
    if return_path_arg:
        overall_contraction = input_subscripts + '->' + output_subscript
        header = ('scaling', 'GEMM', 'current', 'remaining')

        path_print = 'Complete contraction:  %s\n' % overall_contraction
        path_print += '       Naive scaling:%4d\n' % len(indices)
        path_print += '-' * 80 + '\n'
        path_print += '%6s %6s %24s %40s\n' % header
        path_print += '-' * 80 + '\n'

        for inds, gemm, einsum_str, remaining in contraction_list:
            remaining_str = ','.join(remaining) + '->' + output_subscript
            path_run = (len(idx_contract), gemm, einsum_str, remaining_str)
            path_print += '%4d    %6s %24s %40s\n' % path_run

        return (path, path_print)

    # Start contraction loop
    for inds, gemm, einsum_str, remaining in contraction_list:
        tmp_operands = []
        for x in inds:
            tmp_operands.append(operands.pop(x))

        # Do the contraction
        new_view = np.einsum(einsum_str, *tmp_operands, order='C')

        # Append new items
        operands.append(new_view)
        del tmp_operands, new_view  # Dereference what we can

    return operands[0]
