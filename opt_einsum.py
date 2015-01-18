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
    output_set - output index set
    returns:
      new_result - the indices of the resulting contraction
      remaining - list of sets that have not been contracted
      index_removed - indices removed from the entire contraction
      index_contract - the indices that are used in the contraction
    """

    index_contract = set()
    index_remain = output_set.copy()
    remaining = []
    for ind, value in enumerate(input_sets):
        if ind in positions:
            index_contract |= value
        else:
            remaining.append(value)
            index_remain |= value

    new_result = index_remain & index_contract
    index_removed = (index_contract - new_result)
    remaining.append(new_result)
    return (new_result, remaining, index_removed, index_contract)


def _path_optimal(inp, out, ind_dict, memory):
    """
    Computes all possible ways to contract the tensors
    inp - list of sets for input indices
    out - set of output indices
    ind_dict - dictionary for the size of each index
    memory - largest allowed number of elements in a new array
    returns path
    """

    inp_set = map(set, inp)
    out_set = set(out)

    current = [(0, [], inp_set)]
    for iteration in range(len(inp) - 1):
        new = []
        # Grab all unique pairs
        comb_iter = zip(*np.triu_indices(len(inp) - iteration, 1))
        for curr in current:
            cost, positions, remaining = curr
            for con in comb_iter:

                contract = _find_contraction(con, remaining, out_set)
                new_result, new_inp, index_removed, index_contract = contract

                # Sieve the results based on memory, prevents unnecessarily large tensors
                if _compute_size_by_dict(new_result, ind_dict) > memory:
                    continue

                # Find cost
                new_cost = _compute_size_by_dict(index_contract, ind_dict)
                if len(index_removed) > 0:
                    new_cost *= 2

                # Build (total_cost, positions, indices_remaining)
                new_cost += cost
                new_pos = positions + [con]
                new.append((new_cost, new_pos, new_inp))

        # Update list to iterate over
        current = new

    # If we have not found anything return single einsum contraction
    if len(new) == 0:
        return [tuple(range(len(inp)))]

    new.sort()
    path = new[0][1]
    return path


def _path_opportunistic(inp, out, ind_dict, memory):
    """
    Finds best path by choosing the best pair contraction
    Best pair is determined by the sorted tuple (-index_removed, cost)
    inp - list of sets for input indices
    out - set of output indices
    ind_dict - dictionary for the size of each index
    memory - largest allowed number of elements in a new array
    returns path
    """

    inp_set = map(set, inp)
    out_set = set(out)

    path = []
    for iteration in range(len(inp) - 1):
        iteration_results = []
        comb_iter = zip(*np.triu_indices(len(inp_set), 1))
        for positions in comb_iter:

            contract = _find_contraction(positions, inp_set, out_set)
            index_result, new_inp, index_removed, index_contract = contract

            # Sieve the results based on memory, prevents unnecessarily large tensors
            if _compute_size_by_dict(index_result, ind_dict) > memory:
                continue

            # Build sort tuple
            removed_size = _compute_size_by_dict(index_removed, ind_dict)
            cost = _compute_size_by_dict(index_contract, ind_dict)
            sort = (-removed_size, cost)

            # Add contraction to possible choices
            iteration_results.append([sort, positions, new_inp])

        # If we did not find a new contraction contract remaining
        if len(iteration_results) == 0:
            path.append(tuple(range(len(inp) - iteration)))
            break

        # Sort based on first index
        iteration_results.sort()
        best = iteration_results[0]
        path.append(best[1])
        inp_set = best[2]

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
    input_set = map(set, input_list)
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))

    # TODO Should probably be cast up to double precision
    arr_dtype = np.result_type(*operands)
    operands = [np.asanyarray(v, dtype=arr_dtype) for v in operands]

    # Make sure number operands is equivalent to the number of terms
    if len(input_list) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the \
                          number of operands.")

    # Get length of each unique index and ensure all dimension are correct
    inds_left = indices.copy()
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

    # Compute size of each input array plus the output array
    size_list = []
    for term in input_list + [output_subscript]:
        size_list.append(_compute_size_by_dict(term, dimension_dict))
    out_size = max(size_list)

    # Grab a few kwargs
    debug_arg = kwargs.get("debug", False)
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

    if debug_arg:
        print('Complete contraction:  %s' % (input_subscripts + '->' + output_subscript))
        print('       Naive scaling:%4d' % len(indices))

    # Compute path
    if not isinstance(path_arg, str):
        path = path_arg
    elif len(input_list) == 1:
        path = [(0)]
    elif len(input_list) == 2:
        path = [(0, 1)]
    elif path_arg == "opportunistic":
        # Maximum memory is an important variable here, should be at most out_size
        memory_arg = min(memory_arg, out_size)
        path = _path_opportunistic(input_list, output_set, dimension_dict, memory_arg)
    elif path_arg == "optimal":
        path = _path_optimal(input_list, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError("Path name %s not found", path_arg)

    # Return path if requested
    if return_path_arg:
        return path

    # Only a single operand - leave it to einsum
    if path[0] == (0):
        return np.einsum(subscripts, operands[0])

    if debug_arg:
        print('-' * 80)
        print('%6s %6s %24s %40s' % ('scaling', 'GEMM', 'current', 'remaining'))
        print('-' * 80)

    # Start contraction loop
    for contract_inds in path:
        # Make sure we remove inds from right to left
        contract_inds = sorted(list(contract_inds), reverse=True)

        contract = _find_contraction(contract_inds, input_set, output_set)
        out_inds, input_set, index_removed, index_contract = contract

        # Build required structures and explicitly delete operands
        # Make sure to loop from right to left
        tmp_operands, tmp_input = [], []
        for x in contract_inds:
            tmp_operands.append(operands.pop(x))
            tmp_input.append(input_list.pop(x))

        # We can choose order of output indices, shortest first
        sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
        index_result = ''.join([x[1] for x in sorted(sort_result)])
        einsum_subscripts = ','.join(tmp_input) + '->' + index_result
        new_view = np.einsum(einsum_subscripts, *tmp_operands, order='C')

        # Print current contraction
        if debug_arg:
            einsum_subscripts = ','.join(tmp_input) + '->' + index_result
            remaining = ','.join(input_list + [index_result]) + '->' + output_subscript
            print('%4d    %6s %24s %40s' % (len(index_contract), can_tdot, einsum_subscripts, remaining))

        # Append new items
        operands += [new_view]
        input_list += [index_result]
        del tmp_operands, new_view  # Dereference what we can

    # We may need to do a final transpose
    if input_list[0] == output_subscript:
        return operands[0]
    else:
        return np.einsum(input_list[0] + '->' + output_subscript, operands[0], order='C').copy()


