import time
import numpy as np


def _compute_size(inds, ind_dict):
    # Computes the product of indices based on a dictionary
    ret = 1
    for i in inds:
        ret *= ind_dict[i]
    return ret


def _find_contraction(positions, input_sets, output_set):
    # Finds the contraction for a given set of input and output sets
    # positions - positions of the input_sets that are contracted
    # input_sets - list of sets in the input
    # output_set - output index set
    # returns:
    #   new_result - the indices of the resulting contraction
    #   remaining - list of sets that have not been contracted
    #   index_removed - indices removed from the entire contraction
    #   index_contract - the indices that are used in the contraction

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
    # Computes all possible ways to contract the tensors
    # inp - list of sets for input indices
    # out - set of output indices
    # ind_dict - dictionary for the size of each index
    # memory - largest allowed number of elements in a new array
    # returns path

    inp_set = map(set, inp)
    out_set = set(out)

    t = time.time()
    current = [(0, [], [], inp_set)]
    for iteration in range(len(inp) - 1):
        new = []
        # Grab all unique pairs
        comb_iter = zip(*np.triu_indices(len(inp) - iteration, 1))
        for curr in current:
            cost, positions, result, remaining = curr
            for con in comb_iter:

                contract = _find_contraction(con, remaining, out_set)
                new_result, new_inp, index_removed, index_contract = contract

                # Sieve the results based on memory, prevents unnecessarly large tensors
                if _compute_size(new_result, ind_dict) > memory:
                    continue

                # Find cost
                new_cost = _compute_size(index_contract, ind_dict)
                if len(index_removed) > 0:
                    new_cost *= 2

                # Build (total_cost, positions, results, indices_remaining)
                new_cost += cost
                new_result = result + [new_result]
                new_pos = positions + [con]
                new.append((new_cost, new_pos, new_result, new_inp))

        current = new

    # If we have not found anything return single einsum contraction
    if len(new) == 0:
        return [tuple(range(len(inp)))]

    new.sort()
    path = new[0][1]
    einsum_string = ','.join(inp) + '->' + ''.join(out)
    print 'Path Optimal time: %6d %2d %3.5f %40s' % (len(new), len(inp), (time.time()-t), einsum_string)

    return path


def _path_opportunistic(inp, out, ind_dict, memory):
    # Finds best path by choosing the best pair contraction
    # inp - list of sets for input indices
    # out - set of output indices
    # ind_dict - dictionary for the size of each index
    # memory - largest allowed number of elements in a new array
    # returns path

    inp_set = map(set, inp)
    out_set = set(out)

    path = []
    for iteration in range(len(inp) - 1):
        iteration_results = []
        comb_iter = zip(*np.triu_indices(len(inp_set), 1))
        for positions in comb_iter:

            contract = _find_contraction(positions, inp_set, out_set)
            index_result, new_inp, index_removed, index_contract = contract

            # Sieve the results based on memory, prevents unnecessarly large tensors
            if _compute_size(index_result, ind_dict) > memory:
                continue

            # Build sort tuple
            removed_size = _compute_size(index_removed, ind_dict)
            cost = _compute_size(index_contract, ind_dict)
            sort = (-removed_size, cost)

            # Add contraction to possible choices
            iteration_results.append([sort, positions, new_inp])

        # If we didnt find a new contraction contract the rest at the same time
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
def opt_einsum(string, *views, **kwargs):
    """
    Attempts to contract tensors in an optimal order using both
    np.einsum and np.tensordot. Primarily aims at reducing the
    overall rank of the contration by building intermediates."

    Parameters
    __________
    string : str
        Einsum string of contractions
    *view : list of views utilized
    debug : int (default: 0)
        Level of printing.
    tensordot : bool, optional
        If true use tensordot where possible.
    path : bool or list, optional
        Choose the type of path.

        - if a list is given uses this as the path.
        - 'opportunistic' means a N^2 algorithm that opportunistically
            chooses the best algorithm.
        - 'optimal' means a N! algorithm that tries all possible ways of
            contracting the listed tensors.

    memory : int, optional
        Maximum number of elements in an array. Defaults to the size of the
            largest ndarry view our output.

    Returns
    -------
    output : ndarray
        The result based on the Einstein summation convention.
    """

    # Split into output and input string
    if '->' in string:
        input_string, output_string = string.split('->')
    else:
        input_string = string
        # Build output string
        tmp_string = string.replace(',', '')
        output_string = ''
        for s in set(tmp_string):
            if tmp_string.count(s) == 1:
                output_string += s

    # Build a few useful list and sets
    input_list = input_string.split(',')
    input_set = map(set, input_list)
    output_set = set(output_string)
    indices = set(input_string.replace(',', ''))

    # Make sure number views is equivalent to the number of terms
    if len(input_list) != len(views):
        raise ValueError("Number of einsum terms must be equal to the number of views.")

    # Get length of each unique index and ensure all dimension are correct
    inds_left = indices.copy()
    dimension_dict = {}
    for tnum, term in enumerate(input_list):
        sh = views[tnum].shape
        if len(sh) != len(term):
            raise ValueError("Dimensions of array and term does not match for term %d.", tnum)
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            if char in dimension_dict.keys():
                if dimension_dict[char] != dim:
                    raise ValueError("Size of label '%s' does not match other terms.", char)
            else:
                dimension_dict[char] = dim

    # Compute size of each input array plus the output array
    size_list = []
    for term in input_list + [output_string]:
        size_list.append(_compute_size(term, dimension_dict))
    out_size = max(size_list)

    # Grab a few kwargs
    debug_arg = kwargs.get("debug", False)
    tdot_arg = kwargs.get("tensordot", True)
    path_arg = kwargs.get("path", "opportunistic")
    memory_arg = kwargs.get("memory", out_size)
    return_path_arg = kwargs.get("return_path", False)

    # Maximum memory is an important variable, should be at most this value
    if memory_arg > out_size:
        memory_arg = out_size

    # If total flops is very small just avoid the overhead altogether
    total_flops = _compute_size(indices, dimension_dict)
    if (total_flops < 1e5) and not return_path_arg:
        return np.einsum(string, *views)

    # If no rank reduction leave it to einsum
    if (indices == output_set):
        if return_path_arg:
            return [tuple(range(len(input_list)))]
        else:
            return np.einsum(string, *views)

    if debug_arg > 0:
        print('Complete contraction:  %s' % (input_string + '->' + output_string))
        print('       Naive scaling:%4d' % len(indices))

    # Compute path
    if not isinstance(path_arg, str):
        path = path_arg
    elif len(input_list) == 2:
        path = [(0, 1)]
    elif path_arg == "opportunistic":
        path = _path_opportunistic(input_list, output_set, dimension_dict, memory_arg)
    elif path_arg == "optimal":
        path = _path_optimal(input_list, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError("Path name %s not found", path_arg)

    # Return path if requested
    if return_path_arg:
        return path

    if debug_arg > 0:
        print('-' * 80)
        print('%6s %6s %24s %40s' % ('scaling', 'GEMM', 'current', 'remaining'))
        print('-' * 80)

    ### Start contraction loop
    views = list(views)
    for contract_inds in path:
        # Make sure we remove inds from right to left
        contract_inds = sorted(list(contract_inds), reverse=True)

        contract = _find_contraction(contract_inds, input_set, output_set)
        out_inds, input_set, index_removed, index_contract = contract

        # Build required structures and explicitly delete views
        no_duplicates = True
        tmp_views, tmp_input = [], []
        for x in contract_inds:
            tmp_views.append(views.pop(x))

            new_string = input_list.pop(x)
            no_duplicates &= (len(set(new_string)) == len(new_string))
            tmp_input.append(new_string)

        ### Consider doing tensordot
        can_dot = tdot_arg & no_duplicates
        can_dot &= (len(tmp_views) == 2) & (len(index_removed) > 0)
        can_dot &= len(set(tmp_input[0]) & set(tmp_input[1])) > 0
        can_dot &= (len(tmp_input[0]) != 0) & (len(tmp_input[1]) != 0)

        # Get index result
        index_result = tmp_input[0] + tmp_input[1]
        for s in index_removed:
            index_result = index_result.replace(s, '')

        can_dot &= (set(tmp_input[0]) & set(tmp_input[1])) == set(index_result)
        can_dot &= (len(set(index_result)) == len(index_result))
        ### End considering tensortdot

        ### If cannot do tensordot, do einsum
        if can_dot is False:
            # We can choose order of output indices, shortest first
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            sort_result.sort()
            index_result = ''.join([x[1] for x in sort_result])

        # Print current contraction
        einsum_string = ','.join(tmp_input) + '->' + index_result
        if debug_arg > 0:
            remaining = ','.join(input_list) + ',' + index_result + '->' + output_string
            print('%4d    %6s %24s %40s' % (len(index_contract), can_dot, einsum_string, remaining))

        # Tensordot
        if can_dot:
            ftpos, stpos = [], []
            for s in index_removed:
                ftpos.append(tmp_input[0].find(s))
                stpos.append(tmp_input[1].find(s))
            # Tensordot does not sort the indices intelligently, we can help it out
            if tmp_views[0].shape[min(ftpos)] > tmp_views[1].shape[min(stpos)]:
                ftpos, stpos = zip(*sorted(zip(ftpos, stpos)))
            else:
                stpos, ftpos = zip(*sorted(zip(stpos, ftpos)))
            new_view = np.tensordot(tmp_views[0], tmp_views[1], axes=(ftpos, stpos))

        # Conventional einsum
        else:
            new_view = np.einsum(einsum_string, *tmp_views)

        # Append new items
        views += [new_view]
        input_list += [index_result]
        del tmp_views, new_view  # Dereference what we can
    ### Finish contraction loop

    # We may need to do a final transpose
    if input_list[0] == output_string:
        return views[0]
    else:
        return np.einsum(input_list[0] + '->' + output_string, views[0])


