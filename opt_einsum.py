import time
import numpy as np


def _compute_size(inds, ind_dict):
    ret = 1
    for i in inds:
        ret *= ind_dict[i]
    return ret


def _find_contraction(positions, input_sets, output_set):
    # Find contraction indices
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
    inp_set = map(set, inp)
    out_set = set(out)
    # Build (total_cost, positions, results, indices_remaining)
    # positions = [(0,1), (0,2),... etc
    # results = ['ijkl', 'ckd',... etc
    # indices_remaining = [set('ijkl'), set('abcd'),... etc

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
                out_size = _compute_size(new_result, ind_dict)

                # if out_size > memory:
                #     continue

                # Find cost
                new_cost = 1
                for x in index_contract:
                    new_cost *= ind_dict[x]
                new_cost *= 1 + len(index_contract - index_removed)

                # Build new contraction
                new_cost += cost
                new_result = result + [new_result]
                new_pos = positions + [con]
                new.append((new_cost, new_pos, new_result, new_inp))

        current = new

    new.sort()
    best = new[0]
    path = zip(best[1], best[2])
    einsum_string = ','.join(inp) + '->' + ''.join(out)
    print 'Path Optimal time: %6d %2d %3.5f %40s' % (len(new), len(inp), (time.time()-t), einsum_string)

    return path


def _path_opportunistic(inp, out, ind_dict, memory):
    inp_set = map(set, inp)
    out_set = set(out)

    path = []
    for iteration in range(len(inp) - 1):
        if len(inp_set) <= 1:
            break
        iteration_results = []
        comb_iter = zip(*np.triu_indices(len(inp_set), 1))
        for positions in comb_iter:

            contract = _find_contraction(positions, inp_set, out_set)
            index_result, new_inp, index_removed, index_contract = contract

            # Sieve the results based on memory, prevents unnecessarly large tensors
            out_size = 1
            for ind in index_result:
                out_size *= ind_dict[ind]

            if out_size > memory:
                continue

            # Build sort tuple
            sum_size = _compute_size(index_contract, ind_dict)

            # Sort tuple
            removed_size = _compute_size(index_removed, ind_dict)
            sort = (-removed_size, sum_size)
            contract = (positions, index_result)

            iteration_results.append([sort, contract, new_inp])

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
    *view  : list of views utilized
    debug  : int (default: 0)
        Level of printing.

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
        output_string = ''

    # Build a few useful list and sets
    input_list = input_string.split(',')
    input_set = map(set, input_list)
    output_set = set(output_string)
    indices = set(input_string.replace(',', ''))

    # If no rank reduction leave it to einsum
    if indices == output_set:
        return np.einsum(string, *views)

    # Make sure number views is equivalent to the number of terms
    if len(input_list) != len(views):
        raise ValueError("Number of einsum terms must equal to the number of views.")

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
        size = 1
        for s in term:
            size *= dimension_dict[s]
        size_list.append(size)

    out_size = max(size_list)

    # Grab a few kwargs
    debug_arg = kwargs.get("debug", False)
    tdot_arg = kwargs.get("tensordot", True)
    path_arg = kwargs.get("path", "opportunistic")
    memory_arg = kwargs.get("memory", out_size)

    if debug_arg > 0:
        print('Complete contraction:  %s' % (input_string + '->' + output_string))
        print('       Naive scaling:%4d' % len(indices))

    # Compute path
    if not isinstance(path_arg, str):
        path = path_arg
    elif path_arg == "opportunistic":
        path = _path_opportunistic(input_list, output_set, dimension_dict, memory_arg)
    elif path_arg == "optimal":
        path = _path_optimal(input_list, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError("Path name %s not found", path_arg)

    if debug_arg > 0:
        print('-' * 80)
        print('%6s %6s %25s %40s' % ('scaling', 'GEMM', 'current', 'remaining'))
        print('-' * 80)

    ### Start contraction loop
    views = list(views)
    for contract_inds, out_inds in path:
        # Make sure we remove inds from right to left
        contract_inds = sorted(list(contract_inds), reverse=True)

        # Build required structures and explicitly delete views
        no_duplicates = True
        tmp_indices = set()
        tmp_views, tmp_input = [], []
        for x in contract_inds:
            new_inp = views[x]
            new_string = input_list[x]

            tmp_views.append(new_inp)
            del views[x]
            tmp_input.append(new_string)
            del input_list[x]

            tmp_indices |= set(new_string)
            no_duplicates &= (len(set(new_string)) == len(new_string))

        index_removed = tmp_indices - out_inds

        ### Consider doing tensordot
        can_dot = tdot_arg & no_duplicates
        can_dot &= (len(tmp_views) == 2) & (len(index_removed) > 0)

        # Get index result
        index_result = tmp_input[0] + tmp_input[1]
        for s in index_removed:
            index_result = index_result.replace(s, '')

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
            print('%4d    %6s %25s %40s' % (len(tmp_indices), can_dot, einsum_string, remaining))

        # Tensordot
        if can_dot:
            ftpos, stpos = [], []
            fs, ss = tmp_input[0], tmp_input[1]

            # Get index result
            for s in index_removed:
                ftpos.append(fs.find(s))
                stpos.append(ss.find(s))
            new_view = np.tensordot(tmp_views[0], tmp_views[1], axes=(ftpos, stpos))

        # Conventional einsum
        else:
            new_view = np.einsum(einsum_string, *tmp_views)

        # Append new items
        views += [new_view]
        input_list += [index_result]
    ### Finish contraction loop

    # We may need to do a final transpose
    if input_list[0] == output_string:
        return views[0]
    else:
        return np.einsum(input_list[0] + '->' + output_string, views[0])


