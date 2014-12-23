import itertools as it
import numpy as np

def compute_size(inds, ind_dict):
    ret = 1
    for i in inds:
        ret *= ind_dict[i]
    return ret


def path_opportunistic(inp, out, ind_dict, memory):
    inp_set = map(set, inp)
    out_set = set(out)

    path = []
    for iteration in range(len(inp)-1):
        if len(inp_set)<=1:
            break
        iteration_results = []
        comb_iter = zip(*np.triu_indices(len(inp_set), 1))
        for positions in comb_iter:
            index_contract = set()
            index_remain = out_set.copy()
            new_inp = []
            for ind, value in enumerate(inp_set):
                if ind in positions:
                    index_contract |= value
                else:
                    new_inp.append(value)
                    index_remain |= value

            # Build index sets
            index_result = index_remain & index_contract
            index_removed = (index_contract - index_result)

            # Sieve the results based on memory, prevents unnecessarly large tensors
            out_size = 1
            for ind in index_result:
                out_size *= ind_dict[ind]

            if out_size > memory:
                continue

            new_inp.append(set(''.join(index_result)))

            # Build sort tuple
            sum_size = compute_size(index_contract, ind_dict)

            # Sort tuple
            removed_size = compute_size(index_removed, ind_dict)
            sort = (-removed_size, sum_size)
            contract = (positions, index_result)

            iteration_results.append([sort, contract, new_inp])

        # Sort based on first index
        iteration_results.sort()
        best = iteration_results[0]
        path.append(best[1])

        inp_set = best[2]

    return path





