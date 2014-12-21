import itertools as it
import numpy as np

def compute_size(inds, ind_dict):
    ret = 1
    for i in inds:
        ret *= ind_dict[i]
    return ret


def find_best_pair(inp, out, ind_dict):
    """
    Returns (do, current_string, contract)
    """

    set_inp = map(set, inp)
    out_size = compute_size(out, ind_dict)
    # Fin  maximum overlap
    # Reminder: everything is set arithmetic
    results = []
    for positions in it.combinations(range(len(inp)), 2):
            index_contract = set()
            index_remain = set(out).copy()
            new_inp = []
            for ind, value in enumerate(set_inp):
                if ind in positions:
                    index_contract |= value
                else:
                    new_inp.append(inp[ind])
                    index_remain |= value
        
            # Build index sets
            index_result = index_remain & index_contract
            index_removed = (index_contract - index_result)

            # Check for other contracts that we can do for free
            # if len(index_removed)==0:
            #     for ind, value in enumerate(set_inp):
            #         if (ind not in positions) and (index_contract >= ind):
            #             positions += (ind,)

            new_inp.append(''.join(index_result))
            contract = ((positions, index_result), new_inp)
        
            # Build sort tuple
            result_size = compute_size(index_result, ind_dict)
            total_intermediate_size = sum([compute_size(inp[x], ind_dict) for x in positions])
            removed_size = compute_size(index_removed, ind_dict)

            size_reduction = result_size - total_intermediate_size

            sum_size = compute_size(index_contract, ind_dict)

            # Sort tuple
            sort = size_reduction - removed_size
#            sort = (size_reduction, -removed_size, -sum_size)
#            sort = (-removed_size, sum_size, size_reduction)
#            print inp[positions[0]], inp[positions[1]], sort

            # Best contraction, indices, result index
            results.append([sort, contract])

    # Sort based on first index
    results.sort()
    # Return best contraction
    return results[0][1]

def path_opportunistic(inp, out, ind_dict):

    path = []
    while True:
        pos, inp = find_best_pair(inp, out, ind_dict)
        # print '-'*50
        path.append(pos)
        if len(inp)<=1:
            break

    return path





