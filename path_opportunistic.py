import itertools as it
import numpy as np

def extract_elements(inds, lst):
    result, extract = [], []
    for num, x in enumerate(lst):
        if num in inds:
            extract.append(x)
        else:
            result.append(x)
    return (result, extract)
            

def find_best_pair(inp, out, ind_dict):
    """
    Returns (do, current_string, contract)
    """

    # Set the sets 
    ts_sets = map(set, inp)
    out_set = set(out)

    # Find maximum overlap
    # Reminder: everything is set arithmetic
    results = []
    for fpos in range(len(inp)):
        first = ts_sets[fpos]
        for spos in range(fpos+1, len(inp)):
            second = ts_sets[spos]

            # Indices of the contraction
            index_contract = first | second

            # See if we can do any other contractions for free 
            positions = [fpos, spos]
            for num, tmp_set in enumerate(ts_sets):
                if num in positions:
                    continue
                if index_contract >= tmp_set:
                    positions.append(num)

            # Indices to contract over
            index_inter = first & second

            # Indices that cannot be removed
            full = inp[:]
            for pos in positions:
                full.remove(inp[pos])

            # Build index sets
            index_remain = set(''.join(full)) | out_set
            index_result = index_contract - (index_inter - index_remain)
            index_removed = index_inter - index_result

            # Can we do tensordot?
            # if len(index_removed) > 0:
            #     tdot = True
            #     # Can only do two terms
            #     positions = [fpos, spos]
            #     # Will get same ordering back out
            #     index_result = inp[fpos] + inp[spos]
            #     for s in index_removed:
            #         index_result = index_result.replace(s, '')
            # else:
            #     index_result = ','.join(index_result)
            #     tdot = False 
            
            # Build contraction syntax
            contract = (tuple(positions), index_result)
            #contract = (tuple(positions), tdot, index_result)

            ### Build sort timings
        
            # Build sort tuple
            rank_reduction = len(index_result) - max(len(first), len(second)) - len(index_removed)
            reduction_size = np.prod([ind_dict[x] for x in index_removed])
            sum_size = np.prod([ind_dict[x] for x in index_contract])

            # Sort tuple
            sort = (rank_reduction, -reduction_size, -sum_size)
            #sort = (rank_reduction, -tdot, -reduction_size, -sum_size)

            # Best contraction, indices, result index
            results.append([sort, contract])

    # Sort based on indices of tuple
    results.sort()
    do = results[0][1]
    # print do
    # exit()
    cont_out = ''.join(do[1])
    string = ','.join(inp[x] for x in do[0])
    string += '->' + cont_out

    new_inp = [inp[x] for x in range(len(inp)) if x not in do[0]]
    new_inp += [cont_out]

    contraction = (string, do[0])

    return new_inp, contraction

def path_opportunistic(inp, out, ind_dict):
    path = [[(), inp + '->' + out]]
    inp = inp.split(',')

    while True:
        # Find best to contract
        inp, contraction = find_best_pair(inp, out, ind_dict)
        remaining = ','.join(inp) + '->' + out
        path.append([contraction,  remaining])
        if len(inp)<=2:
            break

    path.append([(remaining, tuple(range(len(inp)))), ''])    
    return path





