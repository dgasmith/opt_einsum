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

    do = True
   
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

            # Indices to contract over
            index_inter = first & second

            # Indices that cannot be removed
            full = inp[:]
            full.remove(inp[fpos])
            full.remove(inp[spos])
            index_remain = set(''.join(full)) | out_set

            # Indices of the result
            index_result = ((first | second) - (index_inter - index_remain)) 
        
            # Indices removed
            index_removed = index_inter - index_result
        
            # Contraction tuple
            contract = ((fpos, spos), index_result)

            # Build sort tuple
            rank_reduction = len(index_result) - max(len(first), len(second)) - len(index_removed)

            # # See if we can do any other contractions for free 
            # for num, tmp_set in enumerate(ts_sets):
            #     if num in positions:
            #         continue
            #     if (tmp_set == first) or (tmp_set == second):
            #         positions.append(num)

            # Sort tuple
            sort = (rank_reduction)

            # Best contraction, indices, result index
            results.append([sort, contract])

    # Sort based on indices of tuple
    results.sort()
    do = results[0][1]

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





