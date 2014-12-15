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

            positions = [fpos, spos]

            # Indices to contract over
            index_inter = first & second

            # Indices that cannot be removed
            full = inp[:]
            for pos in positions:
                full.remove(inp[pos])

            # Build index sets
            index_remain = set(''.join(full)) | out_set
            index_result = index_contract - (index_inter - index_remain)
            index_removed = (index_inter - index_result)

            # Build einsum string

            # Can we do tensordot?
            rank_reduction = len(index_removed) > 0
            no_dups = (len(first) >= len(inp[fpos])) and (len(second) >= len(inp[spos]))

#            if False:
            if (rank_reduction and no_dups):
                fs, ss = inp[fpos], inp[spos]
                ftpos, stpos = [], []

                # Get index result
                index_result = fs + ss
                for s in index_removed:
                    index_result = index_result.replace(s, '')
                    ftpos.append(fs.find(s))
                    stpos.append(ss.find(s))

                ein_string = ','.join(inp[x] for x in positions) + '->' + index_result
                contract = ('tensordot', ein_string, tuple(positions), (ftpos, stpos))

            else:
                # # See if we can do any other contractions for free 
                # for num, tmp_set in enumerate(ts_sets):
                #     if num in positions:
                #         continue
                #     if index_contract >= tmp_set:
                #         positions.append(num)
                
                index_result = ''.join(index_result)
                ein_string = ','.join(inp[x] for x in positions) + '->' + index_result
                contract = ('einsum', ein_string, tuple(positions))
            
            ### Build sort timings
        
            # Build sort tuple
            rank_reduction = len(index_result) - max(len(first), len(second)) - len(index_removed)
            reduction_size = np.prod([ind_dict[x] for x in index_removed])
            sum_size = np.prod([ind_dict[x] for x in index_contract])

            # Sort tuple
            sort = (rank_reduction, -reduction_size, -sum_size)

            # Best contraction, indices, result index
            results.append([sort, contract])

    # Sort based on indices of tuple
    results.sort()
    best = results[0][1]

    new_inp = [inp[x] for x in range(len(inp)) if x not in best[2]]
    new_inp += [best[1].split('->')[-1]]

    return new_inp, best

def path_opportunistic(inp, out, ind_dict):
    path = [[(), inp + '->' + out]]
    inp = inp.split(',')

    while True:
        # Find best to contract
        inp, contraction = find_best_pair(inp, out, ind_dict)
        remaining = ','.join(inp) + '->' + out
        path.append([contraction,  remaining])
        if len(inp)==1:
            break
    
    path.append([('einsum', remaining, tuple(range(len(inp)))), ''])    

    return path





