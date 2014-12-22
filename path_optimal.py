import itertools as it
import numpy as np
import time

def find_contract(positions, inp, out, ind_dict):
    """
    Returns (do, current_string, contract)
    """

    index_contract = set()
    index_remain = out.copy()
    new_inp = []
    for ind, value in enumerate(inp):
        if ind in positions:
            index_contract |= value
        else:
            new_inp.append(value)
            index_remain |= value

    # Build index sets
    index_result = index_remain & index_contract 
    index_removed = (index_contract - index_result)

    new_inp.append(index_result)

    cost = 1
    for x in index_contract:
        cost *= ind_dict[x]
    cost *= 1 + len(index_contract - index_removed)

    return (new_inp, index_result, cost)

def compute_size(inds, ind_dict):
    size = 1
    for x in inds:
        size *= ind_dict[x]
    return size
    

def path_optimal(inp, out, ind_dict, memory):
    inp_set = map(set, inp)
    out_set = set(out)
    # Build (total_cost, positions, results, indices_remaining)
    # positions = [(0,1), (0,2),... etc
    # results = ['ijkl', 'ckd',... etc
    # indices_remaining = [set('ijkl'),set('abcd'),... etc

    t = time.time()
    current = [(0, [], [], inp_set)]
    for iteration in range(len(inp)-1):
        new = []
        comb_iter = list(it.combinations(range(len(inp)-iteration), 2))
        for curr in current:
            cost, positions, result, remaining = curr
            for con in comb_iter:
                new_remain, new_result, new_cost = find_contract(con, remaining, out_set, ind_dict)

                # Sieve the results, prevents large expansions
                if compute_size(new_result, ind_dict) > memory:
                    continue

                # Build new contraction
                new_cost += cost
                new_result = result + [new_result]
                new_pos = positions + [con]
                new.append((new_cost, new_pos, new_result, new_remain))

        current = new 

    einsum_string = ','.join(inp) + '->' + ''.join(out)
    new.sort()
    best = new[0]
    path = zip(best[1], best[2])
    print 'Path Optimal time: %6d %2d %3.3f %40s' % (len(new), len(inp), (time.time()-t), einsum_string)
    
    return path



