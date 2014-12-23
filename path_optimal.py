import numpy as np
import time


def path_optimal(inp, out, ind_dict, memory):
    inp_set = map(set, inp)
    out_set = set(out)
    # Build (total_cost, positions, results, indices_remaining)
    # positions = [(0,1), (0,2),... etc
    # results = ['ijkl', 'ckd',... etc
    # indices_remaining = [set('ijkl'), set('abcd'),... etc

    t = time.time()
    current = [(0, [], [], inp_set)]
    for iteration in range(len(inp)-1):
        new = []
        # Grab all unique pairs
        comb_iter = zip(*np.triu_indices(len(inp)-iteration, 1))
        for curr in current:
            cost, positions, result, remaining = curr
            for con in comb_iter:

                # Find contraction indices
                index_contract = set()
                index_remain = out_set.copy()
                new_inp = []
                for ind, value in enumerate(remaining):
                    if ind in con:
                        index_contract |= value
                    else:
                        new_inp.append(value)
                        index_remain |= value

                new_result = index_remain & index_contract

                # Sieve the results based on memory, prevents unnecessarly large tensors
                out_size = 1
                for ind in new_result:
                    out_size *= ind_dict[ind]

                if out_size > memory:
                    continue

                # Build result
                index_removed = (index_contract - new_result)
                new_inp.append(new_result)

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



