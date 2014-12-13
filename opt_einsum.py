import time
import numpy as np

sum_verbose = False
#sum_verbose = True


def compute_cost(inp):
    # Tries to estimate
    return 0


def extract_elements(inds, lst):
    result, extract = [], []
    for num, x in enumerate(lst):
        if num in inds:
            extract.append(x)
        else:
            result.append(x)
    return (result, extract)
            

def find_best_pairs(string, verbose=False):
    do = True
    
    ts = string.split(',')
    ts_sets = map(set, ts)

    if len(ts)==0:
        return (False, None)

    # Find maximum overlap
    if sum_verbose:
        print string

    results = []
    for fpos in range(len(ts)):
        first = ts_sets[fpos]
        for spos in range(fpos):
            second = ts_sets[spos]
        
            # If identical go ahead and contract
            if first==second:
                return (True, [[fpos, spos], ''.join(first)])

            inter = first.intersection(second)

            # Find the overlap between the full string and the contraction
            full = ts[:]
            full.remove(ts[fpos])
            full.remove(ts[spos])
            full = set(''.join(full))
            overlap = full.intersection(inter)

            # Resulting contract indices
            result = first.union(second).difference(inter)
            result = result.union(overlap)

            # Dont do the contraction if intermediate is too big
            if len(result)>6:
                continue

            positions = [spos, fpos]

            # # See if we can do any other contractions for free 
            for num, tmp_set in enumerate(ts_sets):
                if num in positions:
                    continue
                if (tmp_set == first) or (tmp_set == second):
                    positions.append(num)

            # Sort tuple
            index_reduction = len(result)-len(first.union(second))
            tup = (index_reduction, len(result), -1*len(inter))

            # Best contraction, indices, result index
            results.append([tup, positions, ''.join(result)])

    # Sort based on indices of tuple
    results.sort()

    #results.sort(reverse=True)
    if len(results)==0:
        return (False, None)

    if sum_verbose:
        for x in results:
            print x        

    
    best = results[0][1:]
    best_tup = results[0][0]


    return do, best

def intermediate_contract(contract, string, views):
    t= time.time()
    ts = string.split(',')

    c_inds = contract[0]
    c_result = contract[1]

    # New contraction
    ts, c_ts = extract_elements(c_inds, ts)
    views, c_views = extract_elements(c_inds, views)

    tmp_str = ','.join(c_ts) + '->' + c_result
    tmp = np.einsum(tmp_str, *c_views)

    # Add in newly contracted
    ts += [c_result]
    views += [tmp]

    string = ','.join(ts)

    td = (time.time() - t)
    cost = compute_cost('.'.join(c_ts)) 
    return string, views


# Rewrite einsum to handle different cases
def opt_einsum(string, *views, **kwargs):

    # Already optimized for two 
    if len(string.split(','))<=2:
        return np.einsum(string, *views)
        
    else:
        string = string.replace('->', '')
        while True:

            # Find best to contract
            do, best = find_best_pairs(string)
            if do is False:
                break

            # Contract
            string, views = intermediate_contract(best, string, views)
            if len(string.split(','))<=2:
                break


        t = time.time()
        if len(string.split(','))==1:
            result = np.sum(views[0]) 
        else:
            result = np.einsum(string, *views)

        td = (time.time() - t)
        cost = compute_cost(string) 
        if sum_verbose:
            print "%-30s    Time: %.2f    Cost: %.2f    Ratio: %.2f" % (string, td, cost, cost/td)
            print ''
        return result

