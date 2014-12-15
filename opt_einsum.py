import time
import numpy as np
import itertools as it

from path_opportunistic import path_opportunistic

def extract_elements(inds, lst):
    result, extract = [], []
    
    for x in inds:
        extract.append(lst[x])
    for num, x in enumerate(lst):
        if num not in inds:
            result.append(x)

    return (result, extract)


cost_dict = {}
def compute_cost(inp, out, ind_sizes):

    key = ','.join(inp) + '->' + out
    if key not in cost_dict.keys():
        # Get scaling
        overall = set(inp.replace(',',''))
        cost = np.prod([ind_sizes[x] for x in overall])

        # Compute prefactor
        prefactor = 1
        prefactor += len(overall - set(out))
        cost_dict[key] = cost*prefactor

    return cost_dict[key]

def run_path(path, views):

    # Loop over contractions, first is original expression
    for p in path[1:]:
        contract, read = p        
        views, p_views = extract_elements(contract[2], views)
        if contract[0] == 'tensordot':
            views += [np.tensordot(p_views[0], p_views[1], axes=contract[3])]
        else:
            views += [np.einsum(contract[1], *p_views)]

    return views[0]


# Rewrite einsum to handle different cases
def opt_einsum(string, *views, **kwargs):


    # Split into output and input string
    if '->' in string:
        inp, out = string.split('->')
    else:
        inp = string
        out = ''

    # Already optimized for two terms
    if len(inp.split(','))<=2:
        return np.einsum(string, *views)

    # Get length of each unique index    
    indices = set(inp.replace(',',''))
    inds_left = indices.copy()
    dim_dict = {}

    for num, x in enumerate(inp.split(',')):
        tmp = inds_left.intersection(set(x))
        for s in tmp:
            dim_dict[s] = views[num].shape[x.find(s)]

        inds_left -= tmp
        
    path = path_opportunistic(inp, out, dim_dict)
#    for x in path: print x

    result = run_path(path, views)
    return result



