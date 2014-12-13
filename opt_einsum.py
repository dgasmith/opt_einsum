import time
import numpy as np
import itertools as it

from path_opportunistic import path_opportunistic

def extract_elements(inds, lst):
    result, extract = [], []
    for num, x in enumerate(lst):
        if num in inds:
            extract.append(x)
        else:
            result.append(x)
    return (result, extract)


# cost_dict = {}
# def compute_cost(inp, out, ind_sizes):
# 
#     key = ','.join(inp) + '->' + out
#     if key not in cost_dict.keys():
#         # Get scaling
#         overall = set(inp.replace(',',''))
#         cost = np.prod([ind_sizes[x] for x in overall])
# 
#         # Compute prefactor
#         prefactor = 1
#         prefactor += len(overall - set(out))
#         cost_dict[key] = cost*prefactor
# 
#     return cost_dict[key]

def run_path(path, views):

    # Loop over contractions, first is original expression
    for p in path[1:]:
        contract, read = p        
        views, p_views = extract_elements(contract[1], views)
        views += [np.einsum(contract[0], *p_views)]

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


    result = run_path(path, views)
    return result


# Current test case

# N = 10
# C = np.random.rand(N, N)
# I = np.random.rand(N, N, N, N)
# 
# string = 'pi,qj,ijkl,rk,sl->pqrs'
# views = [C, C, I, C, C]
# 
# opt = opt_einsum(string, *views)
# conv = np.einsum(string, *views)
# 
# assert np.allclose(opt, conv)



