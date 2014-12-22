import time
import numpy as np
import itertools as it

from path_opportunistic import path_opportunistic
from path_optimal import path_optimal

# Rewrite einsum to handle different cases
def opt_einsum(string, *views, **kwargs):
    """
    Attempts to contract tensors in an optimal order using both
    np.einsum and np.tensordot. Primarily aims at reducing the
    overall rank of the contration by building intermediates."

    Parameters
    __________
    string : str
        Einsum string of contractions
    *view  : list of views utilized
    debug  : int (default: 0)
        Level of printing.

    Returns
    -------
    output : ndarray
        The result based on the Einstein summation convention.
    """

    # Split into output and input string
    if '->' in string:
        input_string, output_string = string.split('->')
    else:
        input_string = string
        output_string = ''

    # Build a few useful list and sets
    input_list = input_string.split(',')
    input_set = map(set, input_list)
    output_set = set(output_string)
    indices = set(input_string.replace(',',''))

    # If no rank reduction leave it to einsum
    if indices == output_set:
        return np.einsum(string, *views)

    # Make sure number views is equivalent to the number of terms
    if len(input_list) != len(views):
        raise ValueError("Number of einsum terms must equal to the number of views.")

    # Get length of each unique index    
    inds_left = indices.copy()
    dimension_dict = {}
    for tnum, term in enumerate(input_list):
        sh = views[tnum].shape
        if len(sh) != len(term):
            raise ValueError("Dimensions of array and term does not match for term %d.", tnum)
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            if char in dimension_dict.keys():
                if dimension_dict[char]!=dim:
                    raise ValueError("Size of label '%s' does not match other terms.", char)
            else:
                dimension_dict[char] = dim

    # Compute size of each input array plus the output array
    size_list = []
    for term in input_list + [output_string]:
        size = 1
        for s in term:
            size *= dimension_dict[s]
        size_list.append(size)

    out_size = max(size_list)

    # Grab a few kwargs
    debug_arg = kwargs.get("debug", False)
    tdot_arg = kwargs.get("tensordot", True)
    path_arg = kwargs.get("path", "opportunistic") 
    memory_arg = kwargs.get("memory", out_size)

    if debug_arg>0:
        print('Complete contraction:  %s' % (input_string + '->' + output_string)) 
        print('       Naive scaling:%4d' % len(indices))

    # Compute best path        
    if path_arg == "opportunistic":
        path = path_opportunistic(input_list, output_set, dimension_dict)
    elif path_arg == "optimal":
        path = path_optimal(input_list, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError("Path name %s not found", path_arg)

    if debug_arg>0:
        print('-' * 80)
        print('%6s %6s %25s %40s' % ('scaling', 'GEMM', 'current', 'remaining'))
        print('-' * 80)

     
    ### Start contraction loop
    views = list(views)
    for contract_inds, out_inds in path:
        # Make sure we remove inds from right to left
        contract_inds = sorted(list(contract_inds), reverse=True) 

        # Build required structures
        no_duplicates = True
        tmp_indices = set()
        tmp_views, tmp_input = [], []
        for x in contract_inds:
            new_inp = views[x]
            new_string = input_list[x]

            tmp_views.append(new_inp)
            del views[x]
            tmp_input.append(new_string)
            del input_list[x]

            tmp_indices |= set(new_string)
            no_duplicates &= (len(set(new_string)) == len(new_string)) 

        index_removed = tmp_indices - out_inds            

        ### Consider doing tensordot
        can_dot = tdot_arg & no_duplicates
        can_dot &= (len(tmp_views)==2) & (len(index_removed)>0)

        # Get index result
        index_result = tmp_input[0] + tmp_input[1]
        for s in index_removed:
            index_result = index_result.replace(s, '')
    
        #can_dot &= (len(set(index_result))==len(index_result))
        ### End considering tensortdot

        ### If cannot do tensordot, do einsum
        if can_dot is False:
            # We can choose order of output indices, shortest first
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            sort_result.sort()
            index_result = ''.join([x[1] for x in sort_result])


        # Print current contraction        
        einsum_string = ','.join(tmp_input) + '->' + index_result
        if debug_arg>0:
            remaining = ','.join(input_list) + ',' + index_result + '->' + output_string
            print('%4d    %6s %25s %40s' % (len(tmp_indices), can_dot, einsum_string, remaining)) 

        # Tensordot
        if can_dot:
            ftpos, stpos = [], []
            fs, ss = tmp_input[0], tmp_input[1]

            # Get index result
            for s in index_removed:
                ftpos.append(fs.find(s))
                stpos.append(ss.find(s))            
            new_view = np.tensordot(tmp_views[0], tmp_views[1], axes=(ftpos, stpos))

        # Conventional einsum
        else:
            new_view = np.einsum(einsum_string, *tmp_views)

        # Append new items        
        views += [new_view]
        input_list += [index_result]
    ### Finish contraction loop

    # We may need to do a final transpose
    if input_list[0] == output_string:
        return views[0]
    else:
        return np.einsum(input_list[0] + '->' + output_string, views[0])    
            




