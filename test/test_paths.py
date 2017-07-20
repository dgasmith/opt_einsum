from __future__ import division, absolute_import, print_function

import numpy as np
import opt_einsum as oe
import pytest


path_tests = {
    'GEMM1' : ([set('abd'), set('ac'), set('bdc')], set(''), {'a': 1, 'b':2, 'c':3, 'd':4}),

    'Inner1' : ([set('abcd'), set('abc'), set('bc')], set(''), {'a': 5, 'b':2, 'c':3, 'd':4}),
    }


def check_path(test_output, benchmark, bypass=False):
    if not isinstance(test_output, list):
        return False

    if len(test_output) != len(benchmark):
        return False

    ret = True
    for pos in range(len(test_output)):
        ret &= isinstance(test_output[pos], tuple)
        ret &= test_output[pos] == benchmark[pos]
    return ret


def assert_contract_order(func, test_data, max_size, benchmark):

    test_output = func(test_data[0], test_data[1], test_data[2], max_size)
    assert check_path(test_output, benchmark)


def test_size_by_dict():
    
    sizes_dict = {}
    for ind, val in zip('abcdez', [2, 5, 9, 11, 13, 0]):
        sizes_dict[ind] = val

    path_func = oe.paths.compute_size_by_dict

    assert np.allclose(1, path_func('', sizes_dict))
    assert np.allclose(2, path_func('a', sizes_dict))
    assert np.allclose(5, path_func('b', sizes_dict))

    assert np.allclose(0, path_func('z', sizes_dict))
    assert np.allclose(0, path_func('az', sizes_dict))
    assert np.allclose(0, path_func('zbc', sizes_dict))

    assert np.allclose(104, path_func('aaae', sizes_dict))
    assert np.allclose(12870, path_func('abcde', sizes_dict))


def test_path_optimal():

    test_func = oe.paths.optimal

    test_data = path_tests['GEMM1']
    assert_contract_order(test_func, test_data, 5000, [(0, 2), (0, 1)])
    assert_contract_order(test_func, test_data, 0, [(0, 1, 2)])

def test_path_greedy():

    test_func = oe.paths.greedy

    test_data = path_tests['GEMM1']
    assert_contract_order(test_func, test_data, 5000, [(0, 2), (0, 1)])
    assert_contract_order(test_func, test_data, 0, [(0, 1, 2)])

def test_memory_paths():
    
    expression = "abc,bdef,fghj,cem,mhk,ljk->adgl"

    views = oe.helpers.build_views(expression) 

    # Test tiny memory limit
    path_ret = oe.contract_path(expression, *views, path="optimal", memory_limit=5) 
    assert check_path(path_ret[0], [(0, 1, 2, 3, 4, 5)])  

    path_ret = oe.contract_path(expression, *views, path="greedy", memory_limit=5) 
    assert check_path(path_ret[0], [(0, 1, 2, 3, 4, 5)])  

    # Check the possibilities, greedy is capped
    path_ret = oe.contract_path(expression, *views, path="optimal", memory_limit=-1) 
    assert check_path(path_ret[0], [(0, 3), (0, 4), (0, 2), (0, 2), (0, 1)])  

    path_ret = oe.contract_path(expression, *views, path="greedy", memory_limit=-1) 
    assert check_path(path_ret[0], [(0, 3), (0, 4), (0, 1, 2, 3)])  
    

    

