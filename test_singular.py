import numpy as np
import pandas as pd
import sys, traceback
import time
import timeit

from opt_einsum import opt_einsum
pd.set_option('display.width', 1000)

tests = {}
# Randomly produced contractions
tests['Random1'] = ['aab,fa,df,ecc->bde',  [25, 18, 23, 16, 28, 14]]
tests['Random2'] = ['ecb,fef,bad,ed->ac',  [19, 20, 17, 21, 12, 21]]

# Index transformations
tests['Index1'] = ['ea,fb,abcd,gc,hd->efgh', [10, 10, 10, 10, 15, 8, 6, 3]]
tests['Index2'] = ['ea,fb,abcd,gc,hd->efgh', [10, 10, 10, 10, 3, 6, 8, 15]]
tests['Index3'] = ['ea,fb,abcd,gc,hd->efgh', [15, 8, 6, 3, 10, 10, 10, 10]]

# Pure hadamard
tests['Hadamard1'] = ['abc,abc->abc', [200, 200, 200]]
tests['Hadamard2'] = ['abc,abc,abc->abc', [200, 200, 200]]
tests['Hadamard3'] = ['abc,ab,abc->abc', [200, 200, 200]]
tests['Hadamard4'] = ['a,ab,abc->abc', [200, 200, 200]]

# Real world test cases
tests['Actual1'] = ['acjl,pbpk,jkib,ilac,jlac,jklabc,ilac', [10, 5, 9, 10, 5, 25, 6, 14, 11]]
tests['Actual2'] = ['cj,bdik,akdb,ijca,jc,ijkbcd,ijac', [10, 14, 9, 10, 13, 12, 13, 14, 11]]
tests['Actual3'] = ['abik,ikjp,pjba,ikab,jab', [10, 22, 15, 10, 17, 25]]
tests['Actual4'] = ['bdk,cji,ajdb,ikca,kbd,ijkcd,ikac', [10, 11, 9, 10, 12, 15, 13, 14, 11]]
tests['Actual5'] = ['cij,bdk,ajbc,ikad,ijc,ijk,ikad', [10, 17, 9, 10, 13, 16, 15, 14, 11]]
#tests['Actual2'] = [, [10, 5, 9, 10, 5, 25, 6, 14, 11]]

# A few tricky cases
tests['Collapse1'] = ['ab,ab,c->c', [200, 200, 200]]
tests['Collapse2'] = ['ab,ab,cd,cd->', [60, 60, 60, 60]]
tests['Collapse3'] = ['ab,ab,cd,cd,ef,ef->', [15, 15, 15, 15, 15, 15]]


def build_views(string, sizes):
    terms = string.split('->')[0].split(',')

    alpha = ''.join(set(''.join(terms)))
    sizes_dict = {alpha:size for alpha,size in zip(alpha, sizes)}

    views = []
    for term in terms:
        term_index = [sizes_dict[x] for x in term]
        views.append(np.random.rand(*term_index))
    return views

#scale_list = [0.5, 0.7, 0.9, 1.1, 1.3]
scale_list = [1]

alpha = list('abcdefghijklmnopqrstuvwyxz')
alpha_dict = {num:x for num, x in enumerate(alpha)}
out = []


key = 'Index1'
key = 'Actual1'
#key = 'Random1'


sum_string, index_size = tests[key]
views = build_views(sum_string, index_size)
ein_result = np.einsum(sum_string, *views)
# opt_ein_result = opt_einsum(sum_string, *views, debug=1, path='opportunistic')
opt_ein_result = opt_einsum(sum_string, *views, debug=1, path='optimal')


print 'Einsum shape:        %s' % (str(ein_result.shape))
print 'Opt einsum shape:    %s' % (str(opt_ein_result.shape))
print 'Allclose:            %s' % np.allclose(ein_result, opt_ein_result)



