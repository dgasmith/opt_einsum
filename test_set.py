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

keys = 'Random2'


for key in tests.keys():
    if key != keys: continue
    sum_string, index_size = tests[key]
    print sum_string
    for scale in scale_list:
        loop_index_size = (np.array(index_size) * scale).astype(np.int)
        loop_index_size[loop_index_size<1] = 1
        views = build_views(sum_string, loop_index_size)

        try:
            ein = np.einsum(sum_string, *views)
        except Exception as error:
            out.append([key, 'Einsum failed', sum_string, scale, 0, 0])
            continue

        try:
            opt = opt_einsum(sum_string, *views)
        except Exception as error:
            out.append([key, 'Opt_einsum failed', sum_string, scale, 0, 0])
            continue

        ident = np.allclose(ein, opt)
        if not ident:
            out.append([key, 'Comparison failed', sum_string, scale, 0, 0])

        setup = "import numpy as np; from opt_einsum import opt_einsum; \
                 from __main__ import sum_string, views"
        einsum_string = "np.einsum(sum_string, *views)"
        opt_einsum_string = "opt_einsum(sum_string, *views)"


        e_n = 1
        o_n = 1
        einsum_time = timeit.timeit(einsum_string, setup=setup, number=e_n) / e_n
        opt_einsum_time = timeit.timeit(opt_einsum_string, setup=setup, number=o_n) / o_n

        out.append([key, ident, sum_string, scale, einsum_time, opt_einsum_time])

df = pd.DataFrame(out)
df.columns = ['Key', 'Flag', 'String', 'Scale', 'Einsum time', 'Opt_einsum time']
df['Ratio'] = df['Einsum time']/df['Opt_einsum time']

df = df.set_index(['Key', 'Scale'])
df = df.sort_index()
print df


print '\nDescription of speedup:'
print df['Ratio'].describe()
print '\nNumber of opt_einsum operations slower than einsum: %d.' % np.sum(df['Ratio']<0.95)

