import numpy as np
import pandas as pd
import timeit

import resource
rsrc = resource.RLIMIT_DATA
limit = int(1e9)
resource.setrlimit(rsrc, (limit, limit))

import opt_einsum as oe
pd.set_option('display.width', 200)


opt_path = 'optimal'

# Number of dimensions
max_dims = 4
min_dims = 2

# Size of each dimension
min_size = 10
max_size = 20

# Number of terms
min_terms = 3
max_terms = 5

# Additional parameters
max_indices = 6
max_doubles = 1E7

alpha = list('abcdefghijklmnopqrstuvwyxz')
alpha_dict = {num:x for num, x in enumerate(alpha)}

print('Maximum term size is %d' % (max_size**max_dims))

def make_term():
    num_dims = np.random.randint(min_dims, max_dims+1)
    term = np.random.randint(0, max_indices, num_dims)
    return term

def get_string(term):
    return ''.join([alpha_dict[x] for x in term])

def random_contraction():

    # Compute number of terms
    num_terms = np.random.randint(min_terms, max_terms)

    # Compute size of each index
    index_size = np.random.randint(min_size, max_size, max_indices)

    # Build random terms and views
    int_terms = [make_term() for x in range(num_terms)]
    views = [np.random.rand(*index_size[s]) for s in int_terms]

    # Compute einsum string and return string
    sum_string = ','.join([get_string(s) for s in int_terms])
    out_string = sum_string.replace(',','')
    out_string = [x for x in alpha if out_string.count(x)==1]

    #sum_string += '->'
    sum_string += '->' + ''.join(out_string)
    return (sum_string, views, index_size)

out = []
for x in range(200):
    sum_string, views, index_size = random_contraction()

    try:
        ein = np.einsum(sum_string, *views)
    except Exception as error:
        out.append(['Einsum failed', sum_string, index_size, 0, 0])
        continue

    try:
        opt = oe.contract(sum_string, *views, path=opt_path)
    except Exception as error:
        out.append(['Opt_einsum failed', sum_string, index_size, 0, 0])
        continue

    current_opt_path = oe.contract_path(sum_string, *views, optimize=opt_path)[0]
    if not np.allclose(ein, opt):
        out.append(['Comparison failed', sum_string, index_size, 0, 0])
        continue

    setup = "import numpy as np; import opt_einsum as oe; \
             from __main__ import sum_string, views, current_opt_path"
    einsum_string = "np.einsum(sum_string, *views)"
    contract_string = "oe.contract(sum_string, *views, path=current_opt_path)"

    e_n = 1
    o_n = 1
    einsum_time = timeit.timeit(einsum_string, setup=setup, number=e_n) / e_n
    contract_time = timeit.timeit(contract_string, setup=setup, number=o_n) / o_n

    out.append([True, sum_string, current_opt_path, einsum_time, contract_time])

df = pd.DataFrame(out)
df.columns = ['Flag', 'String', 'Path', 'Einsum time', 'Opt_einsum time']
df['Ratio'] = df['Einsum time']/df['Opt_einsum time']

diff_flags = df['Flag']!=True
print('\nNumber of contract different than einsum: %d.' % np.sum(diff_flags))
if sum(diff_flags)>0:
    print('Terms different than einsum')
    print(df[df['Flag']!=True])

print('\nDescription of speedup in relative terms:')
print(df['Ratio'].describe())

print('\nNumber of contract slower than einsum:   %d.' % np.sum(df['Ratio']<0.90))
tmp = df.loc[df['Ratio']<0.90].copy()
tmp['Diff (us)'] = np.abs(tmp['Einsum time'] - tmp['Opt_einsum time'])*1e6
tmp = tmp.sort_values('Diff (us)', ascending=False)
print(tmp)

#diff_us = np.abs(tmp['Einsum time'] - tmp['Opt_einsum time'])*1e6
print('\nDescription of slowdown:')
print(tmp.describe())



