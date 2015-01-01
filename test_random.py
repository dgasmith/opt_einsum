import numpy as np
import pandas as pd
import timeit

from opt_einsum import opt_einsum

pd.set_option('display.width', 1000)

# Number of dimensions
max_dims = 3
min_dims = 2

# Size of each dimension
min_size = 10
max_size = 30

# Number of terms
min_terms = 3
max_terms = 5

# Additional parameters
max_indices = 6
max_doubles = 1E7

alpha = list('abcdefghijklmnopqrstuvwyxz')
alpha_dict = {num:x for num, x in enumerate(alpha)}

print 'Maximum term size is %d' % (max_size**max_dims)

def view_traceback():
    ex_type, ex, tb = sys.exc_info()
    traceback.print_tb(tb)

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
        opt = opt_einsum(sum_string, *views)
    except Exception as error:
        out.append(['Opt_einsum failed', sum_string, index_size, 0, 0])
        continue

    ident = np.allclose(ein, opt)
    if not ident:
        out.append(['Comparison failed', sum_string, index_size, 0, 0])
        continue

    setup = "import numpy as np; from opt_einsum import opt_einsum; \
             from __main__ import sum_string, views"
    einsum_string = "np.einsum(sum_string, *views)"
    opt_einsum_string = "opt_einsum(sum_string, *views)"

    e_n = 1
    o_n = 5
    einsum_time = timeit.timeit(einsum_string, setup=setup, number=e_n) / e_n
    opt_einsum_time = timeit.timeit(opt_einsum_string, setup=setup, number=o_n) / o_n

    out.append([ident, sum_string, index_size, einsum_time, opt_einsum_time])

df = pd.DataFrame(out)
df.columns = ['Flag', 'String', 'Shapes', 'Einsum time', 'Opt_einsum time']
df['Ratio'] = df['Einsum time']/df['Opt_einsum time']

diff_flags = df['Flag']!=True
print '\nNumber of opt_einsum different than einsum: %d.' % np.sum(diff_flags)
if sum(diff_flags)>0:
    print 'Terms different than einsum'
    print df[df['Flag']!=True]

print '\nDescription of speedup:'
print df['Ratio'].describe()
print 'Number of opt_einsum slower than einsum:   %d.' % np.sum(df['Ratio']<1)

