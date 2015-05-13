import numpy as np
import pandas as pd
import sys, traceback
import timeit
import test_helper as th
from opt_einsum import contract
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 200)

# Kill if we use more than 10GB of memory
import resource
rsrc = resource.RLIMIT_DATA
limit = int(1e10)
resource.setrlimit(rsrc, (limit, limit))

# Attempts to linearly scale the time, not the dimension size
scale_list = [0.1]

# Filter based on key
#key_filter = 'Dot'
key_filter = ''

# Choose a path
opt_path = 'optimal'
#opt_path = 'opportunistic'

out = []
for key in th.tests.keys():
    if key_filter not in key: continue
    sum_string, index_size = th.tests[key]
    for scale in scale_list:
        views = th.build_views(sum_string, index_size, scale=scale)

        try:
            ein = np.einsum(sum_string, *views)
        except Exception as error:
            out.append([key, 'Einsum failed', sum_string, scale, 0, 0])
            continue

        try:
            opt = contract(sum_string, *views, path=opt_path)
        except Exception as error:
            out.append([key, 'Opt_einsum failed', sum_string, scale, 0, 0])
            continue

        if ~np.allclose(ein, opt):
            out.append([key, 'Comparison failed', sum_string, scale, 0, 0])
            continue

        setup = "import numpy as np; from opt_einsum import contract; \
                 from __main__ import sum_string, views, opt_path"
        einsum_string = "np.einsum(sum_string, *views)"
        contract_string = "contract(sum_string, *views, path=opt_path)"


        # How many times to test each expression, einsum will be significantly slower
        e_n = 1
        o_n = 1
        einsum_time = timeit.timeit(einsum_string, setup=setup, number=e_n) / e_n
        contract_time = timeit.timeit(contract_string, setup=setup, number=o_n) / o_n

        out.append([key, 'True', sum_string, scale, einsum_time, contract_time])

df = pd.DataFrame(out)
df.columns = ['Key', 'Flag', 'String', 'Scale', 'Einsum time', 'Opt_einsum time']
df['Ratio'] = np.around(df['Einsum time']/df['Opt_einsum time'], 2)

df = df.set_index(['Key', 'String', 'Scale'])
df = df.sort_index()

print df

num_failed = (df['Flag']!='True').sum()
if num_failed>0:
    print 'WARNING! %d contract operations failed.' % num_failed 

print '\nDescription of speedup:'
print df['Ratio'].describe()
print '\nNumber of contract slower than optimal: %d.' % np.sum(df['Ratio']<0.90)

