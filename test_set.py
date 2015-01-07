import numpy as np
import pandas as pd
import sys, traceback
import timeit
import test_helper as th
from opt_einsum import opt_einsum
pd.set_option('display.width', 200)

# Kill if we use more than 10GB of memory
import resource
rsrc = resource.RLIMIT_DATA
limit = int(1e10)
resource.setrlimit(rsrc, (limit, limit))

# Attempts to linearly scale the time, not the dimension size
scale_list = [1]

# Filter based on key
#key_filter = 'Hada'
key_filter = ''

# Choose a path
#opt_path = 'optimal'
opt_path = 'opportunistic'

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
            opt = opt_einsum(sum_string, *views, path=opt_path)
        except Exception as error:
            out.append([key, 'Opt_einsum failed', sum_string, scale, 0, 0])
            continue

        if np.allclose(ein, opt) is False:
            out.append([key, 'Comparison failed', sum_string, scale, 0, 0])
            continue

        setup = "import numpy as np; from opt_einsum import opt_einsum; \
                 from __main__ import sum_string, views, opt_path"
        einsum_string = "np.einsum(sum_string, *views)"
        opt_einsum_string = "opt_einsum(sum_string, *views, path=opt_path)"


        # How many times to test each expression, einsum will be significantly slower
        e_n = 1
        o_n = 3
        einsum_time = timeit.timeit(einsum_string, setup=setup, number=e_n) / e_n
        opt_einsum_time = timeit.timeit(opt_einsum_string, setup=setup, number=o_n) / o_n

        out.append([key, 'True', sum_string, scale, einsum_time, opt_einsum_time])

df = pd.DataFrame(out)
df.columns = ['Key', 'Flag', 'String', 'Scale', 'Einsum time', 'Opt_einsum time']
df['Ratio'] = np.around(df['Einsum time']/df['Opt_einsum time'], 2)

df = df.set_index(['Key', 'String', 'Scale'])
df = df.sort_index()
print df

print '\nDescription of speedup:'
print df['Ratio'].describe()
print '\nNumber of opt_einsum slower than optimal: %d.' % np.sum(df['Ratio']<0.90)

