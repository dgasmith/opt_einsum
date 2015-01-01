import numpy as np
import pandas as pd
import sys, traceback
import timeit
import test_helper as th
from opt_einsum import opt_einsum
pd.set_option('display.width', 1000)

import resource
rsrc = resource.RLIMIT_DATA
limit = int(1e10)
resource.setrlimit(rsrc, (limit, limit))


#scale_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
#scale_list = [1, 2, 3, 4, 5]
scale_list = [1]


key_filter = ''
#key_filter = 'EP'


#opt_path = 'optimal'
opt_path = 'opportunistic'

out = []
for key in th.tests.keys():
    if key_filter not in key: continue
    sum_string, index_size = th.tests[key]
    for scale in scale_list:
        loop_index_size = (np.array(index_size) * scale).astype(np.int)
        loop_index_size[loop_index_size<1] = 1
        views = th.build_views(sum_string, loop_index_size)

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

        ident = np.allclose(ein, opt)
        if not ident:
            out.append([key, 'Comparison failed', sum_string, scale, 0, 0])
            continue

        setup = "import numpy as np; from opt_einsum import opt_einsum; \
                 from __main__ import sum_string, views, opt_path"
        einsum_string = "np.einsum(sum_string, *views)"
        opt_einsum_string = "opt_einsum(sum_string, *views, path=opt_path)"


        e_n = 1
        o_n = 2
        einsum_time = timeit.timeit(einsum_string, setup=setup, number=e_n) / e_n
        opt_einsum_time = timeit.timeit(opt_einsum_string, setup=setup, number=o_n) / o_n

        out.append([key, ident, sum_string, scale, einsum_time, opt_einsum_time])

df = pd.DataFrame(out)
df.columns = ['Key', 'Flag', 'String', 'Scale', 'Einsum time', 'Opt_einsum time']
df['Ratio'] = df['Einsum time']/df['Opt_einsum time']

df = df.set_index(['Key', 'String', 'Scale'])
df = df.sort_index()
print df


print '\nDescription of speedup:'
print df['Ratio'].describe()
print '\nNumber of opt_einsum operations slower than einsum: %d.' % np.sum(df['Ratio']<0.90)

