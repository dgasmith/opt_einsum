import numpy as np
import pandas as pd
import sys, traceback
import timeit
import time

import test_helper as th
from opt_einsum import contract
pd.set_option('display.width', 200)

import resource
rsrc = resource.RLIMIT_DATA
limit = int(1e10)
resource.setrlimit(rsrc, (limit, limit))

test_einsum = False
test_paths = True
opt_path_time = True
term_thresh = 4
tdot=True

#scale_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
scale_list = [2]

out = []
for key in th.tests.keys():
    sum_string, index_size = th.tests[key]
    for scale in scale_list:
        views = th.build_views(sum_string, index_size, scale=scale)

        # At this point lets assume everything works correctly
        t = time.time()
        opt_path = contract(sum_string, *views, path='optimal', return_path=True)
        opt_time = time.time() - t
        opp_path = contract(sum_string, *views, path='opportunistic', return_path=True)
        if opt_path_time and (len(views) > term_thresh):
            print 'Path optimal took %3.5f seconds for %d terms.' % (opt_time, len(views))

        # If identical paths lets just skip them
        if all(x==y for x, y in zip(opp_path, opt_path)):
            break

        setup = "import numpy as np; from opt_einsum import contract; \
                 from __main__ import sum_string, views, opt_path, opp_path, tdot"
        opportunistic_string = "contract(sum_string, *views, path=opp_path, tensordot=tdot)"
        optimal_string = "contract(sum_string, *views, path=opt_path, tensordot=tdot)"

        # Optional test
        if test_paths:
            opp = contract(sum_string, *views, path=opp_path)
            opt = contract(sum_string, *views, path=opt_path)
            assert np.allclose(opp, opt)
        if test_einsum and test_paths:
            assert np.allclose(opp, np.einsum(sum_string, *views))

        num_loops = 5
        optimal_time = timeit.timeit(opportunistic_string, setup=setup, number=num_loops) / num_loops
        opportunistic_time = timeit.timeit(optimal_string, setup=setup, number=num_loops) / num_loops

        out.append([key, sum_string, scale, optimal_time, opportunistic_time])

df = pd.DataFrame(out)
df.columns = ['Key', 'String', 'Scale', 'Optimal time', 'Opportunistic time']
df['Ratio'] = np.around(df['Opportunistic time']/df['Optimal time'], 2)

df = df.set_index(['Key', 'Scale'])
df = df.sort_index()
print df


print '\nDescription of speedup:'
print df['Ratio'].describe()
print '\nNumber of optimal paths slower than opportunistic paths: %d.' % np.sum(df['Ratio']<0.8)

