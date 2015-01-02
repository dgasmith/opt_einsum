import numpy as np
import pandas as pd
import time

import test_helper as th
from opt_einsum import opt_einsum
pd.set_option('display.width', 1000)


#key = 'Index1'
#key = 'Actual3'
key = 'Expand4'
#key = 'Dot4'

#path_arg = 'optimal'
path_arg = 'opportunistic'

scale = 1.2

sum_string, index_size = th.tests[key]
index_size = np.ceil(np.array(index_size) * scale).astype(np.int)
views = th.build_views(sum_string, index_size)
t = time.time()
ein_result = np.einsum(sum_string, *views)
print 'Einsum took %3.3f seconds' % (time.time() - t)
print  opt_einsum(sum_string, *views, debug=1, path=path_arg, return_path=True)
t = time.time()
opt_ein_result = opt_einsum(sum_string, *views, debug=1, path=path_arg)
print 'Opt_einsum took %3.3f seconds' % (time.time() - t)


print 'Einsum shape:        %s' % (str(ein_result.shape))
print 'Opt einsum shape:    %s' % (str(opt_ein_result.shape))
print 'Allclose:            %s' % np.allclose(ein_result, opt_ein_result)



