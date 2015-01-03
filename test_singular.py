import numpy as np
import pandas as pd
import time

import test_helper as th
from opt_einsum import opt_einsum
pd.set_option('display.width', 1000)


#key = 'Index1'
#key = 'Actual3'
#key = 'Expand4'
key = 'EP_Theory1'

path_arg = 'optimal'
#path_arg = 'opportunistic'

scale = 1.2


sum_string, index_size = th.tests[key]
#sum_string = 'i,i,j'

index_size = np.ceil(np.array(index_size) * scale).astype(np.int)
views = th.build_views(sum_string, index_size)
t = time.time()
ein_result = np.einsum(sum_string, *views)
print 'Einsum took %3.3f seconds' % (time.time() - t)
t = time.time()
print  opt_einsum(sum_string, *views, debug=1, path=path_arg, return_path=True)
print 'Path %s took %3.5f seconds' % (path_arg, (time.time() - t))
t = time.time()
opt_ein_result = opt_einsum(sum_string, *views, debug=1, path=path_arg)
print 'Opt_einsum took %3.3f seconds' % (time.time() - t)


print 'Einsum shape:        %s' % (str(ein_result.shape))
print 'Opt einsum shape:    %s' % (str(opt_ein_result.shape))
print 'Allclose:            %s' % np.allclose(ein_result, opt_ein_result)



