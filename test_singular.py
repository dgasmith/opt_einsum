import numpy as np
import pandas as pd

import test_helper as th
from opt_einsum import opt_einsum
pd.set_option('display.width', 1000)


#key = 'Index1'
#key = 'Actual3'
key = 'Expand4'
#key = 'Hadamard3'

path_arg = 'optimal'
#path_arg = 'opportunistic'

sum_string, index_size = th.tests[key]
views = th.build_views(sum_string, index_size)
ein_result = np.einsum(sum_string, *views)
print  opt_einsum(sum_string, *views, debug=1, path=path_arg, return_path=True)
opt_ein_result = opt_einsum(sum_string, *views, debug=1, path=path_arg)


print 'Einsum shape:        %s' % (str(ein_result.shape))
print 'Opt einsum shape:    %s' % (str(opt_ein_result.shape))
print 'Allclose:            %s' % np.allclose(ein_result, opt_ein_result)



