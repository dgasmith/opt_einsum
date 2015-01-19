import numpy as np
import pandas as pd
import time

import test_helper as th
from opt_einsum import contract
pd.set_option('display.width', 1000)

### Start variables

# Choose a path
#path_arg = 'optimal'
path_arg = 'opportunistic'

# Choose to roughly scale overall time taken
scale = 1

# From current test set
key = 'Failed10'
sum_string, index_size = th.tests[key]

# Build your own
# sum_string = 'ab,bc->ac'
# index_size = [500, 500, 500]

### End variables 

# Grab index and temporary views
views = th.build_views(sum_string, index_size, scale=scale)

t = time.time()
ein_result = np.einsum(sum_string, *views)
print '\nEinsum took %3.5f seconds.\n' % (time.time() - t)

t = time.time()
print 'Building path...'
path = contract(sum_string, *views, path=path_arg, return_path=True)
print 'Path: ', path[0]
print path[1]
print '...path %s took %3.5f seconds.\n' % (path_arg, (time.time() - t))

t = time.time()
opt_ein_result = contract(sum_string, *views, path=path[0])
print 'Opt_einsum took %3.5f seconds' % (time.time() - t)


print 'Einsum shape:        %s' % (str(ein_result.shape))
print 'Opt einsum shape:    %s' % (str(opt_ein_result.shape))
print 'Allclose:            %s' % np.allclose(ein_result, opt_ein_result)



