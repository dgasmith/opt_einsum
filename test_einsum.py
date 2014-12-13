import numpy as np
import time

from opt_einsum import opt_einsum

# Number of dimensions
max_dims = 6
min_dims = 2

# Size of each dimension
min_size = 5
max_size = 25

# Number of terms
min_terms = 2
max_terms = 7

# Additional parameters
max_indices = 7
max_doubles = 1E7
max_iter = 10

alpha = list('abcdefghijklmnopqrstuvwyxz')
alpha_dict = {num:x for num, x in enumerate(alpha)}
index_size = np.random.randint(min_size, max_size, max_indices)

print 'Maximum term size is %d' % (max_size**max_dims)

def make_term():
    num_dims = np.random.randint(min_dims, max_dims)
    term = np.random.randint(0, max_indices, num_dims)

    return term

def get_string(term):
    return ''.join([alpha_dict[x] for x in term])

def random_test():

    # Compute number of terms    
    num_terms = np.random.randint(min_terms, max_terms)

    # Compute size of each index
    index_size = np.random.randint(min_dims, max_dims, num_terms)

    int_terms = [make_term() for x in range(num_terms)]

    views = [np.random.rand(*t) for t in int_terms]
    sum_string = ','.join(get_string(t) for t in int_terms)

    out_string = sum_string.replace(',','')
    out_string = [x for x in alpha if out_string.count(x)==1]



    sum_string += '->'
    #sum_string += '->' + ''.join(out_string)
    ein = np.einsum(sum_string, *views)
    opt = opt_einsum(sum_string, *views)

    ident = np.allclose(ein, opt)
    return (ident, sum_string)

for x in range(50):
    tmp = random_test()
    if not tmp[0]:
        print tmp[1]

