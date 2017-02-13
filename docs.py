import opt_einsum as oe
import numpy as np
a = np.random.rand(2, 2)
b = np.random.rand(2, 5)
c = np.random.rand(5, 2)
#path_info = oe.contract_path('ij,jk,kl->il', a, b, c)


#print path_info[0]
#print path_info[1]
#
#
#I = np.random.rand(10, 10, 10, 10)
#C = np.random.rand(10, 10)
#path_info = oe.contract_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C)
#print(path_info[0])
#print(path_info[1])
#
#
#print oe.contract('ij,jk,kl->il', a, b, c)

#path =  oe.contract_path("..., ...", 3, 4)
#print oe.contract("..., ...", 3, 4)

path =  oe.contract_path("ab,ab,c", a, a, np.random.rand(2))
#print oe.contract("ab,ab,c", a, a, np.random.rand(2))

out = np.empty((2))
np.einsum('mi,mi,mi->m', a, a, a, out=out)
print out
out[:] = 0
path = oe.contract_path('mi,mi,mi->m', a, a, a)
print path[0]
print path[1]
oe.contract('mi,mi,mi->m', a, a, a, out=out)
print out

