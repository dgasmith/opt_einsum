from __future__ import division, absolute_import, print_function

import os
import sys
import itertools
import traceback

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_allclose, run_module_suite,
    dec
)

from opt_einsum import contract
import time

### Build dictionary of tests

class TestContract(object):
    def setup(self, n=1):
        chars = 'abcdefghij'
        sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4]) 
        if n!=1:
            sizes *= 1 + np.random.rand(sizes.shape[0]) * n 
            sizes = sizes.astype(np.int)
        self.sizes = {c: s for c, s in zip(chars, sizes)}

    def compare(self, string):
        views = []
        terms = string.split('->')[0].split(',')
        for term in terms:
            dims = [self.sizes[x] for x in term]
            views.append(np.random.rand(*dims))

        ein = np.einsum(string, *views)
        opt = contract(string, *views)
        assert_allclose(ein, opt)

    def test_hadamard_like_products(self):
        self.compare('a,ab,abc->abc')
        self.compare('a,b,ab->ab')

    def test_index_transformations():
        self.compare('ea,fb,abcd,gc,hd->efgh')
    
    def test_complex(self):
        self.compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.compare('cd,bdhe,aidb,hgca,gc,hgibcd,hgac')
        self.compare('abhe,hidj,jgba,hiab,gab')
        self.compare('bde,cdh,agdb,hica,ibd,hgicd,hiac')
        self.compare('chd,bde,agbc,hiad,hgc,hgi,hiad')
        self.compare('chd,bde,agbc,hiad,bdi,cgh,agdb')
        self.compare('bdhe,acad,hiab,agac,hibd')

    def test_collapse(self):
        self.compare('ab,ab,c->')
        self.compare('ab,ab,c->c')
        self.compare('ab,ab,cd,cd->')
        self.compare('ab,ab,cd,cd->ac')
        self.compare('ab,ab,cd,cd->cd')
        self.compare('ab,ab,cd,cd,ef,ef->')

    def test_expand(self):
        self.compare('ab,cd,ef->abcdef')
        self.compare('ab,cd,ef->acdf')
        self.compare('ab,cd,de->abcde')
        self.compare('ab,cd,de->be')
        self.compare('ab,bcd,cd->abcd')
        self.compare('ab,bcd,cd->abd') 

    def test_previously_failed(self):
        self.compare('eb,cb,fb->cef')
        self.compare('dd,fb,be,cdb->cef')
        self.compare('bca,cdb,dbf,afc->')
        self.compare('dcc,fce,ea,dbf->ab')
        self.compare('fdf,cdd,ccd,afe->ae')
        self.compare('abcd,ad')
        self.compare('ed,fcd,ff,bcf->be')
        self.compare('baa,dcf,af,cde->be')
        self.compare('bd,db,eac->ace')

    def test_inner_product(self): 
        # Inner products
        self.compare('ab,ab')
        self.compare('ab,ba')
        self.compare('abc,abc')
        self.compare('abc,bac')
        self.compare('abc,cba')

    def test_dot_product(self):
        # GEMM test cases
        self.compare('ab,bc')
        self.compare('ab,cb')
        self.compare('ba,bc')
        self.compare('ba,cb')
        self.compare('abcd,cd')
        self.compare('abcd,ab')
        self.compare('abcd,cdef')
        self.compare('abcd,cdef->feba')
        self.compare('abcd,efdc')
        # Inner than dot
        self.compare('aab,bc->ac')
        self.compare('ab,bcc->ac')
        self.compare('aab,bcc->ac')
        self.compare('baa,bcc->ac')
        self.compare('aab,ccb->ac')

    def test_random_cases(self):
        # Randomly build test caes
        self.compare('aab,fa,df,ecc->bde')
        self.compare('ecb,fef,bad,ed->ac')
        self.compare('bcf,bbb,fbf,fc->')
        self.compare('bb,ff,be->e')
        self.compare('bcb,bb,fc,fff->')
        self.compare('fbb,dfd,fc,fc->')
        self.compare('afd,ba,cc,dc->bf')
        self.compare('adb,bc,fa,cfc->d')
        self.compare('bbd,bda,fc,db->acf')
        self.compare('dba,ead,cad->bce')
        self.compare('aef,fbc,dca->bde')

t = time.time()
c = TestContract()
c.setup()
c.test_hadamard_like_products()
c.test_complex()
c.test_collapse()
c.test_expand()
c.test_expand()
c.test_previously_failed()
c.test_inner_product()
c.test_dot_product()
c.test_random_cases()
print(time.time()-t)

