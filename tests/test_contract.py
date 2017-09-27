from __future__ import division, absolute_import, print_function

import numpy as np
from opt_einsum import contract, contract_path, helpers
import pytest

tests = [
    # test_hadamard_like_products
    'a,ab,abc->abc',
    'a,b,ab->ab',
    # test_index_transformations
    'ea,fb,gc,hd,abcd->efgh',
    'ea,fb,abcd,gc,hd->efgh',
    'abcd,ea,fb,gc,hd->efgh',
    # test_complex
    'acdf,jbje,gihb,hfac,gfac,gifabc,hfac',
    'acdf,jbje,gihb,hfac,gfac,gifabc,hfac',
    'cd,bdhe,aidb,hgca,gc,hgibcd,hgac',
    'abhe,hidj,jgba,hiab,gab',
    'bde,cdh,agdb,hica,ibd,hgicd,hiac',
    'chd,bde,agbc,hiad,hgc,hgi,hiad',
    'chd,bde,agbc,hiad,bdi,cgh,agdb',
    'bdhe,acad,hiab,agac,hibd',
    # test_collapse
    'ab,ab,c->',
    'ab,ab,c->c',
    'ab,ab,cd,cd->',
    'ab,ab,cd,cd->ac',
    'ab,ab,cd,cd->cd',
    'ab,ab,cd,cd,ef,ef->',
    # test_expand
    'ab,cd,ef->abcdef',
    'ab,cd,ef->acdf',
    'ab,cd,de->abcde',
    'ab,cd,de->be',
    'ab,bcd,cd->abcd',
    'ab,bcd,cd->abd',
    # test_previously_failed
    # Random test cases that have previously failed
    'eb,cb,fb->cef',
    'dd,fb,be,cdb->cef',
    'bca,cdb,dbf,afc->',
    'dcc,fce,ea,dbf->ab',
    'fdf,cdd,ccd,afe->ae',
    'abcd,ad',
    'ed,fcd,ff,bcf->be',
    'baa,dcf,af,cde->be',
    'bd,db,eac->ace',
    'fff,fae,bef,def->abd',
    'efc,dbc,acf,fd->abe',
    # test_inner_product
    # Inner products
    'ab,ab',
    'ab,ba',
    'abc,abc',
    'abc,bac',
    'abc,cba',
    # test_dot_product
    # GEMM test cases
    'ab,bc',
    'ab,cb',
    'ba,bc',
    'ba,cb',
    'abcd,cd',
    'abcd,ab',
    'abcd,cdef',
    'abcd,cdef->feba',
    'abcd,efdc',
    # Inner than dot
    'aab,bc->ac',
    'ab,bcc->ac',
    'aab,bcc->ac',
    'baa,bcc->ac',
    'aab,ccb->ac',
    # test_random_cases
    # Randomly build test caes
    'aab,fa,df,ecc->bde',
    'ecb,fef,bad,ed->ac',
    'bcf,bbb,fbf,fc->',
    'bb,ff,be->e',
    'bcb,bb,fc,fff->',
    'fbb,dfd,fc,fc->',
    'afd,ba,cc,dc->bf',
    'adb,bc,fa,cfc->d',
    'bbd,bda,fc,db->acf',
    'dba,ead,cad->bce',
    'aef,fbc,dca->bde',
]
@pytest.mark.parametrize("string", tests)
def test_compare(string):
    views = helpers.build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(string, *views)
    assert np.allclose(ein, opt)

    opt = contract(string, *views, optimize='optimal')
    assert np.allclose(ein, opt)

@pytest.mark.parametrize("string", tests)
def test_compare_blas(string):
    views = helpers.build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(string, *views)
    assert np.allclose(ein, opt)

    opt = contract(string, *views, optimize='optimal')
    assert np.allclose(ein, opt)

def test_printing():
    string = "bbd,bda,fc,db->acf"
    views = helpers.build_views(string)

    ein = contract_path(string, *views)
    assert len(ein[1]) == 729

