from __future__ import division, absolute_import, print_function

import numpy as np
from opt_einsum import contract
import pytest


def test_type_errors():
    # subscripts must be a string
    with pytest.raises(TypeError):
        contract(0, 0)

    # out parameter must be an array
    with pytest.raises(TypeError):
        contract("", 0, out='test')

    # order parameter must be a valid order
    with pytest.raises(TypeError):
        contract("", 0, order='W')

    # casting parameter must be a valid casting
    with pytest.raises(ValueError):
        contract("", 0, casting='blah')

    # dtype parameter must be a valid dtype
    with pytest.raises(TypeError):
        contract("", 0, dtype='bad_data_type')

    # other keyword arguments are rejected
    with pytest.raises(TypeError):
        contract("", 0, bad_arg=0)

    # issue 4528 revealed a segfault with this call
    with pytest.raises(TypeError):
        contract(*(None,)*63)


def test_value_errors():
    with pytest.raises(ValueError):
        contract("")

    # subscripts must be a string
    with pytest.raises(TypeError):
        contract(0, 0)

    # invalid subscript character
    with pytest.raises(ValueError):
        contract("i%...", [0, 0])
    with pytest.raises(ValueError):
        contract("...j$", [0, 0])
    with pytest.raises(ValueError):
        contract("i->&", [0, 0])

    with pytest.raises(ValueError):
        contract("")
    # number of operands must match count in subscripts string
    with pytest.raises(ValueError):
        contract("", 0, 0)
    with pytest.raises(ValueError):
        contract(",", 0, [0], [0])
    with pytest.raises(ValueError):
        contract(",", [0])

    # can't have more subscripts than dimensions in the operand
    with pytest.raises(ValueError):
        contract("i", 0)
    with pytest.raises(ValueError):
        contract("ij", [0, 0])
    with pytest.raises(ValueError):
        contract("...i", 0)
    with pytest.raises(ValueError):
        contract("i...j", [0, 0])
    with pytest.raises(ValueError):
        contract("i...", 0)
    with pytest.raises(ValueError):
        contract("ij...", [0, 0])

    # invalid ellipsis
    with pytest.raises(ValueError):
        contract("i..", [0, 0])
    with pytest.raises(ValueError):
        contract(".i...", [0, 0])
    with pytest.raises(ValueError):
        contract("j->..j", [0, 0])
    with pytest.raises(ValueError):
        contract("j->.j...", [0, 0])

    # invalid subscript character
    with pytest.raises(ValueError):
        contract("i%...", [0, 0])
    with pytest.raises(ValueError):
        contract("...j$", [0, 0])
    with pytest.raises(ValueError):
        contract("i->&", [0, 0])

    # output subscripts must appear in input
    with pytest.raises(ValueError):
        contract("i->ij", [0, 0])

    # output subscripts may only be specified once
    with pytest.raises(ValueError):
        contract("ij->jij", [[0, 0], [0, 0]])

    # dimensions much match when being collapsed
    with pytest.raises(ValueError):
        contract("ii", np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError):
        contract("ii->i", np.arange(6).reshape(2, 3))

    # broadcasting to new dimensions must be enabled explicitly
    with pytest.raises(ValueError):
        contract("i", np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError):
        contract("i->i", [[0, 1], [0, 1]], out=np.arange(4).reshape(2, 2))



@pytest.mark.parametrize("string", [
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
])
def test_compare(string):
    chars = 'abcdefghij'
    sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4])
    sizes = {c: s for c, s in zip(chars, sizes)}

    views = []
    terms = string.split('->')[0].split(',')
    for term in terms:
        dims = [sizes[x] for x in term]
        views.append(np.random.rand(*dims))

    ein = np.einsum(string, *views)
    opt = contract(string, *views)
    assert np.allclose(ein, opt)
