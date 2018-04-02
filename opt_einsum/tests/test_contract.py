"""
Tets a series of opt_einsum contraction paths to ensure the results are the same for different paths
"""

import sys

import numpy as np
import pytest

from opt_einsum import contract, contract_path, helpers, contract_expression

tests = [
    # Test hadamard-like products
    'a,ab,abc->abc',
    'a,b,ab->ab',

    # Test index-transformations
    'ea,fb,gc,hd,abcd->efgh',
    'ea,fb,abcd,gc,hd->efgh',
    'abcd,ea,fb,gc,hd->efgh',

    # Test complex contractions
    'acdf,jbje,gihb,hfac,gfac,gifabc,hfac',
    'acdf,jbje,gihb,hfac,gfac,gifabc,hfac',
    'cd,bdhe,aidb,hgca,gc,hgibcd,hgac',
    'abhe,hidj,jgba,hiab,gab',
    'bde,cdh,agdb,hica,ibd,hgicd,hiac',
    'chd,bde,agbc,hiad,hgc,hgi,hiad',
    'chd,bde,agbc,hiad,bdi,cgh,agdb',
    'bdhe,acad,hiab,agac,hibd',

    # Test collapse
    'ab,ab,c->',
    'ab,ab,c->c',
    'ab,ab,cd,cd->',
    'ab,ab,cd,cd->ac',
    'ab,ab,cd,cd->cd',
    'ab,ab,cd,cd,ef,ef->',

    # Test outer prodcuts
    'ab,cd,ef->abcdef',
    'ab,cd,ef->acdf',
    'ab,cd,de->abcde',
    'ab,cd,de->be',
    'ab,bcd,cd->abcd',
    'ab,bcd,cd->abd',

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

    # Inner products
    'ab,ab',
    'ab,ba',
    'abc,abc',
    'abc,bac',
    'abc,cba',

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

    ein = contract(string, *views, optimize=False, use_blas=False)
    opt = contract(string, *views, optimize='greedy', use_blas=False)
    assert np.allclose(ein, opt)

    opt = contract(string, *views, optimize='optimal', use_blas=False)
    assert np.allclose(ein, opt)


@pytest.mark.skipif(sys.version_info[0] < 3, reason='requires python3')
@pytest.mark.parametrize("string", tests)
def test_compare_greek(string):
    views = helpers.build_views(string)

    ein = contract(string, *views, optimize=False, use_blas=False)

    # convert to greek
    string = ''.join(chr(ord(c) + 848) if c not in ',->.' else c for c in string)

    opt = contract(string, *views, optimize='greedy', use_blas=False)
    assert np.allclose(ein, opt)

    opt = contract(string, *views, optimize='optimal', use_blas=False)
    assert np.allclose(ein, opt)


@pytest.mark.parametrize("string", tests)
def test_compare_blas(string):
    views = helpers.build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(string, *views, optimize='greedy')
    assert np.allclose(ein, opt)

    opt = contract(string, *views, optimize='optimal')
    assert np.allclose(ein, opt)


@pytest.mark.skipif(sys.version_info[0] < 3, reason='requires python3')
@pytest.mark.parametrize("string", tests)
def test_compare_blas_greek(string):
    views = helpers.build_views(string)

    ein = contract(string, *views, optimize=False)

    # convert to greek
    string = ''.join(chr(ord(c) + 848) if c not in ',->.' else c for c in string)

    opt = contract(string, *views, optimize='greedy')
    assert np.allclose(ein, opt)

    opt = contract(string, *views, optimize='optimal')
    assert np.allclose(ein, opt)


@pytest.mark.skipif(sys.version_info[0] < 3, reason='requires python3')
def test_some_non_alphabet_maintains_order():
    # 'c beta a' should automatically go to -> 'a c beta'
    string = 'c' + chr(ord('b') + 848) + 'a'
    # but beta will be temporarily replaced with 'b' for which 'cba->abc'
    # so check manual output kicks in:
    x = np.random.rand(2, 3, 4)
    assert np.allclose(contract(string, x), contract('cxa', x))


def test_printing():
    string = "bbd,bda,fc,db->acf"
    views = helpers.build_views(string)

    ein = contract_path(string, *views)
    assert len(ein[1]) == 729


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("optimize", ['greedy', 'optimal'])
@pytest.mark.parametrize("use_blas", [False, True])
@pytest.mark.parametrize("out_spec", [False, True])
def test_contract_expressions(string, optimize, use_blas, out_spec):
    views = helpers.build_views(string)
    shapes = [view.shape for view in views]
    expected = contract(string, *views, optimize=False, use_blas=False)

    expr = contract_expression(
        string, *shapes, optimize=optimize, use_blas=use_blas)

    if out_spec and ("->" in string) and (string[-2:] != "->"):
        out, = helpers.build_views(string.split('->')[1])
        expr(*views, out=out)
    else:
        out = expr(*views)

    assert np.allclose(out, expected)

    # check representations
    assert string in expr.__repr__()
    assert string in expr.__str__()


def test_contract_expression_checks():
    # check optimize needed
    with pytest.raises(ValueError):
        contract_expression("ab,bc->ac", (2, 3), (3, 4), optimize=False)

    # check sizes are still checked
    with pytest.raises(ValueError):
        contract_expression("ab,bc->ac", (2, 3), (3, 4), (42, 42))

    # check if out given
    out = np.empty((2, 4))
    with pytest.raises(ValueError):
        contract_expression("ab,bc->ac", (2, 3), (3, 4), out=out)

    # check still get errors when wrong ranks supplied to expression
    expr = contract_expression("ab,bc->ac", (2, 3), (3, 4))

    # too few arguments
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3))
    assert "`ContractExpression` takes exactly 2" in str(err)

    # too many arguments
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3), np.random.rand(2, 3), np.random.rand(2, 3))
    assert "`ContractExpression` takes exactly 2" in str(err)

    # wrong shapes
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3, 4), np.random.rand(3, 4))
    assert "Internal error while evaluating `ContractExpression`" in str(err)
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 4), np.random.rand(3, 4, 5))
    assert "Internal error while evaluating `ContractExpression`" in str(err)
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3), np.random.rand(3, 4),
             out=np.random.rand(2, 4, 6))
    assert "Internal error while evaluating `ContractExpression`" in str(err)

    # should only be able to specify out
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3), np.random.rand(3, 4), order='F')
    assert "only valid keyword arguments to a `ContractExpression`" in str(err)


def test_broadcasting_contraction():

    a = np.random.rand(1, 5, 4)
    b = np.random.rand(4, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(10)

    ein_scalar = contract('ijk,kl,jl', a, b, c, optimize=False)
    opt_scalar = contract('ijk,kl,jl', a, b, c, optimize=True)
    assert np.allclose(ein_scalar, opt_scalar)    

    result = ein_scalar * d

    ein = contract('ijk,kl,jl,i->i', a, b, c, d, optimize=False)
    opt = contract('ijk,kl,jl,i->i', a, b, c, d, optimize=True)

    assert np.allclose(ein, result)
    assert np.allclose(opt, result)

def test_broadcasting_contraction2():

    a = np.random.rand(1, 1, 5, 4)
    b = np.random.rand(4, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(7, 7)

    ein_scalar = contract('abjk,kl,jl', a, b, c, optimize=False)
    opt_scalar = contract('abjk,kl,jl', a, b, c, optimize=True)
    assert np.allclose(ein_scalar, opt_scalar)    

    result = ein_scalar * d

    ein = contract('abjk,kl,jl,ab->ab', a, b, c, d, optimize=False)
    opt = contract('abjk,kl,jl,ab->ab', a, b, c, d, optimize=True)

    assert np.allclose(ein, result)
    assert np.allclose(opt, result)

def test_broadcasting_contraction3():

    a = np.random.rand(1, 5, 4)
    b = np.random.rand(4, 1, 6)
    c = np.random.rand(5, 6)
    d = np.random.rand(7, 7)

    ein = contract('ajk,kbl,jl,ab->ab', a, b, c, d, optimize=False)
    opt = contract('ajk,kbl,jl,ab->ab', a, b, c, d, optimize=True)

    assert np.allclose(ein, opt)
