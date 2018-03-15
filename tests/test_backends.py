import pytest
import numpy as np

from opt_einsum import contract, helpers, contract_expression, backends

try:
    import tensorflow as tf
    found_tensorflow = True
except ImportError:
    found_tensorflow = False

try:
    import os
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    import theano
    found_theano = True
except ImportError:
    found_theano = False

try:
    import cupy
    found_cupy = True
except ImportError:
    found_cupy = False

try:
    import dask.array as da
    found_dask = True
except ImportError:
    found_dask = False

try:
    import sparse
    found_sparse = True
except ImportError:
    found_sparse = False


tests = [
    'ab,bc->ca',
    'abc,bcd,dea',
    'abc,def->fedcba',
]


@pytest.mark.skipif(not found_tensorflow, reason="Tensorflow not installed.")
@pytest.mark.parametrize("string", tests)
def test_tensorflow(string):
    views = helpers.build_views(string)
    ein = contract(string, *views, optimize=False, use_blas=False)
    shps = [v.shape for v in views]

    expr = contract_expression(string, *shps, optimize=True)

    sess = tf.Session()
    with sess.as_default():
        opt = expr(*views, backend='tensorflow')

    assert np.allclose(ein, opt)

    # test non-conversion mode
    tensorflow_views = backends.convert_arrays_to_tensorflow(views)
    expr(*tensorflow_views, backend='tensorflow')


@pytest.mark.skipif(not found_cupy, reason="Cupy not installed.")
@pytest.mark.parametrize("string", tests)
def test_cupy(string):
    views = helpers.build_views(string)
    ein = contract(string, *views, optimize=False, use_blas=False)
    shps = [v.shape for v in views]

    expr = contract_expression(string, *shps, optimize=True)

    opt = expr(*views, backend='cupy')
    assert np.allclose(ein, opt)

    # test non-conversion mode
    cupy_views = backends.convert_arrays_to_cupy(views)
    cupy_opt = expr(*cupy_views, backend='cupy')
    assert isinstance(cupy_opt, cupy.ndarray)
    assert np.allclose(ein, cupy.asnumpy(cupy_opt))


@pytest.mark.skipif(not found_theano, reason="Theano not installed.")
@pytest.mark.parametrize("string", tests)
def test_theano(string):
    views = helpers.build_views(string)
    ein = contract(string, *views, optimize=False, use_blas=False)
    shps = [v.shape for v in views]

    expr = contract_expression(string, *shps, optimize=True)

    opt = expr(*views, backend='theano')
    assert np.allclose(ein, opt)

    # test non-conversion mode
    theano_views = backends.convert_arrays_to_theano(views)
    theano_opt = expr(*theano_views, backend='theano')
    assert isinstance(theano_opt, theano.tensor.TensorVariable)


@pytest.mark.skipif(not found_dask, reason="Dask not installed.")
@pytest.mark.parametrize("string", tests)
def test_dask(string):
    views = helpers.build_views(string)
    ein = contract(string, *views, optimize=False, use_blas=False)
    shps = [v.shape for v in views]
    expr = contract_expression(string, *shps, optimize=True)

    # test non-conversion mode
    da_views = [da.from_array(x, chunks=(2)) for x in views]
    da_opt = expr(*da_views, backend='dask')

    # check type is maintained when not using numpy arrays
    assert isinstance(da_opt, da.Array)

    assert np.allclose(ein, np.array(da_opt))


@pytest.mark.skipif(not found_sparse, reason="Sparse not installed.")
@pytest.mark.parametrize("string", tests)
def test_sparse(string):
    views = helpers.build_views(string)

    # sparsify views so they don't become dense during contraction
    for view in views:
        mask = np.random.choice([False, True], view.shape, True, [0.1, 0.9])
        view[mask] = 0

    ein = contract(string, *views, optimize=False, use_blas=False)
    shps = [v.shape for v in views]
    expr = contract_expression(string, *shps, optimize=True)

    # test non-conversion mode
    sparse_views = [sparse.COO.from_numpy(x) for x in views]
    sparse_opt = expr(*sparse_views, backend='sparse')

    # check type is maintained when not using numpy arrays
    assert isinstance(sparse_opt, sparse.COO)

    assert np.allclose(ein, sparse_opt.todense())
