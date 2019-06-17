import pytest

llcreader = pytest.importorskip("xmitgcm.llcreader")
from .test_xmitgcm_common import llc_mds_datadirs

EXPECTED_VARS = ['Eta', 'KPPhbl', 'oceFWflx', 'oceQnet', 'oceQsw', 'oceSflux',
            'oceTAUX', 'oceTAUY', 'PhiBot', 'Salt', 'SIarea', 'SIheff',
            'SIhsalt', 'SIhsnow', 'SIuice', 'SIvice', 'Theta', 'U', 'V', 'W']

########### Generic llcreader tests on local data ##############################

@pytest.fixture(scope='module')
def local_llc90_store(llc_mds_datadirs):
    from fsspec.implementations.local import LocalFileSystem
    dirname, expected = llc_mds_datadirs
    fs = LocalFileSystem()
    return llcreader.BaseStore(fs, base_path=dirname)

@pytest.fixture(scope='module')
def llc90_kwargs():
    return dict(varnames=['S', 'T', 'U', 'V', 'Eta'],
                iter_start=0, iter_stop=9, iter_step=8)

def test_llc90_local_faces(local_llc90_store, llc90_kwargs):
    store = local_llc90_store
    model = llcreader.LLC90Model(store)
    ds_faces = model.get_dataset(**llc90_kwargs)
    assert set(llc90_kwargs['varnames']) == set(ds_faces.data_vars)
    assert ds_faces.dims == {'face': 13, 'i': 90, 'i_g': 90, 'j': 90, 'j_g': 90,
                             'k': 50, 'k_u': 50, 'k_l': 50, 'k_p1': 50, 'time': 2}

def test_llc90_local_latlon(local_llc90_store, llc90_kwargs):
    store = local_llc90_store
    model = llcreader.LLC90Model(store)
    ds_latlon = model.get_dataset(type='latlon', **llc90_kwargs)
    assert set(llc90_kwargs['varnames']) == set(ds_latlon.data_vars)
    assert ds_latlon.dims == {'i': 360, 'time': 2, 'k_p1': 50, 'face': 13,
                              'i_g': 360, 'k_u': 50, 'k': 50, 'k_l': 50,
                              'j_g': 270, 'j': 270}

@pytest.mark.parametrize('rettype', ['faces', 'latlon'])
@pytest.mark.parametrize('k_levels', [None, [0, 2, 7, 9, 10, 20]])
@pytest.mark.parametrize('k_chunksize', [1, 2])
def test_llc90_local_faces_load(local_llc90_store, llc90_kwargs, rettype, k_levels,
                                k_chunksize):
    store = local_llc90_store
    model = llcreader.LLC90Model(store)
    ds = model.get_dataset(k_levels=k_levels, k_chunksize=k_chunksize,
                           type=rettype, **llc90_kwargs)
    if k_levels is None:
        assert list(ds.k.values) == list(range(50))
    else:
        assert list(ds.k.values) == k_levels
    assert all([cs==k_chunksize for cs in ds['T'].data.chunks[1]])

    ds.load()

########### ECCO Portal Tests ##################################################

@pytest.fixture(scope='module')
def ecco_portal_model():
    # TODO: include other models like 1080, etc.
    return llcreader.ECCOPortalLLC4320Model()

def test_ecco_portal_faces(ecco_portal_model):
    ds_faces = ecco_portal_model.get_dataset(iter_start=10368, iter_stop=11000)
    assert ds_faces.dims == {'face': 13, 'i': 4320, 'i_g': 4320, 'j': 4320,
                              'j_g': 4320, 'k': 90, 'k_u': 90, 'k_l': 90,
                              'k_p1': 90, 'time': 5}
    assert set(EXPECTED_VARS) == set(ds_faces.data_vars)

def test_ecco_portal_load(ecco_portal_model):
    # an expensive test because it actually loads data
    # results depend on the specific dataset
    ds_faces = ecco_portal_model.get_dataset(iter_start=10368, iter_stop=11000)
    assert ds_faces.Eta[0, 0, -1, -1].values.item() == -1.262018084526062

def test_ecco_portal_latlon(ecco_portal_model):
    ds_ll = ecco_portal_model.get_dataset(iter_start=10368, iter_stop=11000,
                                             type='latlon')
    assert ds_ll.dims == {'i': 17280, 'k_u': 90, 'k_l': 90, 'time': 5,
                             'k': 90, 'j_g': 12960, 'i_g': 17280, 'k_p1': 90,
                             'j': 12960, 'face': 13}
    assert set(EXPECTED_VARS) == set(ds_ll.data_vars)
