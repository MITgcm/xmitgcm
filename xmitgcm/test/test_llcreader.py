import pytest
from dask.array.core import Array as dsa

llcreader = pytest.importorskip("xmitgcm.llcreader")

from .test_xmitgcm_common import llc_mds_datadirs

EXPECTED_VARS = ['Eta', 'KPPhbl', 'oceFWflx', 'oceQnet', 'oceQsw', 'oceSflux',
            'oceTAUX', 'oceTAUY', 'PhiBot', 'Salt', 'SIarea', 'SIheff',
            'SIhsalt', 'SIhsnow', 'SIuice', 'SIvice', 'Theta', 'U', 'V', 'W']
GRID_VARNAMES = ['AngleCS', 'AngleSN', 'DRC', 'DRF', 'DXC', 'DXG', 'DYC', 'DYG',
                 'Depth', 'PHrefC', 'PHrefF', 'RAC', 'RAS', 'RAW', 'RAZ', 'RC', 'RF',
                 'RhoRef', 'XC', 'XG', 'YC', 'YG', 'hFacC', 'hFacS', 'hFacW']


EXPECTED_COORDS = {2160: ['CS','SN','Depth',
                          'drC','drF','dxC','dxF','dxG','dyC','dyF','dyG',
                          'hFacC','hFacS','hFacW','PHrefC','PHrefF','rA','rAs','rAw',
                          'Z','Zp1','Zl','Zu','rhoRef','rLowC','rLowS','rLowW',
                          'rSurfC','rSurfS','rSurfW','XC','YC','rAz','XG','YG',
                          'dxV','dyU'],
                   4320: ['CS','SN','Depth',
                          'drC','drF','dxC','dxF','dxG','dyC','dyF','dyG',
                          'hFacC','hFacS','hFacW','PHrefC','PHrefF',
                          'rA','rAs','rAw','rhoRef','Z','Zp1','Zl','Zu','XC','YC',
                          'rAz','XG','YG','dxV','dyU'],
                   'aste_270': ["CS", "Depth", "PHrefC", "PHrefF", "SN",
                                "XC", "XG", "YC", "YG", "Z",
                                "Zl", "Zp1", "Zu", "drC", "drF",
                                "dxC", "dxG", "dyC", "dyG", "hFacC",
                                "hFacS", "hFacW", "maskC", "maskCtrlC", "maskCtrlS",
                                "maskCtrlW", "maskInC", "maskInS", "maskInW", "maskS",
                                "maskW", "niter", "rA", "rAs", "rAw",
                                "rAz", "rhoRef"]}

########### Generic llcreader tests on local data ##############################

@pytest.fixture(scope='module')
def local_llc90_store(llc_mds_datadirs):
    from fsspec.implementations.local import LocalFileSystem
    dirname, expected = llc_mds_datadirs
    fs = LocalFileSystem()
    store = llcreader.BaseStore(fs, base_path=dirname, grid_path=dirname)
    return store

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
                             'k': 50, 'k_u': 50, 'k_l': 50, 'k_p1': 51, 'time': 2}

def test_llc90_dim_metadata(local_llc90_store, llc90_kwargs):
    store = local_llc90_store
    model = llcreader.LLC90Model(store)
    ds_faces = model.get_dataset(**llc90_kwargs)
    assert ds_faces.i.attrs['axis'] == 'X'

def test_llc90_local_latlon(local_llc90_store, llc90_kwargs):
    store = local_llc90_store
    model = llcreader.LLC90Model(store)
    ds_latlon = model.get_dataset(type='latlon', **llc90_kwargs)
    assert set(llc90_kwargs['varnames']) == set(ds_latlon.data_vars)
    assert ds_latlon.dims == {'i': 360, 'time': 2, 'k_p1': 51, 'face': 13,
                              'i_g': 360, 'k_u': 50, 'k': 50, 'k_l': 50,
                              'j_g': 270, 'j': 270}


# includes regression test for https://github.com/MITgcm/xmitgcm/issues/233
@pytest.mark.parametrize('rettype', ['faces', 'latlon'])
@pytest.mark.parametrize('k_levels, kp1_levels, k_chunksize',
        [(None, None, 1),
         ([1], [1, 2], 1),
         ([0, 2, 7, 9, 10, 20],
          [0,1,2,3,7,8,9,10,11,20,21], 1),
         ([0, 2, 7, 9, 10, 20],
          [0,1,2,3,7,8,9,10,11,20,21], 2)
         ])
@pytest.mark.parametrize('read_grid', [False, True]
)
def test_llc90_local_faces_load(local_llc90_store, llc90_kwargs, rettype, k_levels,
                                kp1_levels, k_chunksize, read_grid):
    store = local_llc90_store
    model = llcreader.LLC90Model(store)
    model.grid_varnames = GRID_VARNAMES
    ds = model.get_dataset(k_levels=k_levels, k_chunksize=k_chunksize,
                           type=rettype, read_grid=read_grid, **llc90_kwargs)
    if read_grid:
        # doesn't work because the variables change name
        # assert set(GRID_VARNAMES).issubset(set(ds.coords))
        pass
    if k_levels is None:
        assert list(ds.k.values) == list(range(50))
        assert list(ds.k_p1.values) == list(range(51))
    else:
        assert list(ds.k.values) == k_levels
        assert list(ds.k_p1.values) == kp1_levels
    assert all([cs==k_chunksize for cs in ds['T'].data.chunks[1]])

    ds.load()


########### ECCO Portal Tests ##################################################

@pytest.fixture(scope='module', params=[2160, 4320])
def ecco_portal_model(request):
    if request.param==2160:
        return llcreader.ECCOPortalLLC2160Model()
    else:
        return llcreader.ECCOPortalLLC4320Model()

def test_ecco_portal_faces(ecco_portal_model):
    # just get three timesteps
    iter_stop = ecco_portal_model.iter_start + 2 * ecco_portal_model.iter_step + 1
    ds_faces = ecco_portal_model.get_dataset(iter_stop=iter_stop)
    nx = ecco_portal_model.nx
    assert ds_faces.dims == {'face': 13, 'i': nx, 'i_g': nx, 'j': nx,
                              'j_g': nx, 'k': 90, 'k_u': 90, 'k_l': 90,
                              'k_p1': 91, 'time': 3}
    assert set(EXPECTED_VARS) == set(ds_faces.data_vars)
    assert set(EXPECTED_COORDS[nx]).issubset(set(ds_faces.coords))

    # make sure vertical coordinates are in one single chunk
    for fld in ds_faces[['Z','Zl','Zu','Zp1']].coords:
        if isinstance(ds_faces[fld].data,dsa):
            assert len(ds_faces[fld].data.chunks)==1
            assert (len(ds_faces[fld]),)==ds_faces[fld].data.chunks[0]

@pytest.mark.slow
def test_ecco_portal_load(ecco_portal_model):
    # an expensive test because it actually loads data
    iter_stop = ecco_portal_model.iter_start + 2 * ecco_portal_model.iter_step + 1
    ds_faces = ecco_portal_model.get_dataset(varnames=['Eta'], iter_stop=iter_stop)
    # a lookup table
    expected = {2160: -1.3054643869400024, 4320: -1.262018084526062}
    assert ds_faces.Eta[0, 0, -1, -1].values.item() == expected[ecco_portal_model.nx]

def test_ecco_portal_latlon(ecco_portal_model):
    iter_stop = ecco_portal_model.iter_start + 2 * ecco_portal_model.iter_step + 1
    ds_ll = ecco_portal_model.get_dataset(iter_stop=iter_stop, type='latlon')
    nx = ecco_portal_model.nx
    assert ds_ll.dims == {'i': 4*nx, 'k_u': 90, 'k_l': 90, 'time': 3,
                             'k': 90, 'j_g': 3*nx, 'i_g': 4*nx, 'k_p1': 91,
                             'j': 3*nx, 'face': 13}
    assert set(EXPECTED_VARS) == set(ds_ll.data_vars)
    assert set(EXPECTED_COORDS[nx]).issubset(set(ds_ll.coords))

    # make sure vertical coordinates are in one single chunk
    for fld in ds_ll[['Z','Zl','Zu','Zp1']].coords:
        if isinstance(ds_ll[fld].data,dsa):
            assert len(ds_ll[fld].data.chunks)==1
            assert (len(ds_ll[fld]),)==ds_ll[fld].data.chunks[0]

########### ASTE Portal Tests ##################################################
@pytest.fixture(scope='module')
def aste_portal_model():
    return llcreader.CRIOSPortalASTE270Model()

def test_aste_portal_faces(aste_portal_model):
    # just get three timesteps
    iters = aste_portal_model.iters[:3]
    ds_faces = aste_portal_model.get_dataset(iters=iters)
    nx = aste_portal_model.nx
    assert ds_faces.dims == {'face': 6, 'i': nx, 'i_g': nx, 'j': nx,
                              'j_g': nx, 'k': 50, 'k_u': 50, 'k_l': 50,
                              'k_p1': 51, 'time': 3}
    assert set(aste_portal_model.varnames) == set(ds_faces.data_vars)
    assert set(EXPECTED_COORDS['aste_270']).issubset(set(ds_faces.coords))

    # make sure vertical coordinates are in one single chunk
    for fld in ds_faces[['Z','Zl','Zu','Zp1']].coords:
        if isinstance(ds_faces[fld].data,dsa):
            assert len(ds_faces[fld].data.chunks)==1
            assert (len(ds_faces[fld]),)==ds_faces[fld].data.chunks[0]

@pytest.mark.slow
def test_aste_portal_load(aste_portal_model):
    # an expensive test because it actually loads data
    iters = aste_portal_model.iters[:3]
    ds_faces = aste_portal_model.get_dataset(varnames=['ETAN'], iters=iters)
    expected = 0.641869068145752
    assert ds_faces.ETAN[0, 1, 0, 0].values.item() == expected

def test_aste_portal_latlon(aste_portal_model):
    iters = aste_portal_model.iters[:3]
    with pytest.raises(TypeError):
        ds_ll = aste_portal_model.get_dataset(iters=iters,type='latlon')


