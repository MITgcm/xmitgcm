from xmitgcm.test.test_xmitgcm_common import *
import xmitgcm
import xarray as xr
from glob import glob
from shutil import copyfile
import py

_TESTDATA_FILENAME = 'testdata.tar.gz'
_TESTDATA_ITERS = [39600, ]
_TESTDATA_DELTAT = 86400

_EXPECTED_GRID_VARS = ['XC', 'YC', 'XG', 'YG', 'Zl', 'Zu', 'Z', 'Zp1', 'dxC',
                       'rAs', 'rAw', 'Depth', 'rA', 'dxG', 'dyG', 'rAz', 'dyC',
                       'PHrefC', 'drC', 'PHrefF', 'drF',
                       'hFacS', 'hFacC', 'hFacW']


# a meta test
def test_file_hiding(all_mds_datadirs):
    dirname, _ = all_mds_datadirs
    basenames = ['XC.data', 'XC.meta']
    for basename in basenames:
        assert os.path.exists(os.path.join(dirname, basename))
    with hide_file(dirname, *basenames):
        for basename in basenames:
            assert not os.path.exists(os.path.join(dirname, basename))
    for basename in basenames:
        assert os.path.exists(os.path.join(dirname, basename))

#########################################################
### Below are all tests that actually create datasets ###
#########################################################


def test_open_mdsdataset_minimal(all_mds_datadirs):
    """Create a minimal xarray object with only dimensions in it."""

    dirname, expected = all_mds_datadirs

    ds = xmitgcm.open_mdsdataset(
        dirname, iters=None, read_grid=False, swap_dims=False,
        geometry=expected['geometry'])

    # the expected dimensions of the dataset
    eshape = expected['shape']
    if len(eshape) == 3:
        nz, ny, nx = eshape
        nface = None
    elif len(eshape) == 4:
        if expected['geometry'] == 'cs':
            nz, ny, nface, nx = eshape
        else:
            nz, nface, ny, nx = eshape
    else:
        raise ValueError("Invalid expected shape")
    coords = {'i': np.arange(nx),
              'i_g': np.arange(nx),
              # 'i_z': np.arange(nx),
              'j': np.arange(ny),
              'j_g': np.arange(ny),
              # 'j_z': np.arange(ny),
              'k': np.arange(nz),
              'k_u': np.arange(nz),
              'k_l': np.arange(nz),
              'k_p1': np.arange(nz+1)}
    if nface is not None:
        coords['face'] = np.arange(nface)

    if 'layers' in expected:
        for layer_name, n_layers in expected['layers'].items():
            for suffix, offset in zip(['bounds', 'center', 'interface'],
                                      [0, -1, -2]):
                dimname = 'l' + layer_name[0] + '_' + suffix[0]
                index = np.arange(n_layers + offset)
                coords[dimname] = index

    ds_expected = xr.Dataset(coords=coords)
    assert ds_expected.equals(ds)

    # check that the datatypes are correct
    for key in coords:
        assert ds[key].dtype == np.int64

    # check for comodo metadata needed by xgcm
    assert ds['i'].attrs['axis'] == 'X'
    assert ds['i_g'].attrs['axis'] == 'X'
    assert ds['i_g'].attrs['c_grid_axis_shift'] == -0.5
    assert ds['j'].attrs['axis'] == 'Y'
    assert ds['j_g'].attrs['axis'] == 'Y'
    assert ds['j_g'].attrs['c_grid_axis_shift'] == -0.5
    assert ds['k'].attrs['axis'] == 'Z'
    assert ds['k_l'].attrs['axis'] == 'Z'
    assert ds['k_l'].attrs['c_grid_axis_shift'] == -0.5


def test_read_grid(all_mds_datadirs):
    """Make sure we read all the grid variables."""
    dirname, expected = all_mds_datadirs
    ds = xmitgcm.open_mdsdataset(
        dirname, iters=None, read_grid=True,
        geometry=expected['geometry'])

    for vname in _EXPECTED_GRID_VARS:
        assert vname in ds.variables

    # make sure angle is present
    if expected['geometry'] in ['llc', 'curvilinear']:
        assert 'CS' in ds.coords
        assert 'SN' in ds.coords

    # actually load the data, to check for dask-related errors
    ds.load()

def test_values_and_endianness(all_mds_datadirs):
    """Make sure we read all the grid variables."""
    dirname, expected = all_mds_datadirs

    if expected['geometry']=='llc' and (dask.__version__ < '0.11.2'):
        pytest.xfail("LLC value tests require fixed dask")

    # default endianness
    ds = xmitgcm.open_mdsdataset(
                dirname, iters=None, read_grid=True, swap_dims=False,
                geometry=expected['geometry'])
    # now reverse endianness
    ds_le = xmitgcm.open_mdsdataset(
                dirname, iters=None, read_grid=True, endian='<',
                swap_dims=False,
                geometry=expected['geometry'])

    for vname, (idx, val) in expected['expected_values'].items():
        np.testing.assert_allclose(ds[vname].values[idx], val)
        # dask arrays that have been concatenated revert to native endianness
        # https://github.com/dask/dask/issues/1647
        if ds[vname].dtype.byteorder=='>':
            val_le = ds[vname].values.newbyteorder('<')[idx]
            np.testing.assert_allclose(ds_le[vname].values[idx], val_le)

def test_open_dataset_no_meta(all_mds_datadirs):
    """Make sure we read  variables with no .meta files."""
    dirname, expected = all_mds_datadirs

    shape = expected['shape']

    nz = shape[0]
    ny, nx = shape[-2:]
    shape_2d = shape[1:]
    dims_2d = ('j', 'i')
    if expected['geometry'] == 'llc':
        dims_2d = ('face',) + dims_2d
        ny = nx*shape[-3]
    elif expected['geometry'] == 'cs':
        dims_2d = ('j', 'face', 'i')
        if len(shape) == 4:
            nz, ny, nface, nx = shape
        elif len(shape) == 3:
            ny, nface, nx = shape

    dims_3d = dims_2d if nz == 1 else ('k',) + dims_2d
    dims_2d = ('time',) + dims_2d
    dims_3d = ('time',) + dims_3d

    it = expected['test_iternum']
    kwargs = dict(iters=it, geometry=expected['geometry'], read_grid=False,
                  swap_dims=False, default_dtype=expected['dtype'])

    # a 3D file
    to_hide = ['T.%010d.meta' % it, 'Eta.%010d.meta' % it]
    with hide_file(dirname, *to_hide):
        ds = xmitgcm.open_mdsdataset(dirname, prefix=['T', 'Eta'], **kwargs)
        print(ds['T'].dims)
        print(dims_3d)
        assert ds['T'].dims == dims_3d
        assert ds['T'].values.ndim == len(dims_3d)
        assert ds['Eta'].dims == dims_2d
        assert ds['Eta'].values.ndim == len(dims_2d)

        with pytest.raises(IOError):
            xmitgcm.open_mdsdataset(dirname, prefix=['T', 'Eta'], iters=it,
                                    geometry=expected['geometry'],
                                    read_grid=False)
            pytest.fail("Expecting IOError when default_dtype "
                        "is not precised (i.e., None)")

    # now get rid of the variables used to infer dimensions
    with hide_file(dirname, 'XC.meta', 'RC.meta'):
        with pytest.raises(IOError):
            ds = xmitgcm.open_mdsdataset(dirname, prefix=['T', 'Eta'], **kwargs)
        if expected['geometry']=='llc':
            ds = xmitgcm.open_mdsdataset(dirname, prefix=['T', 'Eta'],
                                         nx=nx, nz=nz, **kwargs)
            with hide_file(dirname, *to_hide):
                ds = xmitgcm.open_mdsdataset(dirname, prefix=['T', 'Eta'],
                                             nx=nx, nz=nz, **kwargs)
        else:
            ds = xmitgcm.open_mdsdataset(dirname, prefix=['T', 'Eta'],
                                         nx=nx, ny=ny, nz=nz, **kwargs)
            with hide_file(dirname, *to_hide):
                ds = xmitgcm.open_mdsdataset(dirname, prefix=['T', 'Eta'],
                                             nx=nx, ny=ny, nz=nz, **kwargs)

    # try just hiding RC
    with hide_file(dirname, 'RC.meta'):
        if expected['geometry']=='llc':
            ds = xmitgcm.open_mdsdataset(dirname, prefix=['T', 'Eta'],
                                         nz=nz, **kwargs)
        else:
            ds = xmitgcm.open_mdsdataset(dirname, prefix=['T', 'Eta'],
                                         nz=nz, **kwargs)

def test_open_dataset_2D_diags(all_mds_datadirs):
    # convert 3D fields with only 2D diagnostic output
    # https://github.com/xgcm/xmitgcm/issues/140
    dirname, expected = all_mds_datadirs

    shape = expected['shape']

    nz = shape[0]
    ny, nx = shape[-2:]
    shape_2d = shape[1:]
    dims_2d = ('j', 'i')
    if expected['geometry']=='llc':
        dims_2d = ('face',) + dims_2d
        ny = nx*shape[-3]
    dims_3d = dims_2d if nz==1 else ('k',) + dims_2d
    dims_2d = ('time',) + dims_2d
    dims_3d = ('time',) + dims_3d

    it = expected['test_iternum']
    kwargs = dict(iters=it, geometry=expected['geometry'], read_grid=False,
                  swap_dims=False)

    to_hide = ['T.%010d.meta' % it, 'T.%010d.data' % it]
    with hide_file(dirname, *to_hide):

        ldir = py.path.local(dirname)
        old_prefix = 'Eta.%010d' % it
        new_prefix = 'T.%010d' % it
        for suffix in ['.data', '.meta']:
            lp = ldir.join(old_prefix + suffix)
            lp.copy(ldir.join(new_prefix + suffix))

        ds = xmitgcm.open_mdsdataset(dirname, prefix=['T'], **kwargs)

def test_swap_dims(all_mds_datadirs):
    """See if we can swap dimensions."""

    dirname, expected = all_mds_datadirs
    kwargs = dict(iters=None, read_grid=True, geometry=expected['geometry'])

    expected_dims = ['XC', 'XG', 'YC', 'YG', 'Z', 'Zl', 'Zp1', 'Zu']

    # make sure we never swap if not reading grid
    assert 'i' in xmitgcm.open_mdsdataset(dirname,
        iters=None, read_grid=False, geometry=expected['geometry'])
    if expected['geometry'] in ('llc', 'cs', 'curvilinear'):
        # make sure swapping is not the default
        ds = xmitgcm.open_mdsdataset(dirname, **kwargs)
        assert 'i' in ds
        # and is impossible
        with pytest.raises(ValueError) as excinfo:
            ds = xmitgcm.open_mdsdataset(
                        dirname, swap_dims=True, **kwargs)
    else:
        # make sure swapping *IS* the default
        assert 'i' not in xmitgcm.open_mdsdataset(dirname, **kwargs)
        ds = xmitgcm.open_mdsdataset(
                    dirname, geometry=expected['geometry'],
                    iters=None, read_grid=True, swap_dims=True,
                    grid_vars_to_coords=True)


        # check for comodo metadata needed by xgcm
        assert ds['XC'].attrs['axis'] == 'X'
        assert ds['XG'].attrs['axis'] == 'X'
        assert ds['XG'].attrs['c_grid_axis_shift'] == -0.5
        assert ds['YC'].attrs['axis'] == 'Y'
        assert ds['YG'].attrs['axis'] == 'Y'
        assert ds['YG'].attrs['c_grid_axis_shift'] == -0.5
        assert ds['Z'].attrs['axis'] == 'Z'
        assert ds['Zl'].attrs['axis'] == 'Z'
        assert ds['Zl'].attrs['c_grid_axis_shift'] == -0.5

        # add extra layers dimensions if needed
        if 'layers' in expected:
            for layer_name in expected['layers']:
                extra_dims = ['layer_' + layer_name + suffix for suffix in
                              ['_bounds', '_center', '_interface']]
                expected_dims += extra_dims

        assert list(ds.dims.keys()) == expected_dims

        # make sure swapping works with multiple iters
        ds = xmitgcm.open_mdsdataset(dirname, geometry=expected['geometry'],
                                     prefix=['S'])
        #print(ds)
        ds.load()
        assert 'XC' in ds['S'].dims
        assert 'YC' in ds['S'].dims



def test_prefixes(all_mds_datadirs):
    """Make sure we read all the grid variables."""

    dirname, expected = all_mds_datadirs
    prefixes = ['U', 'V', 'W', 'T', 'S', 'PH']  # , 'PHL', 'Eta']
    iters = [expected['test_iternum']]
    ds = xmitgcm.open_mdsdataset(
                dirname, iters=iters, prefix=prefixes,
                read_grid=False, geometry=expected['geometry'])

    for p in prefixes:
        assert p in ds

def test_separate_grid_dir(all_mds_datadirs):
    """Make sure we can have the grid files in a separate directory."""

    dirname, expected = all_mds_datadirs
    prefixes = ['U', 'V', 'W', 'T', 'S', 'PH']  # , 'PHL', 'Eta']
    iters = [expected['test_iternum']]

    with hide_file(dirname,
                    *['XC.meta', 'XC.data', 'RC.meta', 'RC.data']) as grid_dir:
        ds = xmitgcm.open_mdsdataset(
                dirname, grid_dir=grid_dir, iters=iters, prefix=prefixes,
                read_grid=False, geometry=expected['geometry'])
        for p in prefixes:
            assert p in ds

def test_multiple_iters(multidim_mds_datadirs):
    """Test ability to load multiple iters into a single dataset."""

    dirname, expected = multidim_mds_datadirs
    # first try specifying the iters
    ds = xmitgcm.open_mdsdataset(
        dirname, read_grid=False, geometry=expected['geometry'],
        iters=expected['all_iters'],
        prefix=expected['prefixes'])
    assert list(ds.iter.values) == expected['all_iters']

    # now infer the iters, should be the same
    ds2 = xmitgcm.open_mdsdataset(
        dirname, read_grid=False, iters='all', geometry=expected['geometry'],
        prefix=expected['prefixes'])
    assert ds.equals(ds2)

    # In the test datasets, there is no PHL.0000000000.data file.
    # By default we infer the prefixes from the first iteration number, so this
    # leads to an error.
    # (Need to specify iters because there is some diagnostics output with
    # weird iterations numbers present in some experiments.)
    with pytest.raises(IOError):
        ds = xmitgcm.open_mdsdataset(
            dirname, read_grid=False, iters=expected['all_iters'],
            geometry=expected['geometry'])

    # now hide all the PH and PHL files: should be able to infer prefixes fine
    missing_files = [os.path.basename(f)
                     for f in glob(os.path.join(dirname, 'PH*.0*data'))]
    print(missing_files)
    with hide_file(dirname, *missing_files):
        ds = xmitgcm.open_mdsdataset(
            dirname, read_grid=False, iters=expected['all_iters'],
            geometry=expected['geometry'])


def test_date_parsing(mds_datadirs_with_refdate):
    """Verify that time information is decoded properly."""
    dirname, expected = mds_datadirs_with_refdate

    ds = xmitgcm.open_mdsdataset(dirname, iters='all', prefix=['S'],
                              ref_date=expected['ref_date'], read_grid=False,
                              delta_t=expected['delta_t'],
                              geometry=expected['geometry'])

    for i, date in expected['expected_time']:
        assert ds.time[i].values == date

    # since time was decoded, this encoding should be removed from attributes
    assert 'units' not in ds.time.attrs
    assert 'calendar' not in ds.time.attrs

def test_serialize_nonstandard_calendar(multidim_mds_datadirs, tmp_path):
    dirname, expected = multidim_mds_datadirs
    ref_date = '2680-01-01 00:00:00'
    calendar = '360_day'
    ds = xmitgcm.open_mdsdataset(dirname, iters='all', prefix=['S'],
                                 ref_date=ref_date,
                                 calendar=calendar,
                                 read_grid=False,
                                 delta_t=expected['delta_t'],
                                 geometry=expected['geometry'])
    ds.to_zarr(tmp_path / 'test.zarr')


def test_diagnostics(mds_datadirs_with_diagnostics):
    """Try reading dataset with diagnostics output."""
    dirname, expected = mds_datadirs_with_diagnostics

    diag_prefix, expected_diags = expected['diagnostics']
    ds = xmitgcm.open_mdsdataset(dirname,
                                              read_grid=False,
                                              iters=expected['test_iternum'],
                                              prefix=[diag_prefix],
                                              geometry=expected['geometry'])
    for diagname in expected_diags:
        assert diagname in ds
        # check vector mates
        if 'mate' in ds[diagname].attrs:
            mate = ds[diagname].attrs['mate']
            assert ds[mate].attrs['mate'] == diagname

def test_default_diagnostics(mds_datadirs_with_diagnostics):
    """Try reading dataset with diagnostics output."""
    dirname, expected = mds_datadirs_with_diagnostics

    diag_prefix, expected_diags = expected['diagnostics']
    with hide_file(dirname, 'available_diagnostics.log'):
        ds = xmitgcm.open_mdsdataset(dirname,
                                              read_grid=False,
                                              iters=expected['test_iternum'],
                                              prefix=[diag_prefix],
                                              geometry=expected['geometry'])
    for diagname in expected_diags:
        assert diagname in ds
        # check vector mates
        if 'mate' in ds[diagname].attrs:
            mate = ds[diagname].attrs['mate']
            assert ds[mate].attrs['mate'] == diagname

def test_avail_diags_in_grid_dir(mds_datadirs_with_diagnostics):
    """Try reading dataset with diagnostics output."""
    dirname, expected = mds_datadirs_with_diagnostics

    diag_prefix, expected_diags = expected['diagnostics']
    iters = expected['test_iternum']

    with hide_file(dirname,
                    *['XC.meta', 'XC.data', 'RC.meta', 'RC.data',
                      'available_diagnostics.log']) as grid_dir:
        ds = xmitgcm.open_mdsdataset(
                dirname, grid_dir=grid_dir, iters=iters, prefix=[diag_prefix],
                read_grid=False, geometry=expected['geometry'])

    for diagname in expected_diags:
        assert diagname in ds
        if 'mate' in ds[diagname].attrs:
            mate = ds[diagname].attrs['mate']
            assert ds[mate].attrs['mate'] == diagname

def test_layers_diagnostics(layers_mds_datadirs):
    """Try reading dataset with layers output."""
    dirname, expected = layers_mds_datadirs
    ds = xmitgcm.open_mdsdataset(dirname, iters='all', swap_dims=False,
                            geometry=expected['geometry'])
    layer_name = list(expected['layers'].keys())[0]
    layer_id = 'l' + layer_name[0]
    for suf in ['bounds', 'center', 'interface']:
        assert ('layer_' + layer_name + '_' + suf) in ds
        assert (layer_id + '_' + suf[0]) in ds.dims

    # a few random expected variables
    expected_vars = {'LaUH' + layer_name:
                     ('time', layer_id + '_c', 'j', 'i_g'),
                     'LaVH' + layer_name:
                     ('time', layer_id + '_c', 'j_g', 'i'),
                     'LaTs' + layer_name:
                     ('time', layer_id + '_i', 'j', 'i')}
    for var, dims in expected_vars.items():
        assert var in ds
        assert ds[var].dims == dims

@pytest.mark.parametrize("method", ["smallchunks"])
@pytest.mark.parametrize("with_refdate", [True, False])
def test_llc_dims(llc_mds_datadirs, method, with_refdate):
    """Check that the LLC file dimensions are correct."""
    dirname, expected = llc_mds_datadirs
    if with_refdate:
        kwargs = {'delta_t': expected['delta_t'],
                  'ref_date': expected['ref_date']}
    else:
        kwargs = {}
    ds = xmitgcm.open_mdsdataset(dirname,
                            iters=expected['test_iternum'],
                            geometry=expected['geometry'], llc_method=method,
                            **kwargs)

    nz, nface, ny, nx = expected['shape']
    nt = 1

    assert ds.dims['face'] == 13
    assert ds.rA.dims == ('face', 'j', 'i')
    assert ds.rA.values.shape == (nface, ny, nx)
    assert ds.U.dims == ('time', 'k', 'face', 'j', 'i_g')
    assert ds.U.values.shape == (nt, nz, nface, ny, nx)
    assert ds.V.dims == ('time', 'k', 'face', 'j_g', 'i')
    assert ds.V.values.shape == (nt, nz, nface, ny, nx)

    print(ds.U.chunks)
    if method == "smallchunks":
        assert ds.U.chunks == (nt*(1,), nz*(1,), nface*(1,), (ny,), (nx,))


def test_drc_length(all_mds_datadirs):
    """Test that open_mdsdataset is adding an extra level to drC if it has length nr"""
    dirname, expected = all_mds_datadirs
    # Only older versions of the gcm have len(drC) = nr, so force len(drC) = nr for the test
    copyfile(os.path.join(dirname, 'DRF.data'),
             os.path.join(dirname, 'DRC.data'))
    copyfile(os.path.join(dirname, 'DRF.meta'),
             os.path.join(dirname, 'DRC.meta'))
    ds = xmitgcm.open_mdsdataset(
        dirname, iters=None, read_grid=True,
        geometry=expected['geometry'])
    assert len(ds.drC) == (len(ds.drF)+1)


def test_mask_values(all_mds_datadirs):
    """Test that open_mdsdataset generates binary masks with correct values"""

    dirname, expected = all_mds_datadirs
    ds = xmitgcm.open_mdsdataset(
        dirname, iters=None, read_grid=True,
        geometry=expected['geometry'])

    hFac_list = ['hFacC', 'hFacW', 'hFacS']
    mask_list = ['maskC', 'maskW', 'maskS']

    for hFac, mask in zip(hFac_list, mask_list):
        xr.testing.assert_equal(ds[hFac] * ds[mask], ds[hFac])

#
# Series of tests which try to open a dataset with different combinations of
# of options, to identify if ref_date can trigger an error
#


@pytest.mark.parametrize("load", [True, False])
# can't call swap_dims without read_grid=True
@pytest.mark.parametrize("swap_dims, read_grid", [(True, True),
                                                  (False, True),
                                                  (False, False)])
def test_ref_date(mds_datadirs_with_refdate, swap_dims, read_grid, load):
    """With ref_date, without grid."""
    dirname, expected = mds_datadirs_with_refdate

    if expected['geometry']=='llc' and swap_dims:
        pytest.skip("can't swap_dims with geometry=='llc'")

    ds = xmitgcm.open_mdsdataset(dirname, iters='all', prefix=['S'],
                                 ref_date=expected['ref_date'],
                                 delta_t=expected['delta_t'],
                                 geometry=expected['geometry'],
                                 read_grid=read_grid, swap_dims=swap_dims)
    if load:
        ds.time.load()


@pytest.mark.parametrize("method", ["smallchunks"])
def test_llc_extra_metadata(llc_mds_datadirs, method):
    """Check that the LLC reads properly when using extra_metadata."""
    dirname, expected = llc_mds_datadirs
    nz, nface, ny, nx = expected['shape']
    nt = 1

    llc = xmitgcm.utils.get_extra_metadata(domain='llc', nx=nx)

    ds = xmitgcm.open_mdsdataset(dirname,
                                 iters=expected['test_iternum'],
                                 geometry=expected['geometry'],
                                 llc_method=method,
                                 extra_metadata=llc)

    assert ds.dims['face'] == 13
    assert ds.rA.dims == ('face', 'j', 'i')
    assert ds.rA.values.shape == (nface, ny, nx)
    assert ds.U.dims == ('time', 'k', 'face', 'j', 'i_g')
    assert ds.U.values.shape == (nt, nz, nface, ny, nx)
    assert ds.V.dims == ('time', 'k', 'face', 'j_g', 'i')
    assert ds.V.values.shape == (nt, nz, nface, ny, nx)

    if method == "smallchunks":
        assert ds.U.chunks == (nt*(1,), nz*(1,), nface*(1,), (ny,), (nx,))


def test_levels_diagnostics(mds_datadirs_with_inputfiles):
    dirname, expected = mds_datadirs_with_inputfiles

    for diagname, (levels, (idx, value)) in expected['diag_levels'].items():
        ds = xmitgcm.open_mdsdataset(dirname, prefix=[diagname], levels=levels,
                                     geometry=expected['geometry'])

        assert ds['Zl'].values[idx] == value

        with pytest.warns(UserWarning, match='nz will be ignored'):
            xmitgcm.open_mdsdataset(dirname, prefix=[diagname], levels=levels, 
                                    geometry=expected['geometry'], nz=12)
