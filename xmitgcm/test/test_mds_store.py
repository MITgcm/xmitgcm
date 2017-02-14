import pytest
import os
import tarfile
import xarray as xr
import numpy as np
import dask
from contextlib import contextmanager
import py
import tempfile
from glob import glob
from shutil import copyfile
import dask
import dask.array as dsa

import xmitgcm

_TESTDATA_FILENAME = 'testdata.tar.gz'
_TESTDATA_ITERS = [39600, ]
_TESTDATA_DELTAT = 86400

_EXPECTED_GRID_VARS = ['XC', 'YC', 'XG', 'YG', 'Zl', 'Zu', 'Z', 'Zp1', 'dxC',
                       'rAs', 'rAw', 'Depth', 'rA', 'dxG', 'dyG', 'rAz', 'dyC',
                       'PHrefC', 'drC', 'PHrefF', 'drF',
                       'hFacS', 'hFacC', 'hFacW']


_xc_meta_content = """ simulation = { 'global_oce_latlon' };
 nDims = [   2 ];
 dimList = [
    90,    1,   90,
    40,    1,   40
 ];
 dataprec = [ 'float32' ];
 nrecords = [     1 ];
"""


@contextmanager
def hide_file(origdir, *basenames):
    """Temporarily hide files within the context."""
    # make everything a py.path.local
    tmpdir = py.path.local(tempfile.mkdtemp())
    origdir = py.path.local(origdir)
    oldpaths = [origdir.join(basename) for basename in basenames]
    newpaths = [tmpdir.join(basename) for basename in basenames]

    # move the files
    for oldpath, newpath in zip(oldpaths, newpaths):
        oldpath.rename(newpath)

    try:
        yield str(tmpdir)
    finally:
        # move them back
        for oldpath, newpath in zip(oldpaths, newpaths):
            newpath.rename(oldpath)


# parameterized fixture are complicated
# http://docs.pytest.org/en/latest/fixture.html#fixture-parametrize

# dictionary of archived experiments and some expected properties
_experiments = {
    'global_oce_latlon': {'geometry': 'sphericalpolar',
                          'shape': (15, 40, 90), 'test_iternum': 39600,
                          'expected_values': {'XC': ((0,0), 2)},
                          'dtype': np.dtype('f4'),
                          'layers': {'1RHO': 31},
                          'diagnostics': ('DiagGAD-T',
                              ['TOTTTEND', 'ADVr_TH', 'ADVx_TH', 'ADVy_TH',
                               'DFrE_TH', 'DFxE_TH', 'DFyE_TH', 'DFrI_TH',
                               'UTHMASS', 'VTHMASS', 'WTHMASS'])},
    'barotropic_gyre': {'geometry': 'cartesian',
                        'shape': (1, 60, 60), 'test_iternum': 10,
                        'dtype': np.dtype('f4'),
                          'expected_values': {'XC': ((0,0), 10000.0)},
                        'all_iters': [0, 10],
                        'prefixes': ['T', 'S', 'Eta', 'U', 'V', 'W']},
    'internal_wave': {'geometry': 'sphericalpolar',
                      'shape': (20, 1, 30), 'test_iternum': 100,
                      'dtype': np.dtype('f8'),
                      'expected_values': {'XC': ((0,0), 109.01639344262296)},
                      'all_iters': [0, 100, 200],
                      'ref_date': "1990-1-1",
                      'delta_t': 60,
                      'expected_time':[
                        (0, np.datetime64('1990-01-01T00:00:00.000000000')),
                        (1, np.datetime64('1990-01-01T01:40:00.000000000'))],
                      # these diagnostics won't load because not all levels
                      # were output...no idea how to overcome that bug
                      # 'diagnostics': ('diagout1', ['UVEL', 'VVEL']),
                      'prefixes': ['T', 'S', 'Eta', 'U', 'V', 'W']},
    'global_oce_llc90': {'geometry': 'llc',
                         'ref_date': "1948-01-01 12:00:00",
                         'delta_t': 3600,
                         'expected_time':[
                             (0, np.datetime64('1948-01-01T12:00:00.000000000')),
                             (1, np.datetime64('1948-01-01T20:00:00.000000000'))],
                         'shape': (50, 13, 90, 90), 'test_iternum': 8,
                         'dtype': np.dtype('f4'),
                         'expected_values': {'XC': ((2,3,5), -32.5)},
                         'diagnostics': ('state_2d_set1', ['ETAN', 'SIarea',
                            'SIheff', 'SIhsnow', 'DETADT2', 'PHIBOT',
                            'sIceLoad', 'MXLDEPTH', 'oceSPDep', 'SIatmQnt',
                            'SIatmFW', 'oceQnet', 'oceFWflx', 'oceTAUX',
                            'oceTAUY', 'ADVxHEFF', 'ADVyHEFF', 'DFxEHEFF',
                            'DFyEHEFF', 'ADVxSNOW', 'ADVySNOW', 'DFxESNOW',
                            'DFyESNOW', 'SIuice', 'SIvice'])},
    'curvilinear_leman': {'geometry': 'curvilinear',
                          'delta_t': 20,
                          'ref_date': "2013-11-12 12:00",
                          'shape': (35, 64, 340),
                          'test_iternum': 6,
                          'dtype': np.dtype('f4'),
                          'expected_values': {'XC': ((0,0), 501919.21875)},
                          'all_iters': [0, 3, 6],
                          'expected_time':[
                            (0, np.datetime64('2013-11-12T12:00:00.000000000')),
                            (1, np.datetime64('2013-11-12T12:02:00.000000000'))],
                          'prefixes': ['THETA']}
}


def setup_mds_dir(tmpdir_factory, request):
    """Helper function for setting up test cases."""
    expt_name = request.param
    expected_results = _experiments[expt_name]
    target_dir = str(tmpdir_factory.mktemp('mdsdata'))
    data_dir = os.path.dirname(request.module.__file__)
    return untar(data_dir, expt_name, target_dir), expected_results


def untar(data_dir, basename, target_dir):
    """Unzip a tar file into the target directory. Return path to unzipped
    directory."""
    datafile = os.path.join(data_dir, basename + '.tar.gz')
    if not os.path.exists(datafile):
        raise IOError('Could not find data file %s' % datafile)
    tar = tarfile.open(datafile)
    tar.extractall(target_dir)
    tar.close()
    # subdirectory where file should have been untarred.
    # assumes the directory is the same name as the tar file itself.
    # e.g. testdata.tar.gz --> testdata/
    fulldir = os.path.join(target_dir, basename)
    if not os.path.exists(fulldir):
        raise IOError('Could not find tar file output dir %s' % fulldir)
    # the actual data lives in a file called testdata
    return fulldir


# find the tar archive in the test directory
# http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
@pytest.fixture(scope='module', params=_experiments.keys())
def all_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)


@pytest.fixture(scope='module', params=['barotropic_gyre', 'internal_wave'])
def multidim_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)

@pytest.fixture(scope='module', params=['global_oce_latlon',
                                        'global_oce_llc90'])
def mds_datadirs_with_diagnostics(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)


@pytest.fixture(scope='module', params=['internal_wave', 'global_oce_llc90'])
def mds_datadirs_with_refdate(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)


@pytest.fixture(scope='module', params=['global_oce_latlon'])
def layers_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)

@pytest.fixture(scope='module', params=['global_oce_llc90'])
def llc_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)

def test_parse_meta(tmpdir):
    """Check the parsing of MITgcm .meta into python dictionary."""

    from xmitgcm.utils import parse_meta_file
    p = tmpdir.join("XC.meta")
    p.write(_xc_meta_content)
    fname = str(p)
    result = parse_meta_file(fname)
    expected = {
        'nrecords': 1,
        'basename': 'XC',
        'simulation': "'global_oce_latlon'",
        'dimList': [[90, 1, 90], [40, 1, 40]],
        'nDims': 2,
        'dataprec': np.dtype('float32')
    }
    for k, v in expected.items():
        assert result[k] == v


def test_read_raw_data(tmpdir):
    """Check our utility for reading raw data."""

    from xmitgcm.utils import read_raw_data
    shape = (2, 4)
    for dtype in [np.dtype('f8'), np.dtype('f4'), np.dtype('i4')]:
        # create some test data
        testdata = np.zeros(shape, dtype)
        # write to a file
        datafile = tmpdir.join("tmp.data")
        datafile.write_binary(testdata.tobytes())
        fname = str(datafile)
        # now test the function
        data = read_raw_data(fname, dtype, shape)
        np.testing.assert_allclose(data, testdata)
        # interestingly, memmaps are also ndarrays, but not vice versa
        assert isinstance(data, np.ndarray) and not isinstance(data, np.memmap)
        # check memmap
        mdata = read_raw_data(fname, dtype, shape, use_mmap=True)
        assert isinstance(mdata, np.memmap)

    # make sure errors are correct
    wrongshape = (2, 5)
    with pytest.raises(IOError):
        _ = read_raw_data(fname, dtype, wrongshape)


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


def test_read_mds(all_mds_datadirs):
    """Check that we can read mds data from .meta / .data pairs"""

    dirname, expected = all_mds_datadirs

    from xmitgcm.utils import read_mds

    prefix = 'XC'
    basename = os.path.join(dirname, prefix)
    res = read_mds(basename)
    assert isinstance(res, dict)
    assert prefix in res
    # should be dask by default
    assert isinstance(res[prefix], dask.array.core.Array)

    # try some options
    res = read_mds(basename, force_dict=False, dask_delayed=False)
    assert isinstance(res, np.memmap)
    res = read_mds(basename, force_dict=False, use_mmap=False,
                   dask_delayed=False)
    assert isinstance(res, np.ndarray)

    # make sure endianness works
    testval = res.newbyteorder('<')[0,0]
    res_endian = read_mds(basename, force_dict=False, use_mmap=False,
                          endian='<', dask_delayed=False)
    val_endian = res_endian[0,0]
    np.testing.assert_allclose(testval, val_endian)

    # try reading with iteration number
    prefix = 'T'
    basename = os.path.join(dirname, prefix)
    iternum = expected['test_iternum']
    res = read_mds(basename, iternum=iternum)
    assert prefix in res

def test_read_mds_no_meta(all_mds_datadirs):
    from xmitgcm.utils import read_mds
    dirname, expected = all_mds_datadirs
    shape = expected['shape']
    ny,nx = shape[-2:]
    if len(shape)==4:
        # we have an llc
        nz, nface = shape[:2]
        ny = nx*nface
    else:
        nz = shape[0]
    dtype = expected['dtype']

    shape_2d = (ny, nx)
    shape_3d = shape_2d if nz==1 else (nz,) + shape_2d

    prefixes = {'XC': shape_2d, 'hFacC': shape_3d}

    for prefix, shape in prefixes.items():
        basename = os.path.join(dirname, prefix)
        with hide_file(dirname, prefix + '.meta'):
            # can't read without specifying shape and dtype
            with pytest.raises(IOError) as ioe:
                res = read_mds(basename)
            res = read_mds(basename, shape=shape, dtype=dtype)
            assert isinstance(res, dict)
            assert prefix in res
            # should be dask by default
            assert isinstance(res[prefix], dask.array.core.Array)
            assert res[prefix].shape == shape

@pytest.mark.parametrize("method", ["smallchunks", "bigchunks"])
@pytest.mark.parametrize("memmap", [True, False])
def test_read_raw_data_llc(llc_mds_datadirs, method, memmap):
    if memmap and method=='smallchunks':
        pytest.skip("Using `method='smallchunks` with `memmap=True` "
                    "opens too many files.")

    dirname, expected = llc_mds_datadirs

    from xmitgcm.utils import read_3d_llc_data

    shape = expected['shape']
    nz, nface, ny, nx = shape
    # the function will also return a nrecs dimension
    nrecs = 1
    shape = (nrecs,) + shape

    dtype = expected['dtype'].newbyteorder('>')

    # if we use memmap=True, we open too many files
    kwargs = dict(method=method, dtype=dtype, memmap=memmap)

    fname = os.path.join(dirname, 'T.%010d.data' % expected['test_iternum'])
    data = read_3d_llc_data(fname, nz, nx, **kwargs)
    assert data.shape == shape
    dc = data.compute()
    assert dc.shape == shape
    # once computed, all arrays are ndarray, even if backed by memmap
    assert isinstance(dc, np.ndarray)

    fname = os.path.join(dirname, 'XC.data')
    data = read_3d_llc_data(fname, 1, nx, **kwargs)
    # the z dimension is squeezed out by MDS, so the function matches that behavior
    shape_2d = (shape[0],) + shape[2:]
    assert data.shape == shape_2d
    assert data.compute().shape == shape_2d

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
    if len(eshape)==3:
        nz, ny, nx = eshape
        nface = None
    elif len(eshape)==4:
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

def test_read_grid(all_mds_datadirs):
    """Make sure we read all the grid variables."""
    dirname, expected = all_mds_datadirs
    ds = xmitgcm.open_mdsdataset(
                dirname, iters=None, read_grid=True,
                geometry=expected['geometry'])

    for vname in _EXPECTED_GRID_VARS:
        assert vname in ds

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
    if expected['geometry']=='llc':
        dims_2d = ('face',) + dims_2d
        ny = nx*shape[-3]
    dims_3d = dims_2d if nz==1 else ('k',) + dims_2d
    dims_2d = ('time',) + dims_2d
    dims_3d = ('time',) + dims_3d

    it = expected['test_iternum']
    kwargs = dict(iters=it, geometry=expected['geometry'], read_grid=False,
                  swap_dims=False, default_dtype=expected['dtype'])

    # a 3D file
    to_hide = ['T.%010d.meta' % it, 'Eta.%010d.meta' % it]
    with hide_file(dirname, *to_hide):
        ds = xmitgcm.open_mdsdataset(dirname, prefix=['T', 'Eta'], **kwargs)
        assert ds['T'].dims == dims_3d
        assert ds['T'].values.ndim == len(dims_3d)
        assert ds['Eta'].dims == dims_2d
        assert ds['Eta'].values.ndim == len(dims_2d)

        with pytest.raises(IOError, message="Expecting IOError when default_dtype "
                                            "is not precised (i.e., None)"):
            xmitgcm.open_mdsdataset(dirname, prefix=['T', 'Eta'], iters=it,
                                    geometry=expected['geometry'],
                                    read_grid=False)

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


def test_swap_dims(all_mds_datadirs):
    """See if we can swap dimensions."""

    dirname, expected = all_mds_datadirs
    kwargs = dict(iters=None, read_grid=True, geometry=expected['geometry'])

    expected_dims = ['XC', 'XG', 'YC', 'YG', 'Z', 'Zl', 'Zp1', 'Zu']

    # make sure we never swap if not reading grid
    assert 'i' in xmitgcm.open_mdsdataset(dirname,
        iters=None, read_grid=False, geometry=expected['geometry'])
    if expected['geometry'] in ('llc', 'curvilinear'):
        # make sure swapping is not the default
        assert 'i' in xmitgcm.open_mdsdataset(dirname, **kwargs)
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


def test_parse_diagnostics(all_mds_datadirs):
    """Make sure we can parse the available_diagnostics.log file."""
    from xmitgcm.utils import parse_available_diagnostics
    dirname, expected = all_mds_datadirs
    diagnostics_fname = os.path.join(dirname, 'available_diagnostics.log')
    ad = parse_available_diagnostics(diagnostics_fname)

    # a somewhat random sampling of diagnostics
    expected_diags = {
        'UVEL': {'dims': ['k', 'j', 'i_g'],
                 'attrs': {'units': 'm/s',
                           'long_name': 'Zonal Component of Velocity (m/s)',
                           'standard_name': 'UVEL',
                           'mate': 'VVEL'}},
        'TFLUX': {'dims': ['j', 'i'],
                  'attrs': {'units': 'W/m^2',
                            'long_name': 'total heat flux (match heat-content '
                            'variations), >0 increases theta',
                            'standard_name': 'TFLUX'}}
     }

    for key, val in expected_diags.items():
        assert ad[key] == val


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

@pytest.mark.parametrize("method", ["smallchunks", "bigchunks"])
def test_llc_dims(llc_mds_datadirs, method):
    """Check that the LLC file dimensions are correct."""
    dirname, expected = llc_mds_datadirs
    ds = xmitgcm.open_mdsdataset(dirname,
                            iters=expected['test_iternum'],
                            geometry=expected['geometry'], llc_method=method)

    nz, nface, ny, nx = expected['shape']
    nt = 1

    assert ds.dims['face'] == 13
    assert ds.rA.dims == ('face', 'j', 'i')
    assert ds.rA.values.shape == (nface, ny, nx)
    assert ds.U.dims == ('time', 'k', 'face', 'j', 'i_g')
    assert ds.U.values.shape == (nt, nz, nface, ny, nx)
    assert ds.V.dims == ('time', 'k', 'face', 'j_g', 'i')
    assert ds.V.values.shape == (nt, nz, nface, ny, nx)

def test_drc_length(all_mds_datadirs):
    """Test that open_mdsdataset is adding an extra level to drC if it has length nr"""
    dirname, expected = all_mds_datadirs
    #Only older versions of the gcm have len(drC) = nr, so force len(drC) = nr for the test
    copyfile(os.path.join(dirname, 'DRF.data'), os.path.join(dirname, 'DRC.data'))
    copyfile(os.path.join(dirname, 'DRF.meta'), os.path.join(dirname, 'DRC.meta'))
    ds = xmitgcm.open_mdsdataset(
                dirname, iters=None, read_grid=True,
                geometry=expected['geometry'])
    assert len(ds.drC)==(len(ds.drF)+1)
