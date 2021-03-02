import pytest
import os
import numpy as np
import xarray
import dask
from xmitgcm.test.test_xmitgcm_common import (hide_file, file_md5_checksum,
    all_mds_datadirs, mds_datadirs_with_diagnostics, llc_mds_datadirs,
    layers_mds_datadirs, all_grid_datadirs, mds_datadirs_with_inputfiles,
    _experiments, cs_mds_datadirs)
from xmitgcm.file_utils import listdir


_xc_meta_content = """ simulation = { 'global_oce_latlon' };
 nDims = [   2 ];
 dimList = [
    90,    1,   90,
    40,    1,   40
 ];
 dataprec = [ 'float32' ];
 nrecords = [     1 ];
"""


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


@pytest.mark.parametrize("dtype", [np.dtype('f8'), np.dtype('f4'), np.dtype('i4')])
def test_read_raw_data(tmpdir, dtype):
    """Check our utility for reading raw data."""

    from xmitgcm.utils import read_raw_data
    shape = (2, 4)
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
        read_raw_data(fname, dtype, wrongshape)

    # test optional functionalities
    shape = (5, 15, 10)
    shape_subset = (15, 10)
    testdata = np.zeros(shape, dtype)
    # create some test data
    x = np.arange(shape[0], dtype=dtype)
    for k in np.arange(shape[0]):
        testdata[k, :, :] = x[k]
    # write to a file
    datafile = tmpdir.join("tmp.data")
    datafile.write_binary(testdata.tobytes())
    fname = str(datafile)
    # now test the function
    for k in np.arange(shape[0]):
        offset = (k * shape[1] * shape[2] * dtype.itemsize)
        data = read_raw_data(fname, dtype, shape_subset,
                             offset=offset, partial_read=True)
        np.testing.assert_allclose(data, testdata[k, :, :])
        assert isinstance(data, np.ndarray) and not isinstance(
            data, np.memmap)
        # check memmap
        mdata = read_raw_data(fname, dtype, shape_subset,
                              offset=offset, partial_read=True,
                              use_mmap=True)
        assert isinstance(mdata, np.memmap)

        # test it breaks when it should
        with pytest.raises(IOError):
            # read with wrong shape
            read_raw_data(fname, dtype, shape_subset,
                          offset=0, partial_read=False)
        with pytest.raises(IOError):
            read_raw_data(fname, dtype, shape_subset,
                          offset=0, partial_read=False, use_mmap=True)
        with pytest.raises(ValueError):
            # use offset when trying to read global file
            read_raw_data(fname, dtype, shape_subset,
                          offset=4, partial_read=False)
        with pytest.raises(ValueError):
            read_raw_data(fname, dtype, shape_subset,
                          offset=4, partial_read=False, use_mmap=True)
            # offset is too big
        with pytest.raises(ValueError):
            read_raw_data(fname, dtype, shape, offset=(
                shape[0]*shape[1]*shape[2]*dtype.itemsize), partial_read=True)
        with pytest.raises(ValueError):
            read_raw_data(fname, dtype, shape, offset=(
                shape[0]*shape[1]*shape[2]*dtype.itemsize), partial_read=True,
                use_mmap=True)

# a meta test of our own utitity funcion
def test_file_hiding(all_mds_datadirs):
    dirname, _ = all_mds_datadirs
    basenames = ['XC.data', 'XC.meta']
    listed_files = listdir(dirname)
    for basename in basenames:
        assert os.path.exists(os.path.join(dirname, basename))
        assert basename in listed_files
    with hide_file(dirname, *basenames):
        listed_files = listdir(dirname)
        for basename in basenames:
            assert not os.path.exists(os.path.join(dirname, basename))
            assert basename not in listed_files
    for basename in basenames:
        assert os.path.exists(os.path.join(dirname, basename))


def test_read_mds(all_mds_datadirs):
    """Check that we can read mds data from .meta / .data pairs"""

    dirname, expected = all_mds_datadirs

    from xmitgcm.utils import read_mds

    prefix = 'XC'
    basename = os.path.join(dirname, prefix)
    # should be dask by default
    res = read_mds(basename)
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], dask.array.core.Array)

    # try some options
    res = read_mds(basename, use_dask=False)
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], np.memmap)

    res = read_mds(basename, use_dask=False, use_mmap=False)
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], np.ndarray)

    res = read_mds(basename, chunks="2D")
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], dask.array.core.Array)

    res = read_mds(basename, chunks="2D", use_dask=False)
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], np.memmap)

    res = read_mds(basename, chunks="2D", use_dask=False,
                   use_mmap=False)
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], np.ndarray)

    # test the extra_metadata
    if expected['geometry'] == 'llc':
        emeta = {'has_faces': True, 'ny': 13*90, 'nx': 90,
                 'ny_facets': [3*90, 3*90, 90, 3*90, 3*90],
                 'face_facets': [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4],
                 'facet_orders': ['C', 'C', 'C', 'F', 'F'],
                 'face_offsets': [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2],
                 'transpose_face': [False, False, False,
                                    False, False, False, False,
                                    True, True, True, True, True, True]}
    else:
        emeta = None
    res = read_mds(basename, chunks="2D", use_dask=False,
                   use_mmap=False, extra_metadata=emeta)
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], np.ndarray)

    # make sure endianness works
    res = read_mds(basename, use_dask=False, use_mmap=False)
    testval = res[prefix].newbyteorder('<')[0, 0]
    res_endian = read_mds(basename, use_mmap=False,
                          endian='<', use_dask=False)
    val_endian = res_endian[prefix][0, 0]
    np.testing.assert_allclose(testval, val_endian)

    # try reading with iteration number
    prefix = 'T'
    basename = os.path.join(dirname, prefix)
    iternum = expected['test_iternum']
    res = read_mds(basename, iternum=iternum)
    assert prefix in res
    assert isinstance(res[prefix], dask.array.core.Array)

    # try some options
    res = read_mds(basename, iternum=iternum, use_dask=False)
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], np.memmap)

    res = read_mds(basename, iternum=iternum, use_dask=False,
                   use_mmap=False)
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], np.ndarray)

    res = read_mds(basename, iternum=iternum, chunks="2D")
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], dask.array.core.Array)

    res = read_mds(basename, iternum=iternum, chunks="2D",
                   use_dask=False)
    assert isinstance(res, dict)
    assert prefix in res
    print(type(res[prefix]))
    assert isinstance(res[prefix], np.ndarray)  # should be memmap

    res = read_mds(basename, iternum=iternum, chunks="2D",
                   use_dask=False, use_mmap=False)
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], np.ndarray)

    # check fails with bad nx, ny
    if expected['geometry'] == 'llc':
        emeta.update({'ny': 13*270, 'nx': 270})
        with pytest.raises(AssertionError):
            res = read_mds(basename, iternum=iternum,
                           chunks="2D",
                           llc=True,
                           use_dask=False,
                           use_mmap=False, extra_metadata=emeta)


def test_read_mds_tokens(mds_datadirs_with_diagnostics):
    from xmitgcm.utils import read_mds
    dirname, expected = mds_datadirs_with_diagnostics
    diagnostics = expected['diagnostics']
    prefix = diagnostics[0]
    basename = os.path.join(dirname, prefix)
    iternum = expected['test_iternum']
    data = read_mds(basename, iternum=iternum)
    dask_keys = set()
    for varname, da in data.items():
        keys = list(da.dask.keys())
        for k in keys:
            token = k[0]
            if 'mds' in token:
                dask_keys.add(token)
    assert len(dask_keys) == len(data)


def test_read_mds_no_meta(all_mds_datadirs):
    from xmitgcm.utils import read_mds
    dirname, expected = all_mds_datadirs
    shape = expected['shape']
    ny, nx = shape[-2:]
    if len(shape) == 4:
        # we have an llc
        nz, nface = shape[:2]
        ny = nx*nface
    else:
        nz = shape[0]
    dtype = expected['dtype']

    shape_2d = (ny, nx)
    shape_3d = shape_2d if nz == 1 else (nz,) + shape_2d

    prefixes = {'XC': shape_2d, 'hFacC': shape_3d}

    for prefix, shape in prefixes.items():
        basename = os.path.join(dirname, prefix)
        with hide_file(dirname, prefix + '.meta'):
            # can't read without specifying shape and dtype
            with pytest.raises(IOError):
                res = read_mds(basename)
            res = read_mds(basename, shape=shape, dtype=dtype, legacy=True)
            assert isinstance(res, dict)
            assert prefix in res
            # should be dask by default
            assert isinstance(res[prefix], dask.array.core.Array)
            assert res[prefix].shape == shape

            res = read_mds(basename, shape=shape, dtype=dtype, legacy=False)
            assert isinstance(res, dict)
            assert prefix in res
            # should be dask by default
            assert isinstance(res[prefix], dask.array.core.Array)
            assert res[prefix].shape == (1,) + shape


@pytest.mark.parametrize("method", ["smallchunks", "bigchunks"])
@pytest.mark.parametrize("memmap", [True, False])
def test_read_raw_data_llc(llc_mds_datadirs, method, memmap):
    if memmap and method == 'smallchunks':
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
    # the z dimension is squeezed out by MDS,
    # so the function matches that behavior
    shape_2d = (shape[0],) + shape[2:]
    assert data.shape == shape_2d
    assert data.compute().shape == shape_2d


@pytest.mark.parametrize("memmap", [True, False])
def test_read_xyz_chunk(all_mds_datadirs, memmap):

    from xmitgcm.utils import _read_xyz_chunk

    dirname, expected = all_mds_datadirs

    file_metadata = expected
    file_metadata.update({'filename': dirname + '/' + 'T.' +
                          str(file_metadata['test_iternum']).zfill(10) +
                          '.data', 'vars': ['T'], 'endian': '>'})
    # set the size of dimensions (could be changed in _experiments)
    if file_metadata['geometry'] in ['llc']:
        file_metadata.update({'nx': file_metadata['shape'][3],
                              'ny': file_metadata['shape'][2],
                              'nface': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'has_faces': True})
        # function not designed for llc grids, except 1d variables
        with pytest.raises(ValueError):
            data = _read_xyz_chunk('T', file_metadata, use_mmap=memmap)
    elif file_metadata['geometry'] in ['cs']:
        file_metadata.update({'nx': file_metadata['shape'][3],
                              'ny': file_metadata['shape'][1] *
                              file_metadata['shape'][2],
                              'nface': file_metadata['shape'][2],
                              'nz': file_metadata['shape'][0],
                              'has_faces': True})
        # function not designed for llc grids, except 1d variables
        with pytest.raises(ValueError):
            data = _read_xyz_chunk('T', file_metadata, use_mmap=memmap)
    else:
        file_metadata.update({'nx': file_metadata['shape'][2],
                              'ny': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'dims_vars': [('nz', 'ny', 'nx')],
                              'has_faces': False})

        data = _read_xyz_chunk('T', file_metadata, use_mmap=memmap)

        if memmap:
            assert isinstance(data, np.memmap)
        else:
            assert isinstance(data, np.ndarray)

        # test it fails for too large number of records
        with pytest.raises(ValueError):
            data = _read_xyz_chunk('T', file_metadata, rec=1, use_mmap=memmap)

    # test 1d variable
    file_metadata.update({'filename': dirname + '/' + 'RC' + '.data',
                          'vars': ['RC'], 'nx': 1, 'ny': 1,
                          'dims_vars': [('nz', 'ny', 'nx')]})

    data = _read_xyz_chunk('RC', file_metadata, use_mmap=memmap)
    if memmap:
        assert isinstance(data, np.memmap)
    else:
        assert isinstance(data, np.ndarray)


@pytest.mark.parametrize("memmap", [True, False])
def test_read_xy_chunk(all_mds_datadirs, memmap):

    from xmitgcm.utils import _read_xy_chunk

    dirname, expected = all_mds_datadirs

    file_metadata = expected
    file_metadata.update({'filename': dirname + '/' + 'T.' +
                          str(file_metadata['test_iternum']).zfill(10) +
                          '.data', 'vars': ['T'], 'endian': '>'})
    # set the size of dimensions (could be changed in _experiments)
    if file_metadata['geometry'] in ['llc']:
        nx = file_metadata['shape'][3]
        file_metadata.update({'nx': file_metadata['shape'][3],
                              'ny': file_metadata['shape'][2],
                              'nface': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'dims_vars': [('nz', 'nface', 'ny', 'nx')],
                              'has_faces': True,
                              'ny_facets': [3*nx, 3*nx, nx, 3*nx, 3*nx],
                              'face_facets':
                              [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4],
                              'facet_orders': ['C', 'C', 'C', 'F', 'F'],
                              'face_offsets':
                              [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2],
                              'transpose_face': [False, False, False, False,
                                                 False, False, False, True,
                                                 True, True, True, True,
                                                 True]})
    else:
        file_metadata.update({'nx': file_metadata['shape'][2],
                              'ny': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'dims_vars': [('nz', 'ny', 'nx')],
                              'has_faces': False})

    data = _read_xy_chunk('T', file_metadata, use_mmap=memmap)

    if memmap:
        assert isinstance(data, np.memmap)
    else:
        assert isinstance(data, np.ndarray)

    # test it fails for too large number of records
    with pytest.raises(ValueError):
        data = _read_xy_chunk('T', file_metadata, rec=1, use_mmap=memmap)
    # test it fails for too large number of levels
    with pytest.raises(ValueError):
        data = _read_xy_chunk('T', file_metadata, lev=9999, use_mmap=memmap)

    # those tests are only available for llc experiment:
    # test reading in multi-variable files
    if expected['geometry'] not in ['llc']:
        pass
    else:
        dimsvar = []
        for kk in np.arange(25):
            dimsvar.append(('ny', 'nx'))
        file_metadata.update({'filename': dirname + '/' +
                              'state_2d_set1.0000000008.data',
                              'vars': expected['diagnostics'][1],
                              'dims_vars': dimsvar})

        for kface in np.arange(13):
            data = _read_xy_chunk('MXLDEPTH', file_metadata, face=kface,
                                  use_mmap=memmap)

        if memmap:
            assert isinstance(data, np.memmap)
        else:
            assert isinstance(data, np.ndarray)


@pytest.mark.parametrize("memmap", [True, False])
@pytest.mark.parametrize("usedask", [True, False])
def test_read_2D_chunks(all_mds_datadirs, memmap, usedask):

    from xmitgcm.utils import read_2D_chunks

    dirname, expected = all_mds_datadirs

    file_metadata = expected
    file_metadata.update({'filename': dirname + '/' + 'T.' +
                          str(file_metadata['test_iternum']).zfill(10) +
                          '.data', 'vars': ['T'], 'endian': '>'})
    # set the size of dimensions (could be changed in _experiments)
    if file_metadata['geometry'] in ['llc']:
        nx = file_metadata['shape'][3]
        file_metadata.update({'nx': file_metadata['shape'][3],
                              'ny': file_metadata['shape'][2],
                              'nface': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'nt': 1,
                              'dims_vars': [('nz', 'nface', 'ny', 'nx')],
                              'has_faces': True,
                              'ny_facets': [3*nx, 3*nx, nx, 3*nx, 3*nx],
                              'face_facets':
                              [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4],
                              'facet_orders': ['C', 'C', 'C', 'F', 'F'],
                              'face_offsets':
                              [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2],
                              'transpose_face': [False, False, False, False,
                                                 False, False, False, True,
                                                 True, True, True, True,
                                                 True]})
    else:
        file_metadata.update({'nx': file_metadata['shape'][2],
                              'ny': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'nt': 1,
                              'dims_vars': [('nz', 'ny', 'nx')],
                              'has_faces': False})

    data = read_2D_chunks('T', file_metadata, use_mmap=memmap,
                          use_dask=usedask)
    if usedask:
        assert isinstance(data, dask.array.core.Array)
        data.compute()  # check accessing data works
    else:
        if memmap:
            assert isinstance(data, np.ndarray)  # should be memmap, need fix
        else:
            assert isinstance(data, np.ndarray)

    # test 1d variable
    file_metadata.update({'filename': dirname + '/' + 'RC' + '.data',
                          'vars': ['RC'], 'nx': 1, 'ny': 1})

    data = read_2D_chunks('RC', file_metadata, use_mmap=memmap,
                          use_dask=usedask)
    if usedask:
        assert isinstance(data, dask.array.core.Array)
        data.compute()
    else:
        if memmap:
            assert isinstance(data, np.ndarray)  # reshape triggers evaluation
        else:
            assert isinstance(data, np.ndarray)


@pytest.mark.parametrize("memmap", [True, False])
@pytest.mark.parametrize("usedask", [True, False])
def test_read_3D_chunks(all_mds_datadirs, memmap, usedask):

    from xmitgcm.utils import read_3D_chunks

    dirname, expected = all_mds_datadirs

    file_metadata = expected
    file_metadata.update({'filename': dirname + '/' + 'T.' +
                          str(file_metadata['test_iternum']).zfill(10) +
                          '.data', 'vars': ['T'], 'endian': '>'})
    # set the size of dimensions (could be changed in _experiments)
    if file_metadata['geometry'] in ['llc']:
        nx = file_metadata['shape'][3]
        file_metadata.update({'nx': file_metadata['shape'][3],
                              'ny': file_metadata['shape'][2],
                              'nface': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'nt': 1,
                              'dims_vars': [('nz', 'nface', 'ny', 'nx')],
                              'has_faces': True,
                              'ny_facets': [3*nx, 3*nx, nx, 3*nx, 3*nx],
                              'face_facets':
                              [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4],
                              'facet_orders': ['C', 'C', 'C', 'F', 'F'],
                              'face_offsets':
                              [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2],
                              'transpose_face': [False, False, False, False,
                                                 False, False, False, True,
                                                 True, True, True, True,
                                                 True]})
    elif file_metadata['geometry'] in ['cs']:
        file_metadata.update({'nx': file_metadata['shape'][3],
                              'ny': file_metadata['shape'][1] *
                              file_metadata['shape'][2],
                              'nface': file_metadata['shape'][2],
                              'nz': file_metadata['shape'][0],
                              'nt': 1,
                              'dims_vars': [('nz', 'nface', 'ny', 'nx')],
                              'has_faces': True})
    else:
        file_metadata.update({'nx': file_metadata['shape'][2],
                              'ny': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'nt': 1,
                              'dims_vars': [('nz', 'ny', 'nx')],
                              'has_faces': False})

    if file_metadata['geometry'] in ['llc', 'cs']:
        with pytest.raises(ValueError):
            data = read_3D_chunks('T', file_metadata, use_mmap=memmap,
                                  use_dask=usedask)
            if usedask:
                data.compute()
    else:
        data = read_3D_chunks('T', file_metadata, use_mmap=memmap,
                              use_dask=usedask)
        if usedask:
            assert isinstance(data, dask.array.core.Array)
            data.compute()
        else:
            if memmap:
                assert isinstance(data, np.memmap)
            else:
                assert isinstance(data, np.ndarray)

    # test 1d variable
    file_metadata.update({'filename': dirname + '/' + 'RC' + '.data',
                          'vars': ['RC'], 'nx': 1, 'ny': 1})

    data = read_3D_chunks('RC', file_metadata, use_mmap=memmap,
                          use_dask=usedask)
    if usedask:
        assert isinstance(data, dask.array.core.Array)
        data.compute()
    else:
        if memmap:
            assert isinstance(data, np.memmap)
        else:
            assert isinstance(data, np.ndarray)


@pytest.mark.parametrize("memmap", [True, False])
@pytest.mark.parametrize("usedask", [True, False])
def test_read_CS_chunks(cs_mds_datadirs, memmap, usedask):

    from xmitgcm.utils import read_CS_chunks

    dirname, expected = cs_mds_datadirs
    print(expected)
    nz, ny, nfaces, nx = expected['shape']
    # 3D array, single variable in file
    file_metadata = {}
    file_metadata['nx'] = nx
    file_metadata['ny'] = ny
    file_metadata['ny_facets'] = [ny, ny, ny, ny, ny, ny]
    file_metadata['nz'] = nz
    file_metadata['nt'] = 1
    file_metadata['dtype'] = np.dtype('>f4')
    file_metadata['test_iternum'] = expected['test_iternum']
    file_metadata.update({'filename': dirname + '/' + 'T.' +
                          str(file_metadata['test_iternum']).zfill(10) +
                          '.data', 'vars': ['T'], 'endian': '>'})
    data = read_CS_chunks('T', file_metadata, use_mmap=memmap, use_dask=usedask)
    if usedask:
        assert isinstance(data, dask.array.core.Array)
    else:
        assert isinstance(data, np.ndarray)
    assert data.shape == (1,) + expected['shape']
    # 1D array, single variable in file
    file_metadata['nx'] = 1
    file_metadata['ny'] = 1
    file_metadata['nz'] = nz
    file_metadata.update({'filename': dirname + '/' + 'RC.data',
                          'vars': ['RC'], 'endian': '>'})
    data = read_CS_chunks('RC', file_metadata, use_mmap=memmap, use_dask=usedask)
    if usedask:
        assert isinstance(data, dask.array.core.Array)
    else:
        assert isinstance(data, np.ndarray)
    assert data.shape == (1, nz, 1, 1, 1)


@pytest.mark.parametrize("memmap", [True, False])
@pytest.mark.parametrize("usedask", [True, False])
def test_read_all_variables(all_mds_datadirs, memmap, usedask):

    from xmitgcm.utils import read_all_variables

    dirname, expected = all_mds_datadirs

    file_metadata = expected
    # test single variable in file
    file_metadata.update({'filename': dirname + '/' + 'T.' +
                          str(file_metadata['test_iternum']).zfill(10) +
                          '.data', 'vars': ['T'], 'endian': '>'})
    # set the size of dimensions (could be changed in _experiments)
    if file_metadata['geometry'] in ['llc']:
        nx = file_metadata['shape'][3]
        file_metadata.update({'nx': file_metadata['shape'][3],
                              'ny': file_metadata['shape'][2],
                              'nface': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'nt': 1,
                              'dims_vars': [('nz', 'nface', 'ny', 'nx')],
                              'has_faces': True,
                              'ny_facets': [3*nx, 3*nx, nx, 3*nx, 3*nx],
                              'face_facets':
                              [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4],
                              'facet_orders': ['C', 'C', 'C', 'F', 'F'],
                              'face_offsets':
                              [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2],
                              'transpose_face': [False, False, False, False,
                                                 False, False, False, True,
                                                 True, True, True, True,
                                                 True]})
    elif file_metadata['geometry'] in ['cs']:
        nx = file_metadata['shape'][3]
        file_metadata.update({'nx': file_metadata['shape'][3],
                              'ny': file_metadata['shape'][1],
                              'nface': file_metadata['shape'][2],
                              'nz': file_metadata['shape'][0],
                              'nt': 1,
                              'dims_vars': [('nz', 'nface', 'ny', 'nx')],
                              'has_faces': True,
                              'ny_facets': [nx, nx, nx, nx, nx, nx],
                              'face_facets':
                              [0, 1, 2, 3, 4, 5],
                              'facet_orders': ['F', 'F', 'F', 'F', 'F', 'F'],
                              'face_offsets':
                              [0, 0, 0, 0, 0, 0],
                              'transpose_face': [False, False, False, False,
                                                 False, False]})
    else:
        file_metadata.update({'nx': file_metadata['shape'][2],
                              'ny': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'nt': 1,
                              'dims_vars': [('nz', 'ny', 'nx')],
                              'has_faces': False})

    # test 3D chunks, fails on llc but not others
    if file_metadata['geometry'] in ['llc', 'cs']:
        with pytest.raises(ValueError):
            dataset = read_all_variables(file_metadata['vars'], file_metadata,
                                         use_mmap=memmap, use_dask=usedask,
                                         chunks="3D")
            if usedask:
                dataset[0].compute()
    else:
        dataset = read_all_variables(file_metadata['vars'], file_metadata,
                                     use_mmap=memmap, use_dask=usedask,
                                     chunks="3D")

        assert isinstance(dataset, list)
        assert len(dataset) == len(file_metadata['vars'])
        if usedask:
            assert isinstance(dataset[0], dask.array.core.Array)
        else:
            if memmap:
                assert isinstance(dataset[0], np.memmap)
            else:
                assert isinstance(dataset[0], np.ndarray)

    # test 2D chunks
    if file_metadata['geometry'] not in ['cs']:
        dataset = read_all_variables(file_metadata['vars'], file_metadata,
                                     use_mmap=memmap, use_dask=usedask,
                                     chunks="2D")
    else:
        dataset = read_all_variables(file_metadata['vars'], file_metadata,
                                     use_mmap=memmap, use_dask=usedask,
                                     chunks="CS")

    assert isinstance(dataset, list)
    assert len(dataset) == len(file_metadata['vars'])
    if usedask:
        assert isinstance(dataset[0], dask.array.core.Array)
    else:
        if memmap:
            # should be memmap, needs fix
            assert isinstance(dataset[0], np.ndarray)
        else:
            assert isinstance(dataset[0], np.ndarray)

    # test multiple variables in file
    # those tests are only available for llc experiment:
    # test reading in multi-variable files
    if expected['geometry'] not in ['llc']:
        pass
    else:
        dimsvar = []
        for kk in np.arange(25):
            dimsvar.append(('ny', 'nx'))
        file_metadata.update({'filename': dirname + '/' +
                              'state_2d_set1.0000000008.data',
                              'vars': expected['diagnostics'][1],
                              'dims_vars': dimsvar})

    dataset = read_all_variables(file_metadata['vars'], file_metadata,
                                 use_mmap=memmap, use_dask=usedask,
                                 chunks="2D")

    assert isinstance(dataset, list)
    assert len(dataset) == len(file_metadata['vars'])
    if usedask:
        assert isinstance(dataset[0], dask.array.core.Array)
    else:
        if memmap:
            # should be memmap, needs fix
            assert isinstance(dataset[0], np.ndarray)
        else:
            assert isinstance(dataset[0], np.ndarray)


@pytest.mark.parametrize("dtype", ['>d', '>f', '>i'])
@pytest.mark.parametrize("memmap", [True, False])
def test_pad_array(tmpdir, memmap, dtype):

    from xmitgcm.utils import _pad_array
    import struct

    # create test data
    gendata = np.array([[1, 2], [3, 4]])

    # write to a file
    datafile = tmpdir.join("testdata")
    fname = str(datafile)
    fid = open(fname, "wb")
    flatdata = gendata.flatten()
    for kk in np.arange(len(flatdata)):
        tmp = struct.pack(dtype, flatdata[kk])
        fid.write(tmp)
    fid.close()

    # then read it
    if memmap:
        data = np.memmap(fname, dtype=dtype, mode='r',
                         shape=(2, 2,), order='C')
    else:
        data = np.fromfile(fname, dtype=dtype)
        data = data.reshape((2, 2,))

    # check my original data
    ny, nx = data.shape
    assert data.shape == (ny, nx)
    assert data.min() == 1
    assert data.max() == 4

    # test no padding
    file_metadata = {}
    data_padded = _pad_array(data, file_metadata)
    assert data_padded.shape == (2, 2)
    if memmap:
        assert isinstance(data_padded, np.memmap)
    else:
        assert isinstance(data_padded, np.ndarray)

    # test padding before
    file_metadata = {'pad_before_y': 2, 'has_faces': False, 'nx': nx}
    data_padded = _pad_array(data, file_metadata)
    assert isinstance(data_padded, np.ndarray)
    assert data_padded.shape == (4, 2)
    assert data_padded.min() == 0
    assert data_padded.max() == 4
    assert data_padded[2, 0] == 1
    assert data_padded[3, 1] == 4

    file_metadata = {'pad_before_y': [2, 3], 'has_faces': True, 'nx': nx,
                     'face_facets': [0, 1]}
    data_padded = _pad_array(data, file_metadata, face=0)
    assert isinstance(data_padded, np.ndarray)
    assert data_padded.shape == (4, 2)
    assert data_padded.min() == 0
    assert data_padded.max() == 4
    assert data_padded[2, 0] == 1
    assert data_padded[3, 1] == 4

    data_padded = _pad_array(data, file_metadata, face=1)
    assert isinstance(data_padded, np.ndarray)
    assert data_padded.shape == (5, 2)
    assert data_padded.min() == 0
    assert data_padded.max() == 4
    assert data_padded[3, 0] == 1
    assert data_padded[4, 1] == 4

    # test padding after
    file_metadata = {'pad_after_y': 2, 'has_faces': False, 'nx': nx}
    data_padded = _pad_array(data, file_metadata)
    assert isinstance(data_padded, np.ndarray)
    assert data_padded.shape == (4, 2)
    assert data_padded.min() == 0
    assert data_padded.max() == 4
    assert data_padded[0, 0] == 1
    assert data_padded[1, 1] == 4

    file_metadata = {'pad_after_y': [2, 3], 'has_faces': True, 'nx': nx,
                     'face_facets': [0, 1]}
    data_padded = _pad_array(data, file_metadata, face=0)
    assert isinstance(data_padded, np.ndarray)
    assert data_padded.shape == (4, 2)
    assert data_padded.min() == 0
    assert data_padded.max() == 4
    assert data_padded[0, 0] == 1
    assert data_padded[1, 1] == 4

    data_padded = _pad_array(data, file_metadata, face=1)
    assert isinstance(data_padded, np.ndarray)
    assert data_padded.shape == (5, 2)
    assert data_padded.min() == 0
    assert data_padded.max() == 4
    assert data_padded[0, 0] == 1
    assert data_padded[1, 1] == 4


def test_parse_diagnostics(all_mds_datadirs, layers_mds_datadirs):
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

    # test layers
    dirname, expected = layers_mds_datadirs
    diagnostics_fname = os.path.join(dirname, 'available_diagnostics.log')
    ad = parse_available_diagnostics(diagnostics_fname)

    expected_diags = {
        'LaUH1RHO': {'dims': ['_UNKNOWN_', 'j', 'i_g'],
                     'attrs': {'units': 'm.m/s',
                               'long_name': 'Layer Integrated  zonal Transport (UH, m^2/s)',
                               'standard_name': 'LaUH1RHO',
                               'mate': 'LaVH1RHO'}},
    }

    for key, val in expected_diags.items():
        assert ad[key] == val


@pytest.mark.parametrize("domain", ['llc', 'aste', 'cs'])
@pytest.mark.parametrize("nx", [90, 270])
def test_get_extra_metadata(domain, nx):
    from xmitgcm.utils import get_extra_metadata
    em = get_extra_metadata(domain=domain, nx=nx)
    assert type(em) == dict

    with pytest.raises(ValueError):
        em = get_extra_metadata(domain='notinlist', nx=nx)


@pytest.mark.parametrize("usedask", [True, False])
def test_get_grid_from_input(all_grid_datadirs, usedask):
    from xmitgcm.utils import get_grid_from_input, get_extra_metadata
    from xmitgcm.utils import read_raw_data
    dirname, expected = all_grid_datadirs
    md = get_extra_metadata(domain=expected['domain'], nx=expected['nx'])
    ds = get_grid_from_input(dirname + '/' + expected['gridfile'],
                             geometry=expected['geometry'],
                             dtype=np.dtype('d'), endian='>',
                             use_dask=usedask,
                             extra_metadata=md)
    # test types
    assert type(ds) == xarray.Dataset
    assert type(ds['XC']) == xarray.core.dataarray.DataArray

    if usedask:
        ds.load()

    # check all variables are in
    expected_variables = ['XC', 'YC', 'DXF', 'DYF', 'RAC',
                          'XG', 'YG', 'DXV', 'DYU', 'RAZ',
                          'DXC', 'DYC', 'RAW', 'RAS', 'DXG', 'DYG']

    for var in expected_variables:
        assert type(ds[var]) == xarray.core.dataarray.DataArray
        assert ds[var].values.shape == expected['shape']

    # check we don't leave points behind
    if expected['geometry'] == 'llc':
        nx = expected['nx'] + 1
        nvars = len(expected_variables)
        sizeofd = 8

        grid = expected['gridfile']
        grid1 = dirname + '/' + grid.replace('<NFACET>', '001')
        grid2 = dirname + '/' + grid.replace('<NFACET>', '002')
        grid3 = dirname + '/' + grid.replace('<NFACET>', '003')
        grid4 = dirname + '/' + grid.replace('<NFACET>', '004')
        grid5 = dirname + '/' + grid.replace('<NFACET>', '005')

        size1 = os.path.getsize(grid1)
        size2 = os.path.getsize(grid2)
        size3 = os.path.getsize(grid3)
        size4 = os.path.getsize(grid4)
        size5 = os.path.getsize(grid5)

        ny1 = int(size1 / sizeofd / nvars / nx)
        ny2 = int(size2 / sizeofd / nvars / nx)
        ny3 = int(size3 / sizeofd / nvars / nx)
        ny4 = int(size4 / sizeofd / nvars / nx)
        ny5 = int(size5 / sizeofd / nvars / nx)

        xc1 = read_raw_data(grid1, dtype=np.dtype('>d'), shape=(ny1, nx),
                            partial_read=True)
        xc2 = read_raw_data(grid2, dtype=np.dtype('>d'), shape=(ny2, nx),
                            partial_read=True)
        xc3 = read_raw_data(grid3, dtype=np.dtype('>d'), shape=(ny3, nx),
                            partial_read=True)
        xc4 = read_raw_data(grid4, dtype=np.dtype('>d'), shape=(ny4, nx),
                            order='F', partial_read=True)
        xc5 = read_raw_data(grid5, dtype=np.dtype('>d'), shape=(ny5, nx),
                            order='F', partial_read=True)

        yc1 = read_raw_data(grid1, dtype=np.dtype('>d'), shape=(ny1, nx),
                            partial_read=True, offset=nx*ny1*sizeofd)
        yc2 = read_raw_data(grid2, dtype=np.dtype('>d'), shape=(ny2, nx),
                            partial_read=True, offset=nx*ny2*sizeofd)
        yc3 = read_raw_data(grid3, dtype=np.dtype('>d'), shape=(ny3, nx),
                            partial_read=True, offset=nx*ny3*sizeofd)
        yc4 = read_raw_data(grid4, dtype=np.dtype('>d'), shape=(ny4, nx),
                            order='F', partial_read=True,
                            offset=nx*ny4*sizeofd)
        yc5 = read_raw_data(grid5, dtype=np.dtype('>d'), shape=(ny5, nx),
                            order='F', partial_read=True,
                            offset=nx*ny5*sizeofd)

        xc = np.concatenate([xc1[:-1, :-1].flatten(), xc2[:-1, :-1].flatten(),
                             xc3[:-1, :-1].flatten(), xc4[:-1, :-1].flatten(),
                             xc5[:-1, :-1].flatten()])

        yc = np.concatenate([yc1[:-1, :-1].flatten(), yc2[:-1, :-1].flatten(),
                             yc3[:-1, :-1].flatten(), yc4[:-1, :-1].flatten(),
                             yc5[:-1, :-1].flatten()])

        xc_from_ds = ds['XC'].values.flatten()
        yc_from_ds = ds['YC'].values.flatten()

        assert xc.min() == xc_from_ds.min()
        assert xc.max() == xc_from_ds.max()
        assert yc.min() == yc_from_ds.min()
        assert yc.max() == yc_from_ds.max()

    # passing llc without metadata should fail
    if expected['geometry'] == 'llc':
        with pytest.raises(ValueError):
            ds = get_grid_from_input(dirname + '/' + expected['gridfile'],
                                     geometry=expected['geometry'],
                                     dtype=np.dtype('d'), endian='>',
                                     use_dask=False,
                                     extra_metadata=None)


@pytest.mark.parametrize("dtype", [np.dtype('d'), np.dtype('f')])
def test_write_to_binary(dtype):
    from xmitgcm.utils import write_to_binary
    import sys

    data = np.arange(2)
    # write
    write_to_binary(data, 'tmp.bin', dtype=dtype)
    # read
    if dtype == np.dtype('f'):
        tmp = np.fromfile('tmp.bin', '>f')
    elif dtype == np.dtype('d'):
        tmp = np.fromfile('tmp.bin', '>d')
    # check
    assert len(data) == len(tmp)
    assert data[0] == tmp[0]
    assert data[1] == tmp[1]
    os.remove('tmp.bin')


@pytest.mark.parametrize("possible_concat_dims", [['i', 'i_g'], ['j', 'j_g']])
def test_find_concat_dim(possible_concat_dims):
    from xmitgcm.utils import find_concat_dim

    # this array contains a concat dim
    a = xarray.DataArray(np.empty((2, 3, 4)), dims=['k', 'j', 'i'])
    out = find_concat_dim(a, possible_concat_dims)
    assert out in possible_concat_dims

    b = xarray.DataArray(np.empty((2, 3, 4)), dims=['k', 'g', 'b'])
    out = find_concat_dim(b, possible_concat_dims)
    assert out is None


@pytest.mark.parametrize("domain", ['aste', 'llc'])
@pytest.mark.parametrize("nx", [90, 270])
def test_find_concat_dim_facet(domain, nx):
    from xmitgcm.utils import find_concat_dim_facet, get_extra_metadata
    md = get_extra_metadata(domain=domain, nx=nx)
    nfacets = len(md['ny_facets'])

    for facet in range(5):
        da = xarray.DataArray(np.empty((nfacets, md['ny_facets'][facet], nx)),
                              dims=['face', 'j', 'i'])
        concat_dim, non_concat_dim = find_concat_dim_facet(da, facet, md)

        print(concat_dim, non_concat_dim)
        if md['facet_orders'][facet] == 'C':
            assert concat_dim == 'j'
            assert non_concat_dim == 'i'
        elif md['facet_orders'][facet] == 'F':
            assert concat_dim == 'i'
            assert non_concat_dim == 'j'


@pytest.mark.parametrize("domain", ['aste', 'llc'])
@pytest.mark.parametrize("nx", [90, 270])
def test_rebuild_llc_facets(domain, nx):
    from xmitgcm.utils import rebuild_llc_facets, get_extra_metadata

    md = get_extra_metadata(domain=domain, nx=nx)
    nfaces = len(md['transpose_face'])

    da = xarray.DataArray(np.empty((nfaces, nx, nx)),
                          dims=['face', 'j', 'i'])

    facets = rebuild_llc_facets(da, md)

    for facet in range(5):
        # test we get the original size
        if md['facet_orders'][facet] == 'C':
            expected_shape = (md['ny_facets'][facet], nx,)
        elif md['facet_orders'][facet] == 'F':
            expected_shape = (nx, md['ny_facets'][facet], )
        if domain == 'aste' and facet == 1:  # this facet is empty
            pass
        else:
            assert facets['facet' + str(facet)].shape == expected_shape


def test_llc_facets_2d_to_compact(llc_mds_datadirs):
    from xmitgcm.utils import llc_facets_2d_to_compact, get_extra_metadata
    from xmitgcm.utils import rebuild_llc_facets, read_raw_data
    from xmitgcm.utils import write_to_binary
    from xmitgcm import open_mdsdataset

    dirname, expected = llc_mds_datadirs

    # open dataset
    ds = open_mdsdataset(dirname,
                         iters=expected['test_iternum'],
                         geometry=expected['geometry'])

    nt, nfaces, ny, nx = expected['shape']
    md = get_extra_metadata(domain=expected['geometry'], nx=nx)
    # split in facets
    facets = rebuild_llc_facets(ds['XC'], md)
    flatdata = llc_facets_2d_to_compact(facets, md)
    # compare with raw data
    raw = read_raw_data(dirname + '/XC.data', np.dtype('>f'), (nfaces, ny, nx))
    flatraw = raw.flatten()

    assert len(flatdata) == len(flatraw)
    assert flatdata.min() == flatraw.min()
    assert flatdata.max() == flatraw.max()

    # write new file
    write_to_binary(flatdata, 'tmp.bin', dtype=np.dtype('f'))
    md5new = file_md5_checksum('tmp.bin')
    md5old = file_md5_checksum(dirname + '/XC.data')
    assert md5new == md5old
    os.remove('tmp.bin')


def test_llc_facets_3d_spatial_to_compact(llc_mds_datadirs):
    from xmitgcm.utils import llc_facets_3d_spatial_to_compact
    from xmitgcm.utils import get_extra_metadata
    from xmitgcm.utils import rebuild_llc_facets, read_raw_data
    from xmitgcm.utils import write_to_binary
    from xmitgcm import open_mdsdataset

    dirname, expected = llc_mds_datadirs

    # open dataset
    ds = open_mdsdataset(dirname,
                         iters=expected['test_iternum'],
                         geometry=expected['geometry'])

    nz, nfaces, ny, nx = expected['shape']
    md = get_extra_metadata(domain=expected['geometry'], nx=nx)
    # split in facets
    facets = rebuild_llc_facets(ds['T'], md)
    flatdata = llc_facets_3d_spatial_to_compact(facets, 'k', md)
    # compare with raw data
    raw = read_raw_data(dirname + '/T.' +
                        str(expected['test_iternum']).zfill(10) + '.data',
                        np.dtype('>f'), (nz, nfaces, ny, nx))
    flatraw = raw.flatten()

    assert len(flatdata) == len(flatraw)
    assert flatdata.min() == flatraw.min()
    assert flatdata.max() == flatraw.max()

    # write new file
    write_to_binary(flatdata, 'tmp.bin', dtype=np.dtype('f'))
    md5new = file_md5_checksum('tmp.bin')
    md5old = file_md5_checksum(dirname + '/T.' +
                               str(expected['test_iternum']).zfill(10) +
                               '.data')
    assert md5new == md5old
    os.remove('tmp.bin')


def test_parse_namelist(tmpdir, mds_datadirs_with_inputfiles):
    from xmitgcm.utils import parse_namelist

    dirname, expected = mds_datadirs_with_inputfiles
    exp_vals = expected['expected_namelistvals']
    # read namelist "data"
    data = parse_namelist(os.path.join(dirname, 'data'))

    assert data['PARM01']['eosType'] == exp_vals['eosType']
    assert data['PARM01']['viscAh'] == exp_vals['viscAh']
    assert data['PARM03']['niter0'] == exp_vals['niter0']
    assert data['PARM04']['delX'] == exp_vals['delX']

    diags = parse_namelist(os.path.join(dirname, 'data.diagnostics'))

    assert diags['DIAGNOSTICS_LIST']['levels'] == exp_vals['levels']
    assert diags['DIAGNOSTICS_LIST']['fileName'] == exp_vals['fileName']

    pkgs = parse_namelist(os.path.join(dirname, 'data.pkg'))

    assert pkgs['PACKAGES']['useDiagnostics'] is exp_vals['useDiagnostics']

    with open(os.path.join(str(tmpdir), 'invalid_namelists'), 'w') as f:
        f.write("# This is an invalid namelist\n"
                " &PARM01\n"
                " tRef= 12*10.,\n"
                " sRef= 15*1e3,\n"
                " cosPower=invalid\n"
                " viscAr=1.E-3,\n"
                " &\n")

    with pytest.warns(UserWarning, match='Unable to read value'):
        data = parse_namelist(os.path.join(str(tmpdir), 'invalid_namelists'),
                              silence_errors=True)
    assert data['PARM01']['cosPower'] is None


    with pytest.raises(ValueError, match='Unable to read value'):
        parse_namelist(os.path.join(str(tmpdir), 'invalid_namelists'),
                       silence_errors=False)



