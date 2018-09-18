from xmitgcm.test.test_xmitgcm_common import *

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

    res = read_mds(basename, chunking_method="2D")
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], dask.array.core.Array)

    res = read_mds(basename, chunking_method="2D", use_dask=False)
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], np.memmap)

    res = read_mds(basename, chunking_method="2D", use_dask=False,
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
        emeta = {'has_faces': False}
    res = read_mds(basename, chunking_method="2D", use_dask=False,
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

    res = read_mds(basename, iternum=iternum, chunking_method="2D")
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], dask.array.core.Array)

    res = read_mds(basename, iternum=iternum, chunking_method="2D",
                   use_dask=False)
    assert isinstance(res, dict)
    assert prefix in res
    print(type(res[prefix]))
    assert isinstance(res[prefix], np.ndarray)  # should be memmap

    res = read_mds(basename, iternum=iternum, chunking_method="2D",
                   use_dask=False, use_mmap=False)
    assert isinstance(res, dict)
    assert prefix in res
    assert isinstance(res[prefix], np.ndarray)


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
    else:
        file_metadata.update({'nx': file_metadata['shape'][2],
                              'ny': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'nt': 1,
                              'dims_vars': [('nz', 'ny', 'nx')],
                              'has_faces': False})

    if file_metadata['geometry'] in ['llc']:
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
    else:
        file_metadata.update({'nx': file_metadata['shape'][2],
                              'ny': file_metadata['shape'][1],
                              'nz': file_metadata['shape'][0],
                              'nt': 1,
                              'dims_vars': [('nz', 'ny', 'nx')],
                              'has_faces': False})

    # test 3D chunks, fails on llc but not others
    if file_metadata['geometry'] in ['llc']:
        with pytest.raises(ValueError):
            dataset = read_all_variables(file_metadata['vars'], file_metadata,
                                         use_mmap=memmap, use_dask=usedask,
                                         chunking_method="3D")
            if usedask:
                dataset[0].compute()
    else:
        dataset = read_all_variables(file_metadata['vars'], file_metadata,
                                     use_mmap=memmap, use_dask=usedask,
                                     chunking_method="3D")

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
    dataset = read_all_variables(file_metadata['vars'], file_metadata,
                                 use_mmap=memmap, use_dask=usedask,
                                 chunking_method="2D")

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
                                 chunking_method="2D")

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


@pytest.mark.parametrize("domain", ['llc', 'aste', 'cs'])
@pytest.mark.parametrize("nx", [90, 270])
def test_get_extra_metadata(domain, nx):
    from xmitgcm.utils import get_extra_metadata
    em = get_extra_metadata(domain=domain, nx=nx)
    assert type(em) == dict

    with pytest.raises(ValueError):
        em = get_extra_metadata(domain='notinlist', nx=nx)
