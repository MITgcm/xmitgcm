"""
Utility functions for reading MITgcm mds files (.meta / .data)
"""
# python 3 compatiblity
from __future__ import print_function, division

import re
import os
import numpy as np
import warnings
from functools import reduce
from dask import delayed
import dask.array as dsa
from dask.base import tokenize

def parse_meta_file(fname):
    """Get the metadata as a dict out of the MITgcm mds .meta file.

    PARAMETERS
    ----------
    fname : str
        Path to the .meta file

    RETURNS
    -------
    flds : dict
        Metadata in dictionary form.
    """
    flds = {}
    basename = re.match('(^.+?)\..+', os.path.basename(fname)).groups()[0]
    flds['basename'] = basename
    with open(fname) as f:
        text = f.read()
    # split into items
    for item in re.split(';', text):
        # remove whitespace at beginning
        item = re.sub('^\s+', '', item)
        match = re.match('(\w+) = (\[|\{)(.*)(\]|\})', item, re.DOTALL)
        if match:
            key, _, value, _ = match.groups()
            # remove more whitespace
            value = re.sub('^\s+', '', value)
            value = re.sub('\s+$', '', value)
            # print key,':', value
            flds[key] = value
    # now check the needed things are there
    needed_keys = ['dimList', 'nDims', 'nrecords', 'dataprec']
    for k in needed_keys:
        assert k in flds
    # transform datatypes
    flds['nDims'] = int(flds['nDims'])
    flds['nrecords'] = int(flds['nrecords'])
    # endianness is set by _read_mds
    flds['dataprec'] = np.dtype(re.sub("'", '', flds['dataprec']))
    flds['dimList'] = [[int(h) for h in
                       re.split(',', g)] for g in
                       re.split(',\n', flds['dimList'])]
    if 'fldList' in flds:
        flds['fldList'] = [re.match("'*(\w+)", g).groups()[0] for g in
                           re.split("'\s+'", flds['fldList'])]
        assert flds['nrecords'] == len(flds['fldList'])
    return flds

def _get_useful_info_from_meta_file(metafile):
    # why does the .meta file contain so much repeated info?
    # Here we just get the part we need
    # and reverse order (numpy uses C order, mds is fortran)
    meta = parse_meta_file(metafile)
    shape = [g[0] for g in meta['dimList']][::-1]
    assert len(shape) == meta['nDims']
    # now add an extra for number of recs
    nrecs = meta['nrecords']
    shape.insert(0, nrecs)
    dtype = meta['dataprec']
    if 'fldList' in meta:
        fldlist = meta['fldList']
        name = fldlist[0]
    else:
        name = meta['basename']
        fldlist = None

    return nrecs, shape, name, dtype, fldlist

def read_mds(fname, iternum=None, use_mmap=True, force_dict=True, endian='>',
             shape=None, dtype=None, dask_delayed=True, llc=False,
             llc_method="smallchunks"):
    """Read an MITgcm .meta / .data file pair


    PARAMETERS
    ----------
    fname : str
        The base name of the data file pair (without a .data or .meta suffix)
    iternum : int, optional
        The iteration number suffix
    use_mmap : bool, optional
        Whether to read the data using a numpy.memmap
    force_dict : bool, optional
        Whether to return a dictionary of ``{varname: data}`` pairs
    endian : {'>', '<', '|'}, optional
        Dndianness of the data
    dtype : numpy.dtype, optional
        Data type of the data (will be inferred from the .meta file by default)
    shape : tuple, optional
        Shape of the data (will be inferred from the .meta file by default)
    dask_delayed : bool, optional
        Whether wrap the reading of the raw data in a ``dask.delayed`` object
    llc : bool, optional
        Whether the data is from an LLC geometry
    llc_method : {'smalchunks', 'bigchunks'}
        Which routine to use for reading raw LLC. "smallchunks" splits the file
        into a individual dask chunk of size (nx x nx) for each face of each
        level (i.e. the total number of chunks is 13 * nz). "bigchunks" loads
        the whole raw data file (either into memory or as a numpy.memmap),
        splits it into faces, and concatenates those faces together using
        ``dask.array.concatenate``. The different methods will have different
        memory and i/o performance depending on the details of the system
        configuration.

    RETURNS
    -------
    data : dict
       The keys correspond to the variable names of the different variables in
       the data file. The values are the data itself, either as an
       ``numpy.ndarray``, ``numpy.memmap``, or ``dask.array.Array`` depending
       on the options selected.
    """

    if iternum is None:
        istr = ''
    else:
        assert isinstance(iternum, int)
        istr = '.%010d' % iternum
    datafile = fname + istr + '.data'
    metafile = fname + istr + '.meta'

    # get metadata
    try:
        nrecs, shape, name, dtype, fldlist = _get_useful_info_from_meta_file(metafile)
        dtype = dtype.newbyteorder(endian)
    except IOError:
        # we can recover from not having a .meta file if dtype and shape have
        # been specified already
        if shape is None:
            raise IOError("Cannot find the shape associated to %s in the metadata." %fname)
        elif dtype is None:
            raise IOError("Cannot find the dtype associated to %s in the metadata, "
                          "please specify the default dtype to avoid this error." %fname)
        else:
            nrecs = 1
            shape = list(shape)
            shape.insert(0, nrecs)
            name = os.path.basename(fname)

    # TODO: refactor overall logic of the code below

    # this will exclude vertical profile files
    if llc and shape[-1]>1:
        # remeberer that the first dim is nrec
        if len(shape)==4:
            _, nz, ny, nx = shape
        else:
            _, ny, nx = shape
            nz = 1

        if llc_method=='bigchunks' and (not use_mmap):
            # this would load a ton of data... need to delay it
            d = dsa.from_delayed(
                delayed(read_3d_llc_data)(datafile, nz, nx, dtype=dtype,
                            memmap=memmap, nrecs=nrecs, method=llc_method)
            )
        else:
            if llc_method=='smallchunks':
                use_mmap=False
            d = read_3d_llc_data(datafile, nz, nx, dtype=dtype, memmap=use_mmap,
                              nrecs=nrecs, method=llc_method)

    elif dask_delayed:
        d = dsa.from_delayed(
              delayed(read_raw_data)(datafile, dtype, shape, use_mmap=use_mmap),
              shape, dtype
            )
    else:
        d = read_raw_data(datafile, dtype, shape, use_mmap=use_mmap)

    if nrecs == 1:
        if force_dict:
            return {name: d[0]}
        else:
            return d[0]
    else:
        # need record names
        out = {}
        for n, name in enumerate(fldlist):
            out[name] = d[n]
        return out


def read_raw_data(datafile, dtype, shape, use_mmap=False, offset=0,
                  order='C', partial_read=False):
    """Read a raw binary file and shape it.

    PARAMETERS
    ----------
    datafile : str
        Path to a .data file
    dtype : numpy.dtype
        Data type of the data
    shape : tuple
        Shape of the data
    use_memmap : bool, optional
        Whether to read the data using a numpy.memmap
    offset : int, optional
        Offset (in bytes) to apply on read
    order : str, optional
        Row/Column Major = 'C' or 'F'
    partial_read : bool, optional
        If reading part of the file

    RETURNS
    -------
    data : numpy.ndarray
        The data (or a memmap to it)
    """

    number_of_values = reduce(lambda x, y: x * y, shape)
    expected_number_of_bytes = number_of_values * dtype.itemsize
    actual_number_of_bytes = os.path.getsize(datafile)
    if not partial_read:
        # first check that partial_read and offset are used together
        if offset != 0:
            raise ValueError(
                'When partial_read==False, offset will not be read')
        # second check to be sure there is the right number of bytes in file
        if expected_number_of_bytes != actual_number_of_bytes:
            raise IOError('File `%s` does not have the correct size '
                          '(expected %g, found %g)' %
                          (datafile,
                           expected_number_of_bytes,
                           actual_number_of_bytes))
    else:
        pass

    if offset < actual_number_of_bytes:
        pass
    else:
        raise ValueError('bytes offset %g is greater than file size %g' %
                         (offset, actual_number_of_bytes))

    with open(datafile, 'rb') as f:
        if use_mmap:
            data = np.memmap(f, dtype=dtype, mode='r', offset=offset,
                             shape=tuple(shape), order=order)
        else:
            f.seek(offset)
            data = np.fromfile(f, dtype=dtype, count=number_of_values)
            data = data.reshape(shape, order=order)
    data.shape = shape
    return data


def parse_available_diagnostics(fname, layers={}):
    """Examine the available_diagnostics.log file and translate it into
    useful variable metadata.

    PARAMETERS
    ----------
    fname : str or buffer
        the path to the diagnostics file or a file buffer
    layers : dict (optional)
        dictionary mapping layers names to dimension sizes

    RETURNS
    -------
    all_diags : a dictionary keyed by variable names with values
        (coords, description, units)
    """
    all_diags = {}
    diag_id_lookup = {}
    mate_lookup = {}

    # mapping between the available_diagnostics.log codes and the actual
    # coordinate names
    # http://mitgcm.org/public/r2_manual/latest/online_documents/node268.html
    xcoords = {'U': 'i_g', 'V': 'i', 'M': 'i', 'Z': 'i_g'}
    ycoords = {'U': 'j', 'V': 'j_g', 'M': 'j', 'Z': 'j_g'}
    rcoords = {'M': 'k', 'U': 'k_u', 'L': 'k_l'}

    # need to be able to accept string filename or buffer
    def process_buffer(f):
        for l in f:
            # will automatically skip first four header lines
            c = re.split('\|', l)
            if len(c) == 7 and c[0].strip() != 'Num':
                # parse the line to extract the relevant variables
                key = c[1].strip()
                diag_id = int(c[0].strip())
                diag_id_lookup[diag_id] = key
                levs = int(c[2].strip())
                mate = c[3].strip()
                if mate:
                    mate = int(mate)
                    mate_lookup[key] = mate
                code = c[4]
                units = c[5].strip()
                desc = c[6].strip()

                # decode what those variables mean
                hpoint = code[1]
                rpoint = code[8]
                xycoords = [ycoords[hpoint], xcoords[hpoint]]
                rlev = code[9]

                if rlev == '1' and levs == 1:
                    zcoord = []
                elif rlev == 'R':
                    zcoord = [rcoords[rpoint]]
                elif rlev == 'X' and layers:
                    layer_name = key.ljust(8)[-4:].strip()
                    n_layers = layers[layer_name]
                    if levs == n_layers:
                        suffix = 'bounds'
                    elif levs == (n_layers-1):
                        suffix = 'center'
                    elif levs == (n_layers-2):
                        suffix = 'interface'
                    else:
                        suffix = None
                        warnings.warn("Could not match rlev = %g to a layers"
                                      "coordiante" % rlev)
                    # dimname = ('layer_' + layer_name + '_' + suffix if suffix
                    dimname = (('l' + layer_name[0] + '_' + suffix[0]) if suffix
                               else '_UNKNOWN_')
                    zcoord = [dimname]
                else:
                    warnings.warn("Not sure what to do with rlev = " + rlev)
                    zcoord = ['_UNKNOWN_']
                coords = zcoord + xycoords
                all_diags[key] = dict(dims=coords,
                                      # we need a standard name
                                      attrs={'standard_name': key,
                                             'long_name': desc,
                                             'units': units})
    try:
        with open(fname) as f:
            process_buffer(f)
    except TypeError:
        process_buffer(fname)

    # add mate information
    for key, mate_id in mate_lookup.items():
        all_diags[key]['attrs']['mate'] = diag_id_lookup[mate_id]
    return all_diags

# stuff related to llc mds file structure
LLC_NUM_FACES=13
facet_strides = ((0,3), (3,6), (6,7), (7,10), (10,13))
facet_orders = ('C', 'C', 'C', 'F', 'F')
face_facets = [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4]
face_offsets = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2]
transpose_face = [False, False, False, False, False, False, False,
                  True, True, True, True, True, True]

def _read_2d_facet(fname, nfacet, nlev, nx, dtype='>f8', memmap=True):
    # make sure we have a valid dtype
    dtype = np.dtype(dtype)
    nbytes = dtype.itemsize

    # where the facet starts in the file
    facet_offset = facet_strides[nfacet][0] * nx * nx * nbytes
    level_offset = LLC_NUM_FACES * nx * nx * nbytes * nlev
    offset = facet_offset + level_offset

    # the array order of the facet
    facet_order = facet_orders[nfacet]

    # the size shape of the facet
    facet_ny = (facet_strides[nfacet][1] - facet_strides[nfacet][0])*nx
    facet_shape = (facet_ny, nx)
    facet_nitems = facet_ny * nx
    with open(fname, 'rb') as f:
        #print("Reading %s facet %g nlev %g" % (fname, nfacet, nlev))
        if memmap:
            data = np.memmap(f, dtype=dtype, mode='r', offset=offset,
                             shape=facet_shape, order=facet_order)
        else:
            f.seek(offset)
            data = np.fromfile(f, dtype=dtype, count=facet_nitems)
            data = data.reshape(facet_shape, order=facet_order)
    return data

def _read_2d_face(fname, nface, nlev, nx, dtype='>f8', memmap=True):
    # make sure we have a valid dtype
    nfacet = face_facets[nface]
    face_slice = slice(nx*face_offsets[nface], nx*(face_offsets[nface]+1))
    facet_offset = nx * face_offsets[nface]
    data_facet = _read_2d_facet(fname, nfacet, nlev, nx,
                               dtype=dtype, memmap=memmap)
    data = data_facet[face_slice]
    if transpose_face[nface]:
        data = data.T
    return data

# manually construct dask graph
def read_3d_llc_data(fname, nz, nx, dtype='>f8', memmap=True, nrecs=1,
                     method="smallchunks"):
    """Read a three-dimensional LLC file using a custom dask graph.

    PARAMETERS
    ----------
    fname : string
        Path to the file on disk
    nz : int
        Number of vertical levels
    nx : int
        Size of each face side dimension
    dtype : np.dtype, optional
        Datatype of the data
    memmap : bool, optional
        Whether to read the data using np.memmap. Forced to be ``False`` for
        ``method="smallchunks"``.
    nrecs : int, optional
        The number of records in a multi-record file
    method : {"smallchunks", "bigchunks"}, optional
        Which routine to use for reading raw LLC. "smallchunks" splits the file
        into a individual dask chunk of size (nx x nx) for each face of each
        level (i.e. the total number of chunks is 13 * nz). "bigchunks" loads
        the whole raw data file (either into memory or as a numpy.memmap),
        splits it into faces, and concatenates those faces together using
        ``dask.array.concatenate``. The different methods will have different
        memory and i/o performance depending on the details of the system
        configuration.

    RETURNS
    -------
    data : dask.array.Array
        The data
    """
    dtype=np.dtype(dtype)

    if method=="smallchunks":

        def load_chunk(nface, nlev):
            return _read_2d_face(fname, nface, nlev, nx,
                            dtype=dtype, memmap=memmap)[None, None, None]

        chunks = (1, 1, 1, nx, nx)
        shape = (nrecs, nz, LLC_NUM_FACES, nx, nx)
        name = 'llc-' + tokenize(fname)  # unique identifier
        # we hack the record number as extra vertical levels
        dsk = {(name, nrec, nlev, nface, 0, 0): (load_chunk, nface,
                                                 nlev + nz*nrec)
                 for nface in range(LLC_NUM_FACES)
                 for nlev in range(nz)
                 for nrec in range(nrecs)}

        data = dsa.Array(dsk, name, chunks, dtype=dtype, shape=shape)

    elif method=="bigchunks":
        shape = (nrecs, nz, LLC_NUM_FACES*nx, nx)
        # the dimension that needs to be reshaped
        jdim = 2
        data = read_raw_data(fname, dtype, shape, use_mmap=memmap)
        data= _reshape_llc_data(data, jdim)

    # automatically squeeze off z dimension; this matches mds file behavior
    if nz==1:
        data = data[:,0]
    return data



# a deprecated function that I can't bear to delete because it was painful to
# write
def _reshape_llc_data(data, jdim):  # pragma: no cover
    """Fix the weird problem with llc data array order."""
    # Can we do this without copying any data?
    # If not, we need to go upstream and implement this at the MDS level
    # Or can we fudge it with dask?
    # this is all very specific to the llc file output
    # would be nice to generalize more, but how?
    nside = data.shape[jdim] // LLC_NUM_FACES
    # how the LLC data is laid out along the j dimension
    strides = ((0,3), (3,6), (6,7), (7,10), (10,13))
    # whether to reshape each face
    reshape = (False, False, False, True, True)
    # this will slice the data into 5 facets
    slices = [jdim * (slice(None),) + (slice(nside*st[0], nside*st[1]),)
              for st in strides]
    facet_arrays = [data[sl] for sl in slices]
    face_arrays = []
    for ar, rs, st in zip(facet_arrays, reshape, strides):
        nfaces_in_facet = st[1] - st[0]
        shape = list(ar.shape)
        if rs:
            # we assume the other horizontal dimension is immediately after jdim
            shape[jdim] = ar.shape[jdim+1]
            shape[jdim+1] = ar.shape[jdim]
        # insert a length-1 dimension along which to concatenate
        shape.insert(jdim, 1)
        # this modify the array shape in place, with no copies allowed
        # but it doesn't work with dask arrays
        # ar.shape = shape
        ar = ar.reshape(shape)
        # now ar is propery shaped, but we still need to slice it into faces
        face_slice_dim = jdim + 1 + rs
        for n in range(nfaces_in_facet):
            face_slice = (face_slice_dim * (slice(None),) +
                          (slice(nside*n, nside*(n+1)),))
            data_face = ar[face_slice]
            face_arrays.append(data_face)

    # We can't concatenate using numpy (hcat etc.) because it makes a copy,
    # presumably loading the memmaps into memory.
    # Using dask gets around this.
    # But what if we want different chunks, or already chunked the data
    # upstream? Doesn't seem like this is ideal
    # TODO: Refactor handling of dask arrays and chunking
    #return np.concatenate(face_arrays, axis=jdim)
    # the dask version doesn't work because of this:
    # https://github.com/dask/dask/issues/1645
    face_arrays_dask = [dsa.from_array(fa, chunks=fa.shape)
                        for fa in face_arrays]
    concat = dsa.concatenate(face_arrays_dask, axis=jdim)
    return concat


def _llc_face_shape(llc_id):
    """Given an integer identifier for the llc grid, return the face shape."""

    # known valid LLC configurations
    if llc_id in (90, 270, 1080, 2160, 4320):
        return (llc_id, llc_id)
    else:
        raise ValueError("%g is not a valid llc identifier" % llc_id)

def _llc_data_shape(llc_id, nz=None):
    """Given an integer identifier for the llc grid, and possibly a number of
    vertical grid points, return the expected shape of the full data field."""

    # this is a constant for all LLC setups
    NUM_FACES = 13

    face_shape = _llc_face_shape(llc_id)
    data_shape = (NUM_FACES,) + face_shape
    if nz is not None:
        data_shape = (nz,) + data_shape

    # should we accomodate multiple records?
    # no, not in this function
    return data_shape


def read_all_variables(variable_list, file_metadata, use_mmap=False,
                       chunks="small"):
    """
    Return a dictionary of dask arrays for variables in a MDS file

    Parameters
    ----------
    variable_list : list
                    list of MITgcm variables, from fldList in .meta
    file_metadata : dict
                    internal metadata for binary file
    use_mmap      : bool, optional
                    Whether to read the data using a numpy.memmap
    chunks        : str, optional
                    Whether to read small (default) or big chunks
    Returns
    -------
    list of dask arrays, corresponding to variables from given list
    in the file described by file_metadata

    """

    out = []
    for variable in variable_list:
        if chunks == "small":
            out.append(read_small_chunks(variable, file_metadata,
                                         use_mmap=use_mmap))
        elif chunks == "big":
            out.append(read_big_chunks(variable, file_metadata,
                                       use_mmap=use_mmap))

    return out


def read_small_chunks(variable, file_metadata, use_mmap=False, use_dask=True):
    """
    Return dask array for variable, from the file described by file_metadata,
    using the "small chunks" method.

    Parameters
    ----------
    variable : string
               name of the variable to read
    file_metadata : dict
               internal file_metadata for binary file
    use_mmap : bool, optional
               Whether to read the data using a numpy.memmap
    use_dask : bool, optional
               collect the data lazily or eagerly

    Returns
    -------
    dask array for variable, with 2d (ny, nx) chunks

    """

    if (file_metadata['nx'] == 1) and (file_metadata['ny'] == 1) and \
       (len(file_metadata['vars']) == 1):
            # vertical coordinate
        data_raw = read_raw_data(file_metadata['filename'],
                                 file_metadata['dtype'],
                                 (file_metadata['nz'],), use_mmap=use_mmap,
                                 offset=0, order='C', partial_read=False)

        shape = (file_metadata['nt'], file_metadata['nz'], 1,
                 file_metadata['ny'], file_metadata['nx'])
        data_raw = np.reshape(data_raw, shape)
        chunks = (file_metadata['nt'], 1, 1,
                  file_metadata['ny'], file_metadata['nx'])
        data = dsa.from_array(data_raw, chunks=chunks)

    else:
        if file_metadata['has_faces']:
            def load_chunk(face, lev, rec):
                return _read_xy_chunk(variable, file_metadata, rec=rec,
                                      lev=lev, face=face,
                                      use_mmap=use_mmap)[None, None, None]

            chunks = (1, 1, 1, file_metadata['nx'], file_metadata['nx'])
            shape = (file_metadata['nt'], file_metadata['nz'],
                     len(file_metadata['face_facets']),
                     file_metadata['nx'], file_metadata['nx'])
            name = 'llc-' + tokenize(file_metadata['filename'])

            dsk = {(name, rec, lev, face, 0, 0): (load_chunk, face,
                                                  lev, rec)
                   for face in range(len(file_metadata['face_facets']))
                   for lev in range(file_metadata['nz'])
                   for rec in range(file_metadata['nt'])}

        else:
            def load_chunk(lev, rec):
                return _read_xy_chunk(variable, file_metadata,
                                      rec=rec, lev=lev,
                                      face=0, use_mmap=use_mmap)[None, None]

            chunks = (1, 1, file_metadata['ny'], file_metadata['nx'])
            shape = (file_metadata['nt'], file_metadata['nz'],
                     file_metadata['ny'], file_metadata['nx'])
            name = 'reg-' + tokenize(file_metadata['filename'])

            dsk = {(name, rec, lev, 0, 0): (load_chunk, lev, rec)
                   for lev in range(file_metadata['nz'])
                   for rec in range(file_metadata['nt'])}

        data = dsa.Array(dsk, name, chunks,
                         dtype=file_metadata['dtype'], shape=shape)

    if not use_dask:
        data = data.compute()

    return data


def read_big_chunks(variable, file_metadata, use_mmap=False, use_dask=True):
    """
    Return dask array for variable, from the file described by file_metadata,
    using the "big chunks" method. Not suitable for llc data.

    Parameters
    ----------
    variable : string
               name of the variable to read
    file_metadata : dict
               internal file_metadata for binary file
    use_mmap : bool, optional
               Whether to read the data using a numpy.memmap
    use_dask : bool, optional
               collect the data lazily or eagerly

    Returns
    -------
    dask array for variable, with 3d (nz, ny, nx) chunks

    """

    def load_chunk(rec):
        return _read_3d_chunk(variable, file_metadata,
                              rec=rec,
                              use_mmap=use_mmap)[None]

    chunks = (1, file_metadata['nz'], file_metadata['ny'], file_metadata['nx'])
    shape = (file_metadata['nt'], file_metadata['nz'],
             file_metadata['ny'], file_metadata['nx'])
    name = 'reg-' + tokenize(file_metadata['filename'])

    dsk = {(name, rec, 0, 0, 0): (load_chunk, rec)
           for rec in range(file_metadata['nt'])}

    data = dsa.Array(dsk, name, chunks,
                     dtype=file_metadata['dtype'], shape=shape)

    if not use_dask:
        data = data.compute()

    return data


def _read_3d_chunk(variable, file_metadata, rec=0, use_mmap=False):
    """
    Read a 3d chunk (x,y,z) of variable from file described in
    file_metadata.

    Parameters
    ----------
    variable : string
               name of the variable to read
    file_metadata : dict
               file_metadata for binary file
    rec      : integer, optional
               time record to read (default=0)
    use_mmap : bool, optional
               Whether to read the data using a numpy.memmap

    Returns
    -------
    numpy array or memmap
    """

    if file_metadata['has_faces'] and ((file_metadata['nx'] > 1) or
                                       (file_metadata['ny'] > 1)):
        raise ValueError("_read_3d_chunk cannot be called with llc type grid")

    # size of the data element
    nbytes = file_metadata['dtype'].itemsize
    # byte order
    file_metadata['datatype'] = file_metadata['dtype'].newbyteorder(
        file_metadata['endian'])
    # find index of variable
    idx_var = file_metadata['vars'].index(variable)

    # 1. compute offset_variable, init to zero
    offset_vars = 0
    # loop on variables before the one to read
    for jvar in np.arange(idx_var):
        # inspect its dimensions
        dims = file_metadata['dims_vars'][jvar]
        # compute the byte size of this variable
        nbytes_thisvar = 1*nbytes
        for dim in dims:
            nbytes_thisvar = nbytes_thisvar*file_metadata[dim]
        # update offset from previous variables
        offset_vars = offset_vars+nbytes_thisvar

    # 2. get dimensions of desired variable
    dims = file_metadata['dims_vars'][idx_var]
    # inquire for values of dimensions, else return 1
    nt, nz, ny, nx = [file_metadata.get(dimname, 1)
                      for dimname in ('nt', 'nz', 'ny', 'nx')]

    # 3. compute offset from previous records of current variable
    if (rec > nt-1):
        raise ValueError("time record %g greater than number of records %g" %
                         (rec, nt))
    else:
        offset_timerecords = rec * nz * ny * nx * nbytes

    # 4. compute the offset of the previous variables, records and levels
    offset = offset_vars + offset_timerecords
    shape = (nz, ny, nx,)

    # check if we do a partial read of the file
    if (nt > 1) or (nz > 1) or (len(file_metadata['vars']) > 1):
        partial_read = True
    else:
        partial_read = False

    # define the order (row/column major)
    # in conventional grids, it's in C
    order = 'C'

    # 5. Do the actual read
    data = read_raw_data(file_metadata['filename'],
                         file_metadata['datatype'],
                         shape, use_mmap=use_mmap, offset=offset,
                         order=order, partial_read=partial_read)

    return data


def _read_xy_chunk(variable, file_metadata, rec=0, lev=0, face=0,
                   use_mmap=False):
    """
    Read a 2d chunk along (x,y) of variable from file described in
    file_metadata.

    Parameters
    ----------
    variable : string
               name of the variable to read
    file_metadata : dict
               file_metadata for binary file
    rec      : integer, optional
               time record to read (default=0)
    lev      : integer, optional
               vertical level to read (default=0)
    face     : integer, optional
               face to read for llc configurations (default=0)
    use_mmap : bool, optional
               Whether to read the data using a numpy.memmap

    Returns
    -------
    numpy array or memmap
    """

    # size of the data element
    nbytes = file_metadata['dtype'].itemsize
    # byte order
    file_metadata['datatype'] = file_metadata['dtype'].newbyteorder(
        file_metadata['endian'])
    # find index of variable
    idx_var = file_metadata['vars'].index(variable)

    # 1. compute offset_variable, init to zero
    offset_vars = 0
    # loop on variables before the one to read
    for jvar in np.arange(idx_var):
        # inspect its dimensions
        dims = file_metadata['dims_vars'][jvar]
        # compute the byte size of this variable
        nbytes_thisvar = 1*nbytes
        for dim in dims:
            nbytes_thisvar = nbytes_thisvar*file_metadata[dim]
        # update offset from previous variables
        offset_vars = offset_vars+nbytes_thisvar

    # 2. get dimensions of desired variable
    dims = file_metadata['dims_vars'][idx_var]
    # inquire for values of dimensions, else return 1
    nt, nz, ny, nx = [file_metadata.get(dimname, 1)
                      for dimname in ('nt', 'nz', 'ny', 'nx')]

    # 3. compute offset from previous records of current variable
    if (rec > nt-1):
        raise ValueError("time record %g greater than number of records %g" %
                         (rec, nt))
    else:
        offset_timerecords = rec * nz * ny * nx * nbytes

    # 4. compute offset from previous vertical levels of current variable
    if (lev > nz-1):
        raise ValueError("level %g is greater than number of levels %g" %
                         (lev, nz))
    else:
        offset_verticallevels = lev * ny * nx * nbytes

    # 5. compute the offset of the previous variables, records and levels
    offset = offset_vars + offset_timerecords + offset_verticallevels

    # 6. compute offset due to faces
    if file_metadata['has_faces']:
        # determin which facet the face belong to
        facet_origin = file_metadata['face_facets'][face]
        # compute the offset from previous facets
        ny_facets = np.array(file_metadata['ny_facets'])
        nyglo_facets = np.concatenate(([0], ny_facets.cumsum()[:-1]), axis=0)
        offset_facets = nyglo_facets[facet_origin] * \
            file_metadata['nx'] * nbytes
        # update offset
        offset = offset + offset_facets
        # shape if shape of the facet
        shape = (file_metadata['ny_facets'][facet_origin], nx,)
    else:
        # no need to update offset and shape is simply:
        shape = (ny, nx,)

    # check if we do a partial read of the file
    if (nt > 1) or (nz > 1) or (len(file_metadata['vars']) > 1) or \
       file_metadata['has_faces']:
        partial_read = True
    else:
        partial_read = False

    # define the order (row/column major)
    if file_metadata['has_faces']:
        # in llc, we can have either C or F
        order = file_metadata['facet_orders'][facet_origin]
    else:
        # in conventional grids, it's in C
        order = 'C'

    # 7. Do the actual read
    data_raw = read_raw_data(file_metadata['filename'],
                             file_metadata['datatype'],
                             shape, use_mmap=use_mmap, offset=offset,
                             order=order, partial_read=partial_read)

    # 8. Pad data, if needed
    data_padded_after = _pad_array(data_raw, file_metadata, face=face)

    # 9. extract the face from the facet
    if file_metadata['has_faces'] and ('face_offsets' in file_metadata):
        face_slice = slice(nx*file_metadata['face_offsets'][face],
                           nx*(file_metadata['face_offsets'][face]+1))

        data = data_padded_after[face_slice]
    else:
        data = data_padded_after

    # 10. Transpose face, if needed
    if file_metadata['has_faces'] and ('transpose_face' in file_metadata):
        if file_metadata['transpose_face'][face]:
            data = data.transpose()

    return data


def _pad_array(data, file_metadata, face=0):
    """
    Return a padded array. If input data is a numpy.memmap and no padding
    is necessary, the function preserves its type. Otherwise, the concatenate
    forces it to load into memory.

    Parameters
    ----------

    data          : numpy array or memmap
                    input data
    file_metadata : dict
                    metadata for file
    face          : int, optional
                    llc face if applicable

    Returns
    -------
    numpy.array or numpy.memmap

    """

    # Pad data before in y direction
    if 'pad_before_y' in file_metadata:
        if file_metadata['has_faces']:
            facet_origin = file_metadata['face_facets'][face]
            nypad_before = file_metadata['pad_before_y'][facet_origin]
        else:
            nypad_before = file_metadata['pad_before_y']

        pad_before = np.zeros((nypad_before, file_metadata['nx']))
        data_padded_before = np.concatenate(
            (pad_before, data), axis=0)
    else:
        data_padded_before = data

    # Pad data after in y direction
    if 'pad_after_y' in file_metadata:
        if file_metadata['has_faces']:
            facet_origin = file_metadata['face_facets'][face]
            nypad_after = file_metadata['pad_after_y'][facet_origin]
        else:
            nypad_after = file_metadata['pad_after_y']

        pad_after = np.zeros((nypad_after, file_metadata['nx']))
        data_padded_after = np.concatenate(
            (data_padded_before, pad_after), axis=0)
    else:
        data_padded_after = data_padded_before

    return data_padded_after
