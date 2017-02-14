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


def read_raw_data(datafile, dtype, shape, use_mmap=False):
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

    RETURNS
    -------
    data : numpy.ndarray
        The data (or a memmap to it)
    """

    #print("Reading raw data in %s" % datafile)
    # first check to be sure there is the right number of bytes in the file
    number_of_values = reduce(lambda x, y: x * y, shape)
    expected_number_of_bytes = number_of_values * dtype.itemsize
    actual_number_of_bytes = os.path.getsize(datafile)
    if expected_number_of_bytes != actual_number_of_bytes:
        raise IOError('File `%s` does not have the correct size '
                      '(expected %g, found %g)' %
                      (datafile,
                       expected_number_of_bytes,
                       actual_number_of_bytes))
    if use_mmap:
        # print("Reading %s using memmap" % datafile)
        d = np.memmap(datafile, dtype, 'r')
    else:
        # print("Reading %s using fromfile" % datafile)
        d = np.fromfile(datafile, dtype)
    d.shape = shape
    return d


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

    tile_shape = _llc_face_shape(llc_id)
    data_shape = (NUM_FACES,) + face_shape
    if nz is not None:
        data_shape = (nz,) + data_shape

    # should we accomodate multiple records?
    # no, not in this function
    return data_shape
