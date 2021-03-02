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
import xarray as xr
import sys

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


def read_mds(fname, iternum=None, use_mmap=None, endian='>', shape=None,
             dtype=None, use_dask=True, extra_metadata=None, chunks="3D",
             llc=False, llc_method="smallchunks", legacy=True):
    """Read an MITgcm .meta / .data file pair


    PARAMETERS
    ----------
    fname : str
        The base name of the data file pair (without a .data or .meta suffix)
    iternum : int, optional
        The iteration number suffix
    use_mmap : bool, optional
        Whether to read the data using a numpy.memmap.
        Mutually exclusive with `use_dask`.
    endian : {'>', '<', '|'}, optional
        Dndianness of the data
    dtype : numpy.dtype, optional
        Data type of the data (will be inferred from the .meta file by default)
    shape : tuple, optional
        Shape of the data (will be inferred from the .meta file by default)
    use_dask : bool, optional
        Whether wrap the reading of the raw data in a ``dask.delayed`` object.
        Mutually exclusive with `use_mmap`.
    extra_metadata : dict, optional
        Dictionary containing some extra metadata that will be appended to
        content of MITgcm meta file to create the file_metadata. This is needed
        for llc type configurations (global or regional). In this case the
        extra metadata used is of the form :

        aste = {'has_faces': True, 'ny': 1350, 'nx': 270,
        'ny_facets': [450,0,270,180,450],
        'pad_before_y': [90,0,0,0,0],
        'pad_after_y': [0,0,0,90,90],
        'face_facets': [0, 0, 2, 3, 4, 4],
        'facet_orders' : ['C', 'C', 'C', 'F', 'F'],
        'face_offsets' : [0, 1, 0, 0, 0, 1],
        'transpose_face' : [False, False, False,
        True, True, True]}

        llc90 = {'has_faces': True, 'ny': 13*90, 'nx': 90,
        'ny_facets': [3*90, 3*90, 90, 3*90, 3*90],
        'face_facets': [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4],
        'facet_orders': ['C', 'C', 'C', 'F', 'F'],
        'face_offsets': [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2],
        'transpose_face' : [False, False, False,
        False, False, False, False,
        True, True, True, True, True, True]}

        llc grids have typically 5 rectangular facets and will be mapped onto
        N (=13 for llc, =6 for aste) square faces.
        Keys for the extra_metadata dictionary can be of different types and
        length:


        * bool:

        #. has_faces : True if domain is combination of connected grids

        * list of len=nfacets:

        #. ny_facets : number of points in y direction of each facet
        (usually n * nx)
        #. pad_before_y (Regional configuration) : pad data with N zeros
        before array
        #. pad_after_y (Regional configuration) : pad data with N zeros
        after array
        #. facet_order : row/column major order of this facet

        * list of len=nfaces:

        #. face_facets : facet of origin for this face

        #. face_offsets : position of the face in the facet (0 = start)

        #. transpose_face : transpose the data for this face

    chunks : {'3D', '2D', 'CS'}
        Which routine to use for chunking data. '2D' splits the file
        into a individual dask chunk of size (nx x nx) for each face (if llc)
        of each record of each level.
        '3D' loads the whole raw data file (either into memory or as a
        numpy.memmap) and is not suitable for llc configurations.
        The different methods will have different memory and i/o performance
        depending on the details of the system configuration.
        'CS' loads 2d (nx, ny) chunks for each face of the Cube Sphere model.

    obsolete : llc and llc_methods, kept for testing

    RETURNS
    -------
    data : dict
       The keys correspond to the variable names of the different variables in
       the data file. The values are the data itself, either as an
       ``numpy.ndarray``, ``numpy.memmap``, or ``dask.array.Array`` depending
       on the options selected.
    """

    if use_mmap and use_dask:
        raise TypeError('`use_mmap` and `use_dask` are mutually exclusive:'
                        ' Both memory-mapped and dask arrays'
                        ' use lazy evaluation.')
    elif use_mmap is None:
        use_mmap = False if use_dask else True

    if iternum is None:
        istr = ''
    else:
        assert isinstance(iternum, int)
        istr = '.%010d' % iternum
    datafile = fname + istr + '.data'
    metafile = fname + istr + '.meta'

    if use_mmap and use_dask:
        raise TypeError('nope')
    elif use_mmap is None:
        use_mmap = False if use_dask else True

    # get metadata
    try:
        metadata = parse_meta_file(metafile)
        nrecs, shape, name, dtype, fldlist = \
            _get_useful_info_from_meta_file(metafile)
        dtype = dtype.newbyteorder(endian)
    except IOError:
        # we can recover from not having a .meta file if dtype and shape have
        # been specified already
        if shape is None:
            raise IOError("Cannot find the shape associated to %s in the \
                          metadata." % fname)
        elif dtype is None:
            raise IOError("Cannot find the dtype associated to %s in the \
                          metadata, please specify the default dtype to \
                          avoid this error." % fname)
        else:
            # add time dimensions
            shape = (1,) + shape
            shape = list(shape)
            name = os.path.basename(fname)

            metadata = {'basename': name, 'shape': shape}

    # figure out dimensions
    ndims = len(shape)-1
    if ndims == 3:
        _, nz, ny, nx = shape
        dims_vars = ('nz', 'ny', 'nx')
    elif ndims == 2:
        _, ny, nx = shape
        nz = 1
        dims_vars = ('ny', 'nx')

    # and variables
    if 'fldList' not in metadata:
        metadata['fldList'] = [metadata['basename']]

    # if not provided in extra_metadata, we assume that the variables in file
    # have the same shape
    if extra_metadata is None or 'dims_vars' not in extra_metadata:
        dims_vars_list = []
        for var in metadata['fldList']:
            dims_vars_list.append(dims_vars)

    # add extra dim information and set aside
    metadata.update({'dims_vars': dims_vars_list,
                     'dtype': dtype, 'endian': endian,
                     'nx': nx, 'ny': ny,
                     'nz': nz, 'nt': 1})  # parse_meta harcoded for nt = 1

    file_metadata = metadata.copy()

    # by default, we set to non-llc grid
    file_metadata.update({'filename': datafile, 'vars': metadata['fldList'],
                          'has_faces': False})

    # extra_metadata contains informations about llc/regional llc grid
    if extra_metadata is not None and llc:
        nhpts_ex = extra_metadata['nx'] * extra_metadata['ny']
        nhpts = metadata['nx'] * metadata['ny']
        # check that nx * ny is consistent between extra_metadata and meta file
        # unless it's a vertical profile nx = ny = 1
        if nhpts > 1:
            assert nhpts_ex == nhpts
    if extra_metadata is not None:
        file_metadata.update(extra_metadata)

    # --------------- LEGACY --------------------------
    # from legacy code (needs to be phased out)
    # transition code to keep unit tests working
    if llc:
        chunks = "2D"
    # --------------- /LEGACY --------------------------

    # it is possible to override the values of nx, ny, nz from extra_metadata
    # (needed for bug meta file ASTE) except if those are = 1 (vertical coord)
    # where we override by values found in meta file
    for dim in ['nx', 'ny', 'nz']:
        if metadata[dim] == 1:
            file_metadata.update({dim: 1})

    # read all variables from file into the list d
    d = read_all_variables(file_metadata['fldList'], file_metadata,
                           use_mmap=use_mmap, use_dask=use_dask,
                           chunks=chunks)

    # convert list into dictionary
    out = {}
    for n, name in enumerate(file_metadata['fldList']):
        if ndims == 3:
            out[name] = d[n]
        elif ndims == 2:
            out[name] = d[n][:, 0, :]

    # --------------- LEGACY --------------------------
    # from legacy code (needs to be phased out)
    # transition code to keep unit tests working
    if legacy:
        for n, name in enumerate(file_metadata['fldList']):
            out[name] = out[name][0, :]
    # --------------- /LEGACY --------------------------
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


def parse_namelist(file, silence_errors=True):
    """Read a FOTRAN namelist file into a dictionary.

    PARAMETERS
    ----------
    file : str
        Path to the namelist file to read.

    RETURNS
    -------
    data : dict
        Dictionary of each namelist as dictionaries
    """
    def parse_val(val):
        """Parse a string and cast it in the appropriate python type."""
        if ',' in val:  # It's a list, parse recursively
            return [parse_val(subval.strip()) for subval in val.split(',')]
        elif val.startswith("'"):  # It's a string, remove quotes.
            return val[1:-1].strip()
        elif '*' in val:  # It's shorthand for a repeated value
            repeat, number = val.split('*')
            return [parse_val(number)] * int(repeat)
        elif val in ['.TRUE.', '.FALSE.']:
            return val == '.TRUE.'
        elif '.' in val or 'E' in val:  # It is a Real (float)
            return float(val)
        # Finally try for an int
        return int(val)

    data = {}
    current_namelist = ''
    raw_lines = []
    with open(file) as f:
        for line in f:
            # Remove comments
            line = line.split('#')[0].strip()
            if '=' in line or '&' in line:
                raw_lines.append(line)
            elif line:
                raw_lines[-1] += line

    for line in raw_lines:
        if line.startswith('&'):
            current_namelist = line.split('&')[1]
            if current_namelist:  # else : it's the end of a namelist.
                data[current_namelist] = {}
        else:
            field, value = map(str.strip, line[:-1].split('='))
            try:
                value = parse_val(value)
            except ValueError:
                mess = ('Unable to read value for field {field} in file {file}: {value}'
                        ).format(field=field, file=file, value=value)
                if silence_errors:
                    warnings.warn(mess)
                    value = None
                else:
                    raise ValueError(mess)

            if '(' in field:  # Field is an array
                field, idxs = field[:-1].split('(')
                if field not in data[current_namelist]:
                    data[current_namelist][field] = []
                # For generality, we will assign a slice, so we cast in list
                value = value if isinstance(value, list) else [value]
                idxs = [slice(int(idx.split(':')[0]) - 1,
                              int(idx.split(':')[1]))
                        if ':' in idx else slice(int(idx) - 1, int(idx))
                        for idx in idxs.split(',')]

                datafield = data[current_namelist][field]
                # Array are 1D or 2D, if 2D we extend it to the good shape,
                # filling it with [] and pass the appropriate sublist.
                # Only works with slice assign (a:b) in first position.
                missing_spots = idxs[-1].stop - len(datafield)
                if missing_spots > 0:
                    datafield.extend([] for i in range(missing_spots))
                if len(idxs) == 2:
                    datafield = datafield[idxs[1].start]
                datafield[idxs[0]] = value
            else:
                data[current_namelist][field] = value
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
                elif rlev == 'L':  # pragma : no cover
                    # max(Nr, Nrphys) according to doc...
                    # this seems to be only used in atmos
                    # with different levels for dynamics and physics
                    # setting to Nr meanwhile
                    zcoord = [rcoords[rpoint]]
                elif rlev == 'X':
                    if layers:
                        layer_name = key.ljust(8)[-4:].strip()
                        n_layers = layers[layer_name]
                        if levs == n_layers:
                            suffix = 'bounds'
                        elif levs == (n_layers-1):
                            suffix = 'center'
                        elif levs == (n_layers-2):
                            suffix = 'interface'
                        else:  # pragma: no cover
                            suffix = None
                            warnings.warn("Could not match rlev = %g to a "
                                          "layers coordiante" % rlev)
                        # dimname = ('layer_' + layer_name + '_' +
                        #            suffix if suffix
                        dimname = (('l' + layer_name[0] + '_' +
                                    suffix[0]) if suffix else '_UNKNOWN_')
                        zcoord = [dimname]
                    else:
                        zcoord = ['_UNKNOWN_']
                else:  # pragma: no cover
                    warnings.warn("Not sure what to do with rlev = " + rlev)
                    warnings.warn("corresponding diag_id  = " + str(diag_id))
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
    dtype = np.dtype(dtype)

    if method == "smallchunks":

        def load_chunk(nface, nlev):
            return _read_2d_face(fname, nface, nlev, nx,
                                 dtype=dtype, memmap=memmap)[None, None, None]

        chunks = (1, 1, 1, nx, nx)
        shape = (nrecs, nz, LLC_NUM_FACES, nx, nx)
        name = 'llc-' + tokenize(fname, shape, dtype,
                                 method)  # unique identifier
        # we hack the record number as extra vertical levels
        dsk = {(name, nrec, nlev, nface, 0, 0): (load_chunk, nface,
                                                 nlev + nz*nrec)
               for nface in range(LLC_NUM_FACES)
               for nlev in range(nz)
               for nrec in range(nrecs)}

        data = dsa.Array(dsk, name, chunks, dtype=dtype, shape=shape)

    elif method == "bigchunks":
        shape = (nrecs, nz, LLC_NUM_FACES*nx, nx)
        # the dimension that needs to be reshaped
        jdim = 2
        data = read_raw_data(fname, dtype, shape, use_mmap=memmap)
        data = _reshape_llc_data(data, jdim)

    # automatically squeeze off z dimension; this matches mds file behavior
    if nz == 1:
        data = data[:, 0]
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
                       use_dask=False, chunks="3D"):
    """
    Return a dictionary of dask arrays for variables in a MDS file

    PARAMETERS
    ----------
    variable_list   : list
                      list of MITgcm variables, from fldList in .meta
    file_metadata   : dict
                      internal metadata for binary file
    use_mmap        : bool, optional
                      Whether to read the data using a numpy.memmap
    chunks : str, optional
                      Whether to read 2D (default) or 3D chunks
                      2D chunks are reading (x,y) levels and 3D chunks
                      are reading the a (x,y,z) field
    RETURNS
    -------
    out : list
        list of data arrays (dask.array, numpy.ndarray or memmap)
        corresponding to variables from given list in the file
        described by file_metadata

    """

    out = []
    for variable in variable_list:
        if chunks == "2D":
            out.append(read_2D_chunks(variable, file_metadata,
                                      use_mmap=use_mmap, use_dask=use_dask))
        elif chunks == "3D":
            out.append(read_3D_chunks(variable, file_metadata,
                                      use_mmap=use_mmap, use_dask=use_dask))
        elif chunks == "CS":
            out.append(read_CS_chunks(variable, file_metadata,
                                      use_mmap=use_mmap, use_dask=use_dask))

    return out


def read_CS_chunks(variable, file_metadata, use_mmap=False, use_dask=False):
    """
    Return dask array for variable, from the file described by file_metadata,
    using the "cube sphere chunks" method.

    Parameters
    ----------
    variable : string
               name of the variable to read
    file_metadata : dict
               internal file_metadata for binary file
    use_mmap : bool, optional
               Whether to read the data using a numpy.memmap
    use_dask : bool, optional (not working yet)
               collect the data lazily or eagerly

    Returns
    -------
    dask array for variable, with 2d (ny, nx) chunks
    or numpy.ndarray or memmap, depending on input args

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
        data_raw = np.reshape(data_raw, shape)  # memmap -> ndarray
        chunks = (file_metadata['nt'], 1, 1,
                  file_metadata['ny'], file_metadata['nx'])
        data = dsa.from_array(data_raw, chunks=chunks)

    else:
        nfaces = len(file_metadata['ny_facets'])
        shape = (file_metadata['nt'], file_metadata['nz'],
                 file_metadata['ny'], nfaces, file_metadata['nx'])

        data_raw = read_raw_data(file_metadata['filename'],
                                 file_metadata['dtype'],
                                 shape, use_mmap=use_mmap,
                                 offset=0, order='C', partial_read=False)
        # data_raw = np.reshape(data_raw, shape)  # memmap -> ndarray
        chunks = (file_metadata['nt'], 1,
                  file_metadata['ny'], 1, file_metadata['nx'])
        data = dsa.from_array(data_raw, chunks=chunks)

    if not use_dask:
        data = data.compute()

    return data


def read_2D_chunks(variable, file_metadata, use_mmap=False, use_dask=False):
    """
    Return dask array for variable, from the file described by file_metadata,
    reading 2D chunks.

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
    or numpy.ndarray or memmap, depending on input args

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
        data_raw = np.reshape(data_raw, shape)  # memmap -> ndarray
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
            name = 'llcmds-' + tokenize(file_metadata, variable)

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
            name = 'mds-' + tokenize(file_metadata, variable)

            dsk = {(name, rec, lev, 0, 0): (load_chunk, lev, rec)
                   for lev in range(file_metadata['nz'])
                   for rec in range(file_metadata['nt'])}

        data = dsa.Array(dsk, name, chunks,
                         dtype=file_metadata['dtype'], shape=shape)

    if not use_dask:
        data = data.compute()

    return data


def read_3D_chunks(variable, file_metadata, use_mmap=False, use_dask=False):
    """
    Return dask array for variable, from the file described by file_metadata,
    reading 3D chunks. Not suitable for llc data.

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
    or numpy.ndarray or memmap, depending on input args

    """

    def load_chunk(rec):
        return _read_xyz_chunk(variable, file_metadata,
                               rec=rec,
                               use_mmap=use_mmap)[None]

    chunks = (1, file_metadata['nz'], file_metadata['ny'], file_metadata['nx'])
    shape = (file_metadata['nt'], file_metadata['nz'],
             file_metadata['ny'], file_metadata['nx'])
    name = 'mds-' + tokenize(file_metadata, variable)

    dsk = {(name, rec, 0, 0, 0): (load_chunk, rec)
           for rec in range(file_metadata['nt'])}

    data = dsa.Array(dsk, name, chunks,
                     dtype=file_metadata['dtype'], shape=shape)

    if not use_dask:
        data = data.compute()

    return data


def _read_xyz_chunk(variable, file_metadata, rec=0, use_mmap=False):
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
        raise ValueError(
            "_read_xyz_chunk cannot be called with llc or cs type grid")

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
    if (nt > 1) or (len(file_metadata['vars']) > 1):
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


def get_extra_metadata(domain='llc', nx=90):
    """ 
    Return the extra_metadata dictionay for selected domains

    PARAMETERS
    ----------
    domain : str
        domain can be llc, aste, cs
    nx : int
        size of the face in the x direction

    RETURNS
    -------
    extra_metadata : dict
        all extra_metadata to handle multi-faceted grids
    """

    available_domains = ['llc', 'aste', 'cs']
    if domain not in available_domains:
        raise ValueError('not an available domain')

    # domains
    llc = {'has_faces': True, 'ny': 13*nx, 'nx': nx,
           'ny_facets': [3*nx, 3*nx, nx, 3*nx, 3*nx],
           'face_facets': [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4],
           'facet_orders': ['C', 'C', 'C', 'F', 'F'],
           'face_offsets': [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2],
           'transpose_face': [False, False, False,
                              False, False, False, False,
                              True, True, True, True, True, True]}

    aste = {'has_faces': True, 'ny': 5*nx, 'nx': nx,
            'ny_facets': [int(5*nx/3.), 0, nx,
                          int(2*nx/3.), int(5*nx/3.)],
            'pad_before_y': [int(1*nx/3.), 0, 0, 0, 0],
            'pad_after_y': [0, 0, 0, int(1*nx/3.), int(1*nx/3.)],
            'face_facets': [0, 0, 2, 3, 4, 4],
            'facet_orders': ['C', 'C', 'C', 'F', 'F'],
            'face_offsets': [0, 1, 0, 0, 0, 1],
            'transpose_face': [False, False, False,
                               True, True, True]}

    cs = {'has_faces': True, 'ny': nx, 'nx': nx,
          'ny_facets': [nx, nx, nx, nx, nx, nx],
          'face_facets': [0, 1, 2, 3, 4, 5],
          'facet_orders': ['F', 'F', 'F', 'F', 'F', 'F'],
          'face_offsets': [0, 0, 0, 0, 0, 0],
          'transpose_face': [False, False, False,
                             False, False, False]}

    if domain == 'llc':
        extra_metadata = llc
    elif domain == 'aste':
        extra_metadata = aste
    elif domain == 'cs':
        extra_metadata = cs

    return extra_metadata


def get_grid_from_input(gridfile, nx=None, ny=None, geometry='llc',
                        dtype=np.dtype('d'), endian='>', use_dask=False,
                        extra_metadata=None):
    """ 
    Read grid variables from grid input files, this is especially useful
    for llc and cube sphere configurations used with land tiles
    elimination. Reading the input grid files (e.g. tile00[1-5].mitgrid)
    allows to fill in the blanks of eliminated land tiles.

    PARAMETERS
    ----------
    gridfile : str
        gridfile must contain <NFACET> as wildcard (e.g. tile<NFACET>.mitgrid)
    nx : int
        size of the face in the x direction
    ny : int
        size of the face in the y direction
    geometry : str
        domain geometry can be llc, cs or carthesian not supported yet
    dtype : np.dtype
        numeric precision (single/double) of input data
    endian : string
        endianness of input data
    use_dask : bool
        use dask or not
    extra_metadata : dict
        dictionary of extra metadata, needed for llc configurations

    RETURNS
    ------- 
    grid : xarray.Dataset
        all grid variables
    """

    file_metadata = {}
    # grid variables are stored in this order
    file_metadata['fldList'] = ['XC', 'YC', 'DXF', 'DYF', 'RAC',
                                'XG', 'YG', 'DXV', 'DYU', 'RAZ',
                                'DXC', 'DYC', 'RAW', 'RAS', 'DXG', 'DYG']

    file_metadata['vars'] = file_metadata['fldList']
    dims_vars_list = []
    for var in file_metadata['fldList']:
        dims_vars_list.append(('ny', 'nx'))
    file_metadata['dims_vars'] = dims_vars_list

    # no vertical levels or time records
    file_metadata['nz'] = 1
    file_metadata['nt'] = 1

# for curvilinear non-facet grids (TO DO)
#    if nx is not None:
#        file_metadata['nx'] = nx
#    if ny is not None:
#        file_metadata['ny'] = ny
    if extra_metadata is not None:
        file_metadata.update(extra_metadata)

    # numeric representation
    file_metadata['endian'] = endian
    file_metadata['dtype'] = dtype

    if geometry in ['llc', 'cs']:
        try:
            nfaces = len(file_metadata['face_facets'])
        except:
            raise ValueError('metadata must contain face_facets')
    if geometry == 'llc':
        nfacets = 5
    elif geometry == 'cs':
        nfacets = 6

    # create placeholders for data
    gridfields = {}
    for field in file_metadata['fldList']:
        gridfields.update({field: None})

    if geometry in ['llc', 'cs']:
        for kfacet in range(nfacets):
            # we need to adapt the metadata to the grid file
            grid_metadata = file_metadata.copy()

            fname = gridfile.replace('<NFACET>', str(kfacet+1).zfill(3))
            grid_metadata['filename'] = fname

            if file_metadata['facet_orders'][kfacet] == 'C':
                nxgrid = file_metadata['nx'] + 1
                nygrid = file_metadata['ny_facets'][kfacet] + 1
            elif file_metadata['facet_orders'][kfacet] == 'F':
                nxgrid = file_metadata['ny_facets'][kfacet] + 1
                nygrid = file_metadata['nx'] + 1

            grid_metadata.update({'nx': nxgrid, 'ny': nygrid,
                                  'has_faces': False})

            raw = read_all_variables(grid_metadata['vars'], grid_metadata,
                                     use_dask=use_dask)

            rawfields = {}
            for kfield in np.arange(len(file_metadata['fldList'])):

                rawfields.update(
                    {file_metadata['fldList'][kfield]: raw[kfield]})

            for field in file_metadata['fldList']:
                # symetrize
                tmp = rawfields[field][:, :, :-1, :-1].squeeze()
                # transpose
                if grid_metadata['facet_orders'][kfacet] == 'F':
                    tmp = tmp.transpose()

                for face in np.arange(nfaces):
                    # identify faces that need to be filled
                    if grid_metadata['face_facets'][face] == kfacet:
                        # get offset of face from facet
                        offset = file_metadata['face_offsets'][face]
                        nx = file_metadata['nx']
                        # pad data, if needed (would trigger eager data eval)
                        # needs a new array not to pad multiple times
                        padded = _pad_array(tmp, file_metadata, face=face)
                        # extract the data
                        dataface = padded[offset*nx:(offset+1)*nx, :]
                        # transpose, if needed
                        if file_metadata['transpose_face'][face]:
                            dataface = dataface.transpose()
                        # assign values
                        dataface = dsa.stack([dataface], axis=0)
                        if face == 0:
                            gridfields[field] = dataface
                        else:
                            gridfields[field] = dsa.concatenate(
                                [gridfields[field], dataface], axis=0)

    # create the dataset
    if geometry == 'llc':
        grid = xr.Dataset({'XC':  (['face', 'j', 'i'],     gridfields['XC']),
                           'YC':  (['face', 'j', 'i'],     gridfields['YC']),
                           'DXF': (['face', 'j', 'i'],     gridfields['DXF']),
                           'DYF': (['face', 'j', 'i'],     gridfields['DYF']),
                           'RAC': (['face', 'j', 'i'],     gridfields['RAC']),
                           'XG':  (['face', 'j_g', 'i_g'], gridfields['XG']),
                           'YG':  (['face', 'j_g', 'i_g'], gridfields['YG']),
                           'DXV': (['face', 'j', 'i'],     gridfields['DXV']),
                           'DYU': (['face', 'j', 'i'],     gridfields['DYU']),
                           'RAZ': (['face', 'j_g', 'i_g'], gridfields['RAZ']),
                           'DXC': (['face', 'j', 'i_g'],   gridfields['DXC']),
                           'DYC': (['face', 'j_g', 'i'],   gridfields['DYC']),
                           'RAW': (['face', 'j', 'i_g'],   gridfields['RAW']),
                           'RAS': (['face', 'j_g', 'i'],   gridfields['RAS']),
                           'DXG': (['face', 'j_g', 'i'],   gridfields['DXG']),
                           'DYG': (['face', 'j', 'i_g'],   gridfields['DYG'])
                           },
                          coords={'i': (['i'], np.arange(file_metadata['nx'])),
                                  'j': (['j'], np.arange(file_metadata['nx'])),
                                  'i_g': (['i_g'],
                                          np.arange(file_metadata['nx'])),
                                  'j_g': (['j_g'],
                                          np.arange(file_metadata['nx'])),
                                  'face': (['face'], np.arange(nfaces))
                                  }
                          )
    elif geometry == 'cs':
        grid = xr.Dataset({'XC':  (['face', 'i', 'j'],     gridfields['XC']),
                           'YC':  (['face', 'i', 'j'],     gridfields['YC']),
                           'DXF': (['face', 'i', 'j'],     gridfields['DXF']),
                           'DYF': (['face', 'i', 'j'],     gridfields['DYF']),
                           'RAC': (['face', 'i', 'j'],     gridfields['RAC']),
                           'XG':  (['face', 'i_g', 'j_g'], gridfields['XG']),
                           'YG':  (['face', 'i_g', 'j_g'], gridfields['YG']),
                           'DXV': (['face', 'i', 'j'],     gridfields['DXV']),
                           'DYU': (['face', 'i', 'j'],     gridfields['DYU']),
                           'RAZ': (['face', 'i_g', 'j_g'], gridfields['RAZ']),
                           'DXC': (['face', 'i', 'j_g'],   gridfields['DXC']),
                           'DYC': (['face', 'i_g', 'j'],   gridfields['DYC']),
                           'RAW': (['face', 'i', 'j_g'],   gridfields['RAW']),
                           'RAS': (['face', 'i_g', 'j'],   gridfields['RAS']),
                           'DXG': (['face', 'i_g', 'j'],   gridfields['DXG']),
                           'DYG': (['face', 'i', 'j_g'],   gridfields['DYG'])
                           },
                          coords={'i': (['i'], np.arange(file_metadata['nx'])),
                                  'j': (['j'], np.arange(file_metadata['nx'])),
                                  'i_g': (['i_g'],
                                          np.arange(file_metadata['nx'])),
                                  'j_g': (['j_g'],
                                          np.arange(file_metadata['nx'])),
                                  'face': (['face'], np.arange(nfaces))
                                  }
                          )
    else:  # pragma: no cover
        grid = xr.Dataset({'XC':  (['j', 'i'],     gridfields['XC']),
                           'YC':  (['j', 'i'],     gridfields['YC']),
                           'DXF': (['j', 'i'],     gridfields['DXF']),
                           'DYF': (['j', 'i'],     gridfields['DYF']),
                           'RAC': (['j', 'i'],     gridfields['RAC']),
                           'XG':  (['j_g', 'i_g'], gridfields['XG']),
                           'YG':  (['j_g', 'i_g'], gridfields['YG']),
                           'DXV': (['j', 'i'],     gridfields['DXV']),
                           'DYU': (['j', 'i'],     gridfields['DYU']),
                           'RAZ': (['j_g', 'i_g'], gridfields['RAZ']),
                           'DXC': (['j', 'i_g'],   gridfields['DXC']),
                           'DYC': (['j_g', 'i'],   gridfields['DYC']),
                           'RAW': (['j', 'i_g'],   gridfields['RAW']),
                           'RAS': (['j_g', 'i'],   gridfields['RAS']),
                           'DXG': (['j_g', 'i'],   gridfields['DXG']),
                           'DYG': (['j', 'i_g'],   gridfields['DYG'])
                           },
                          coords={'i': (['i'], np.arange(file_metadata['nx'])),
                                  'j': (['j'], np.arange(file_metadata['ny'])),
                                  'i_g': (['i_g'],
                                          np.arange(file_metadata['nx'])),
                                  'j_g': (['j_g'],
                                          np.arange(file_metadata['ny']))
                                  }
                          )

    return grid


########## WRITING BINARIES #############################

def find_concat_dim_facet(da, facet, extra_metadata):
    """ In llc grids, find along which horizontal dimension to concatenate
    facet between i, i_g and j, j_g. If the order of the facet is F, concat
    along i or i_g. If order is C, concat along j or j_g. Also return
    horizontal dim not to concatenate

    PARAMETERS
    ----------
    da : xarray.DataArray
        xmitgcm llc data array
    facet : int
        facet number
    extra_metadata : dict
        dict of extra_metadata from get_extra_metadata

    RETURNS
    -------
    concat_dim, nonconcat_dim : str, str
        names of the dimensions for concatenation or not

    """
    order = extra_metadata['facet_orders'][facet]
    if order == 'C':
        possible_concat_dims = ['j', 'j_g']
    elif order == 'F':
        possible_concat_dims = ['i', 'i_g']

    concat_dim = find_concat_dim(da, possible_concat_dims)

    # we also need to other horizontal dimension for vector indexing
    all_dims = list(da.dims)
    # discard face
    all_dims.remove('face')
    # remove the concat_dim to find horizontal non_concat dimension
    all_dims.remove(concat_dim)
    non_concat_dim = all_dims[0]
    return concat_dim, non_concat_dim


def find_concat_dim(da, possible_concat_dims):
    """ look for available dimensions in dataaray and pick the one
    from a list of candidates

    PARAMETERS
    ----------
    da : xarray.DataArray
        xmitgcm llc data array
    possible_concat_dims : list
        list of potential dims

    RETURNS
    -------
    out : str
        dimension on which to concatenate

    """
    out = None
    for d in possible_concat_dims:
        if d in da.dims:
            out = d
    return out


def rebuild_llc_facets(da, extra_metadata):
    """ For LLC grids, rebuilds facets from a xmitgcm dataarray and
    store into a dictionary

    PARAMETERS
    ----------
    da : xarray.DataArray
        xmitgcm llc data array
    extra_metadata : dict
        dict of extra_metadata from get_extra_metadata

    RETURNS
    -------
    facets : dict
        all facets data in xarray.DataArray form packed into a dictionary

    """

    nfacets = len(extra_metadata['facet_orders'])
    nfaces = len(extra_metadata['face_facets'])
    facets = {}

    # rebuild the facets (with padding if present)
    for kfacet in range(nfacets):
        facets.update({'facet' + str(kfacet): None})

        concat_dim, non_concat_dim = find_concat_dim_facet(
            da, kfacet, extra_metadata)

        for kface in range(nfaces):
            # concatenate faces back into facets
            if extra_metadata['face_facets'][kface] == kfacet:
                if extra_metadata['face_offsets'][kface] == 0:
                    # first face of facet
                    tmp = da.sel(face=kface)
                else:
                    # any other face needs to be concatenated
                    newface = da.sel(face=kface)
                    tmp = xr.concat([facets['facet' + str(kfacet)],
                                     newface], dim=concat_dim)

                facets['facet' + str(kfacet)] = tmp

    # if present, remove padding from facets
    for kfacet in range(nfacets):

        concat_dim, non_concat_dim = find_concat_dim_facet(
            da, kfacet, extra_metadata)

        # remove pad before
        if 'pad_before_y' in extra_metadata:
            pad = extra_metadata['pad_before_y'][kfacet]
            # padded array
            padded = facets['facet' + str(kfacet)]

            if pad != 0:
                # we need to relabel the grid cells
                ng = len(padded[concat_dim].values)
                padded[concat_dim] = np.arange(ng)
                # select index from non-padded array
                unpadded_bef = padded.isel({concat_dim: range(pad, ng)})
            else:
                unpadded_bef = padded

            facets['facet' + str(kfacet)] = unpadded_bef

        # remove pad after
        if 'pad_after_y' in extra_metadata:
            pad = extra_metadata['pad_after_y'][kfacet]
            # padded array
            padded = facets['facet' + str(kfacet)]

            if pad != 0:
                # we need to relabel the grid cells
                ng = len(padded[concat_dim].values)
                padded[concat_dim] = np.arange(ng)
                # select index from non-padded array
                last = ng-pad
                unpadded_aft = padded.isel({concat_dim: range(last)})
            else:
                unpadded_aft = padded

            facets['facet' + str(kfacet)] = unpadded_aft

    return facets


def llc_facets_3d_spatial_to_compact(facets, dimname, extra_metadata):
    """ Write in compact form a list of 3d facets

    PARAMETERS
    ----------
    facets : dict
        dict of xarray.dataarrays for the facets
    extra_metadata : dict
        extra_metadata from get_extra_metadata

    RETURNS
    -------
    flatdata : numpy.array
        all the data in vector form
    """

    nz = len(facets['facet0'][dimname])
    nfacets = len(facets)
    flatdata = np.array([])

    for kz in range(nz):
        # rebuild the dict
        tmpdict = {}
        for kfacet in range(nfacets):
            this_facet = facets['facet' + str(kfacet)]
            if this_facet is not None:
                tmpdict['facet' + str(kfacet)] = this_facet.isel(k=kz)
            else:
                tmpdict['facet' + str(kfacet)] = None
        # concatenate all 2d arrays
        compact2d = llc_facets_2d_to_compact(tmpdict, extra_metadata)
        flatdata = np.concatenate([flatdata, compact2d])

    return flatdata


def llc_facets_2d_to_compact(facets, extra_metadata):
    """ Write in compact form a list of 2d facets

    PARAMETERS
    ----------
    facets: dict
        dict of xarray.dataarrays for the facets
    extra_metadata: dict
        extra_metadata from get_extra_metadata

    RETURNS
    -------
    flatdata : numpy.array
        all the data in vector form
    """

    flatdata = np.array([])
    # loop over facets
    for kfacet in range(len(facets)):
        if facets['facet' + str(kfacet)] is not None:
            tmp = np.reshape(facets['facet' + str(kfacet)].values, (-1))
            flatdata = np.concatenate([flatdata, tmp])

    return flatdata


def write_to_binary(flatdata, fileout, dtype=np.dtype('f')):
    """ write data in binary file

    PARAMETERS
    ----------
    flatdata: numpy.array
        vector of data to write
    fileout: str
        output file name
    dtype: np.dtype
        single/double precision

    RETURNS
    -------
    None
    """
    # write data to binary files
    fid = open(fileout, "wb")
    tmp = flatdata.astype(dtype)
    if sys.byteorder == 'little':
        tmp = tmp.byteswap(True)
    fid.write(tmp.tobytes())
    fid.close()
    return None
