import numpy as np
import dask.array as dar
from dask import delayed
from dask.base import tokenize

# data related to llc mds file structure
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
        print("Reading %s facet %g nlev %g" % (fname, nfacet, nlev))
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
def read_3d_llc_data(fname, nz, nx, dtype='>f8', memmap=True):
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
        Whether to read the data using np.memmap

    RETURNS
    -------
    data : dask.array.Array
        The data
    """
    dtype=np.dtype(dtype)

    def load_chunk(nface, nlev):
        return _read_2d_face(fname, nface, nlev, nx,
                        dtype=dtype, memmap=memmap)[None, None]

    chunks = (1, 1, nx, nx)
    shape = (nz, LLC_NUM_FACES, nx, nx)

    name = 'llc-' + tokenize(fname)  # unique identifier
    dsk = {(name, nlev, nface, 0, 0): (load_chunk, nface, nlev)
             for nface in range(LLC_NUM_FACES)
             for nlev in range(nz)}

    data = dar.Array(dsk, name, chunks, dtype=dtype, shape=shape)

    # automatically squeeze off z dimension; this matches mds file behavior
    if nz==1:
        data = data[0]

    return data


# a deprecated function that I can't bear to delete because it was painful to
# write
def _reshape_llc_data(data, jdim):
    """Fix the weird problem with llc data array order."""
    # Can we do this without copying any data?
    # If not, we need to go upstream and implement this at the MDS level
    # Or can we fudge it with dask?
    # this is all very specific to the llc file output
    # would be nice to generalize more, but how?
    nside = data.shape[jdim] / LLC_NUM_FACES
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
    face_arrays_dask = [da.from_array(fa, chunks=fa.shape)
                        for fa in face_arrays]
    concat = da.concatenate(face_arrays_dask, axis=jdim)
    return concat
