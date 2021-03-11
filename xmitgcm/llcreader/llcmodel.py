import numpy as np
import dask
import dask.array as dsa
from dask.base import tokenize, normalize_token
import xarray as xr
import warnings

from .duck_array_ops import concatenate
from .shrunk_index import all_index_data
from ..variables import dimensions

def _get_grid_metadata():
    # keep this separate from get_var_metadata
    # because grid stuff is weird
    from ..mds_store import _get_all_grid_variables
    from ..variables import extra_grid_variables, vertical_coordinates

    # get grid info
    grid_vars = _get_all_grid_variables('llc')
    grid_vars.update(extra_grid_variables)

    # make dictionary with keys as filenames
    grid_metadata = {}
    for key,val in grid_vars.items():
        # masks use hFac filename to be computed in mds_store
        if 'filename' in val and key[:4]!='mask':
            val.update({'real_name':key})
            grid_metadata[val['filename']] = val
        else:
            grid_metadata[key] = val

    # force RF to point to Zp1, deal with this manually..
    grid_metadata['RF']=vertical_coordinates['Zp1']
    grid_metadata['RF']['real_name'] = 'Zp1'
    for zv in ['Zu','Zl']:
        grid_metadata[zv] = vertical_coordinates[zv]

    return grid_metadata

def _get_var_metadata():
    # The LLC run data comes with zero metadata. So we import metadata from
    # the xmitgcm package.
    from ..variables import state_variables, package_state_variables
    from ..utils import parse_available_diagnostics
    from ..default_diagnostics import diagnostics
    from io import StringIO

    diag_file = StringIO(diagnostics)
    available_diags = parse_available_diagnostics(diag_file)
    var_metadata = state_variables.copy()
    var_metadata.update(package_state_variables)
    var_metadata.update(available_diags)

    # even the file names from the LLC data differ from standard MITgcm output
    aliases = {'Eta': 'ETAN', 'PhiBot': 'PHIBOT', 'Salt': 'SALT',
               'Theta': 'THETA'}
    for a, b in aliases.items():
        var_metadata[a] = var_metadata[b]

    # add grid metadata
    var_metadata.update(_get_grid_metadata())

    return var_metadata

_VAR_METADATA = _get_var_metadata()

def _is_vgrid(vname):
    # check for 1d, vertical grid variables
    dims = _VAR_METADATA[vname]['dims']
    return len(dims)==1 and dims[0][0]=='k'

def _get_variable_point(vname, mask_override):
    # fix for https://github.com/MITgcm/xmitgcm/issues/191
    if vname in mask_override:
        return mask_override[vname]
    dims = _VAR_METADATA[vname]['dims']
    if 'i' in dims and 'j' in dims:
        point = 'c'
    elif 'i_g' in dims and 'j' in dims:
        point = 'w'
    elif 'i' in dims and 'j_g' in dims:
        point = 's'
    elif 'i_g' in dims and 'j_g' in dims:
        raise ValueError("Don't have masks for corner points!")
    else:
        raise ValueError("Variable `%s` is not a horizontal variable." % vname)
    return point

def _get_scalars_and_vectors(varnames, type):

    for vname in varnames:
        if vname not in _VAR_METADATA:
            raise ValueError("Varname `%s` not found in metadata." % vname)

    if type != 'latlon':
        return varnames, []

    scalars = []
    vector_pairs = []
    for vname in varnames:
        meta = _VAR_METADATA[vname]
        try:
            mate = meta['attrs']['mate']
            if mate not in varnames:
                raise ValueError("Vector pairs are required to create "
                                 "latlon type datasets. Varname `%s` is "
                                 "missing its vector mate `%s`"
                                 % vname, mate)
            vector_pairs.append((vname, mate))
            varnames.remove(mate)
        except KeyError:
            scalars.append(vname)

def _decompress(data, mask, dtype):
    data_blank = np.full_like(mask, np.nan, dtype=dtype)
    data_blank[mask] = data
    data_blank.shape = mask.shape
    return data_blank

def _pad_facet(data,facet_shape,reshape,pad_before,pad_after,dtype):
    """add padding to facet data that are irregularly shaped, returning
    data of size ``facet_shape`` to make equally sized faces.
    See
    https://xmitgcm.readthedocs.io/en/latest/llcreader.html#aste-release-1-example
    for an example

    Parameters
    ----------
    data : array like
        data to be reshaped
    facet_shape : tuple
        "expected" facet shape determined by the _facet_shape function
    reshape : bool
        whether to reshape each face, True if facet is "rotated".
        This determines the axis along which pad_before or pad_after refers to
    pad_before, pad_after : int
        size of padding added to either the i or j dimension where
        before vs after determines ordering of: data-then-pad or pad-then-data
    dtype : numpy.dtype
        Datatype of the data

    Returns
    -------
    padded_data : array like
        which has shape = facet_shape, with nan's for padding
    """
    pre_shape=list(facet_shape)
    pad_shape=list(facet_shape)
    if reshape:
        concat_axis = -1
        pad_shape[concat_axis] = pad_after
        pre_shape[concat_axis] -= pad_after
        padded= [data,np.full(pad_shape,np.nan,dtype=dtype)]
    else:
        concat_axis = -2
        pad_shape[concat_axis] = pad_before
        pre_shape[concat_axis] -= pad_before
        padded = [np.full(pad_shape,np.nan,dtype=dtype),data]

    data.shape = pre_shape
    return concatenate(padded,axis=concat_axis)

def _facet_strides(nfaces):
    if nfaces == 13:
        return ((0,3), (3,6), (6,7), (7,10), (10,13))
    elif nfaces == 6:
        return ((0,2), (2,2), (2,3), (3,4), (4,6))
    else:
        raise TypeError(f'Unexpected nfaces {nfaces} for _facet_strides')

# whether to reshape each face
_facet_reshape = (False, False, False, True, True)
_nfacets = 5

def _uncompressed_facet_index(nfacet, nside, nfaces):
    face_size = nside**2
    start = _facet_strides(nfaces)[nfacet][0] * face_size
    end = _facet_strides(nfaces)[nfacet][1] * face_size
    return start, end

def _facet_shape(nfacet, nside, nfaces):
    facet_length = _facet_strides(nfaces)[nfacet][1] - _facet_strides(nfaces)[nfacet][0]
    if _facet_reshape[nfacet]:
        facet_shape = (1, nside, facet_length*nside)
    else:
        facet_shape = (1, facet_length*nside, nside)
    return facet_shape

def _facet_to_faces(data, nfacet, nfaces):
    shape = data.shape
    # facet dimension
    nf, ny, nx = shape[-3:]
    other_dims = shape[:-3]
    assert nf == 1
    facet_length = _facet_strides(nfaces)[nfacet][1] - _facet_strides(nfaces)[nfacet][0]
    if _facet_reshape[nfacet]:
        new_shape = other_dims +  (ny, facet_length, nx / facet_length) if facet_length > 0 else 0
        data_rs = data.reshape(new_shape)
        data_rs = np.moveaxis(data_rs, -2, -3) # dask-safe
    else:
        new_shape = other_dims + (facet_length, ny / facet_length, nx) if facet_length > 0 else 0
        data_rs = data.reshape(new_shape) if facet_length>0 else None
    return data_rs

def _facets_to_faces(facets, nfaces):
    all_faces = []
    for nfacet, data_facet in enumerate(facets):
        data_rs = _facet_to_faces(data_facet, nfacet, nfaces)
        if data_rs is not None:
            all_faces.append(data_rs)
    return concatenate(all_faces, axis=-3)

def _faces_to_facets(data, nfaces, facedim=-3):
    assert data.shape[facedim] == nfaces
    facets = []
    for nfacet, (strides, reshape) in enumerate(zip(_facet_strides(nfaces), _facet_reshape)):
        face_data = [data[(...,) + (slice(nface, nface+1), slice(None), slice(None))]
                     for nface in range(*strides)]
        if reshape:
            concat_axis = facedim + 2
        else:
            concat_axis = facedim + 1
        # todo: use duck typing for concat
        facet_data = concatenate(face_data, axis=concat_axis) if len(face_data)!=0 else np.array([])
        facets.append(facet_data)
    return facets


def _rotate_scalar_facet(facet):
    facet_transposed = np.moveaxis(facet, -1, -2)
    facet_rotated = np.flip(facet_transposed, -2)
    return facet_rotated


def _facets_to_latlon_scalar(all_facets):
    rotated = (all_facets[:2]
               + [_rotate_scalar_facet(facet) for facet in all_facets[-2:]])
    # drop facet dimension
    rotated = [r[..., 0, :, :] for r in rotated]
    return concatenate(rotated, axis=-1)


def _faces_to_latlon_scalar(data, nfaces):
    data_facets = _faces_to_facets(data, nfaces)
    return _facets_to_latlon_scalar(data_facets)


# dask's pad function doesn't work
# it does weird things to non-pad dimensions
# need to roll our own
def shift_and_pad(a,left=True):
    if left:
        a_shifted = a[..., 1:]
        pad_array = dsa.zeros_like(a[..., -2:-1])
        return concatenate([a_shifted, pad_array], axis=-1)
    else:
        a_shifted = a[..., :-1]
        pad_array = dsa.zeros_like(a[..., 0:1])
        return concatenate([pad_array, a_shifted], axis=-1)

def transform_v_to_u(facet):
    return _rotate_scalar_facet(facet)

def transform_u_to_v(facet, metric=False):
    # "shift" u component by 1 pixel
    pad_width = (facet.ndim - 1) * (None,) + ((1, 0),)
    #facet_padded = dsa.pad(facet[..., 1:], pad_width, 'constant')
    facet_padded = shift_and_pad(facet)
    assert facet.shape == facet_padded.shape
    facet_rotated = _rotate_scalar_facet(facet_padded)
    if not metric:
        facet_rotated = -facet_rotated
    return facet_rotated

def _facets_to_latlon_vector(facets_u, facets_v, metric=False):
    # need to pad the rotated v values
    ndim = facets_u[0].ndim
    # second-to-last axis is the one to pad, plus a facet axis
    assert ndim >= 3

    # drop facet dimension
    facets_u_drop = [f[..., 0, :, :] for f in facets_u]
    facets_v_drop = [f[..., 0, :, :] for f in facets_v]

    u_rot = (facets_u_drop[:2]
             + [transform_v_to_u(facet) for facet in facets_v_drop[-2:]])
    v_rot = (facets_v_drop[:2]
             + [transform_u_to_v(facet, metric) for facet in facets_u_drop[-2:]])

    u = concatenate(u_rot, axis=-1)
    v = concatenate(v_rot, axis=-1)
    return u, v

def _faces_to_latlon_vector(u_faces, v_faces, nfaces, metric=False):
    u_facets = _faces_to_facets(u_faces, nfaces)
    v_facets = _faces_to_facets(v_faces, nfaces)
    u, v = _facets_to_latlon_vector(u_facets, v_facets, metric=metric)
    return u, v

def _drop_facedim(dims):
    dims = list(dims)
    dims.remove('face')
    return dims

def _add_face_to_dims(dims):
    new_dims = dims.copy()
    if 'j' in dims:
        j_dim = dims.index('j')
        new_dims.insert(j_dim, 'face')
    elif 'j_g' in dims:
        j_dim = dims.index('j_g')
        new_dims.insert(j_dim, 'face')
    return new_dims

def _faces_coords_to_latlon(ds):
    coords = ds.reset_coords().coords.to_dataset()
    ifac = 4
    jfac = 3
    dim_coords = {}
    for vname in coords.coords:
        if vname[0] == 'i':
            data = np.arange(ifac * coords.dims[vname])
        elif vname[0] == 'j':
            data = np.arange(jfac * coords.dims[vname])
        else:
            data = coords[vname].data
        var = xr.Variable(ds[vname].dims, data, ds[vname].attrs)
        dim_coords[vname] = var
    return xr.Dataset(dim_coords)

def faces_dataset_to_latlon(ds, metric_vector_pairs=[('dxC', 'dyC'), ('dyG', 'dxG')]):
    """Transform a 13-face LLC xarray Dataset into a rectancular grid,
    discarding the Arctic.

    Parameters
    ----------
    ds : xarray.Dataset
        A 13-face LLC dataset
    metric_vector_pairs : list, optional
        Pairs of variables that are positive-definite metrics located at grid
        edges.

    Returns
    -------
    out : xarray.Dataset
        Transformed rectangular dataset
    """

    coord_vars = list(ds.coords)
    ds_new = _faces_coords_to_latlon(ds)

    vector_pairs = []
    scalars = []
    vnames = list(ds.reset_coords().variables)

    for vname in vnames:
        try:
            mate = ds[vname].attrs['mate']
        except KeyError:
            mate = None

        # Raises an exception if the mate of a variable in vnames is missing.
        if mate is not None:
            vector_pairs.append((vname, mate))
            try:
                vnames.remove(mate)
            except ValueError:
                msg = 'If {} in varnames, {} must also be in varnames'.format(vname, mate)
                raise ValueError(msg)

    all_vector_components = [inner for outer in (vector_pairs + metric_vector_pairs)
                             for inner in outer]
    scalars = [vname for vname in vnames if vname not in all_vector_components]
    data_vars = {}

    for vname in scalars:
        if vname=='face' or vname in ds_new:
            continue
        if 'face' in ds[vname].dims:
            data = _faces_to_latlon_scalar(ds[vname].data,nfaces=len(ds['face']))
            dims = _drop_facedim(ds[vname].dims)
        else:
            data = ds[vname].data
            dims = ds[vname].dims
        data_vars[vname] = xr.Variable(dims, data, ds[vname].attrs)

    for vname_u, vname_v in vector_pairs:
        data_u, data_v = _faces_to_latlon_vector(ds[vname_u].data, ds[vname_v].data,
                                                 nfaces=len(ds['face']))
        data_vars[vname_u] = xr.Variable(_drop_facedim(ds[vname_u].dims), data_u, ds[vname_u].attrs)
        data_vars[vname_v] = xr.Variable(_drop_facedim(ds[vname_v].dims), data_v, ds[vname_v].attrs)
    for vname_u, vname_v in metric_vector_pairs:
        data_u, data_v = _faces_to_latlon_vector(ds[vname_u].data, ds[vname_v].data, nfaces=len(ds['face']), metric=True)
        data_vars[vname_u] = xr.Variable(_drop_facedim(ds[vname_u].dims), data_u, ds[vname_u].attrs)
        data_vars[vname_v] = xr.Variable(_drop_facedim(ds[vname_v].dims), data_v, ds[vname_v].attrs)


    ds_new = ds_new.update(data_vars)
    ds_new = ds_new.set_coords([c for c in coord_vars if c in ds_new])
    return ds_new


# below are data transformers

def _all_facets_to_faces(data_facets, meta, nfaces):
    return {vname: _facets_to_faces(data, nfaces)
            for vname, data in data_facets.items()}


def _all_facets_to_latlon(data_facets, meta, nfaces=None):

    vector_pairs = []
    scalars = []
    vnames = list(data_facets)

    for vname in vnames:
        try:
            mate = meta[vname]['attrs']['mate']
        except KeyError:
            mate = None

        # Raises an exception if the mate of a variable in vnames is missing.
        if mate is not None:
            vector_pairs.append((vname, mate))
            try:
                vnames.remove(mate)
            except ValueError:
                msg = 'If {} in varnames, {} must also be in varnames'.format(vname, mate)
                raise ValueError(msg)

    all_vector_components = [inner for outer in vector_pairs for inner in outer]
    scalars = [vname for vname in vnames if vname not in all_vector_components]

    data = {}
    for vname in scalars:
        data[vname] = _facets_to_latlon_scalar(data_facets[vname])

    for vname_u, vname_v in vector_pairs:
        data_u, data_v = _facets_to_latlon_vector(data_facets[vname_u],
                                                  data_facets[vname_v])
        data[vname_u] = data_u
        data[vname_v] = data_v

    return data

def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def _get_facet_chunk(store, varname, iternum, nfacet, klevels, nx, nz, nfaces,
                     dtype, mask_override, domain, pad_before, pad_after):

    fs, path = store.get_fs_and_full_path(varname, iternum)

    assert (nfacet >= 0) & (nfacet < _nfacets)

    file = fs.open(path)

    # insert singleton axis for time (if not grid var) and k level
    facet_shape = (1,) + _facet_shape(nfacet, nx, nfaces)
    facet_shape = (1,) + facet_shape if iternum is not None else facet_shape

    level_data = []


    if (store.shrunk and iternum is not None) or \
       (store.shrunk_grid and iternum is None):
        # the store tells us whether we need a mask or not
        point = _get_variable_point(varname, mask_override)
        mykey = nx if domain == 'global' else f'{domain}_{nx}'
        index = all_index_data[mykey][point]
        zgroup = store.open_mask_group()
        mask = zgroup['mask_' + point].astype('bool')
    else:
        index = None
        mask = None

    # Offset start/end read position due to padding facet before me
    pre_pad = np.cumsum([x+y for x,y in zip(pad_before,pad_after)])
    pre_pad = shift_and_pad(pre_pad,left=False).compute()
    tot_pad = pre_pad[-1]+pad_after[-1]

    for k in klevels:
        assert (k >= 0) & (k < nz)

        # figure out where in the file we have to read to get the data
        # for this level and facet
        if index:
            i = np.ravel_multi_index((k, nfacet), (nz, _nfacets))
            start = index[i]
            end = index[i+1]
        else:
            level_start = k * (nx**2 * nfaces - nx * tot_pad)
            facet_start, facet_end = _uncompressed_facet_index(nfacet, nx, nfaces)
            start = level_start + facet_start
            end = level_start + facet_end - nx*pad_after[nfacet]
            end = end - nx*(pad_before[nfacet]) if k*nfacet==0 else end

            start,end = [x - nx*pre_pad[nfacet] if k+nfacet!=0 else x for x in [start,end]]


        read_offset = start * dtype.itemsize # in bytes
        read_length  = (end - start) * dtype.itemsize # in bytes
        file.seek(read_offset)
        buffer = file.read(read_length)
        data = np.frombuffer(buffer, dtype=dtype)

        assert len(data) == (end - start)

        if mask:
            mask_level = mask[k]
            mask_facets = _faces_to_facets(mask_level,nfaces)
            this_mask = mask_facets[nfacet]
            data = _decompress(data, this_mask, dtype)

        elif pad_before[nfacet]+pad_after[nfacet]>0:

            # Extra care for pad after with rotated fields
            data = _pad_facet(data,facet_shape,_facet_reshape[nfacet],
                              pad_before[nfacet],pad_after[nfacet],dtype)


        # this is the shape this facet is supposed to have
        data.shape = facet_shape
        level_data.append(data)

    out = np.concatenate(level_data, axis=-4)
    return out

def _get_1d_chunk(store, varname, klevels, nz, dtype):
    """for 1D vertical grid variables"""

    fs, path = store.get_fs_and_full_path(varname, None)

    file = fs.open(path)

    # read all levels for 1D variables
    read_length = nz*dtype.itemsize # all levels in bytes
    buffer = file.read(read_length)
    data = np.frombuffer(buffer,dtype=dtype)

    return data[klevels]

class BaseLLCModel:
    """Class representing an LLC Model Dataset.

    Parameters
    ----------
    store : llcreader.BaseStore
        The store object where the data can be found
    mask_ds : zarr.Group
        Must contain variables `mask_c`, `masc_w`, `mask_s`

    Attributes
    ----------
    dtype : numpy.dtype
        Datatype of the data in the dataset
    nx : int
        Number of gridpoints per face (e.g. 90, 1080, 4320, etc.)
    nz : int
        Number of vertical gridpoints
    delta_t : float
        Numerical timestep
    time_units : str
        Date unit string, e.g 'seconds since 1948-01-01 12:00:00'
    iter_start : int
        First model iteration number (inclusive; follows python range conventions)
    iter_stop : int
        Final model iteration number (exclusive; follows python range conventions)
    iter_step : int
        Spacing between iterations
    iters : list of ints
        Specific iteration numbers in a list, possibly with nonuniform spacing.
        Either provide this or the iter parameters above.
    varnames, grid_varnames : list
        List of data variable and grid variable names contained in the dataset
    mask_override : dict
        Override inference of masking variable, e.g. ``{'oceTAUX': 'c'}``
    """

    nface = 13
    dtype = np.dtype('>f4')
    # should be implemented by child classes
    nx = None
    nz = None
    delta_t = None
    time_units = None
    iter_start = None
    iter_stop = None
    iter_step = None
    iters = None
    varnames = []
    grid_varnames = []
    mask_override = {}
    domain = 'global'
    pad_before = [0]*_nfacets
    pad_after  = [0]*_nfacets

    def __init__(self, store):
        """Initialize model

        Parameters
        ----------
        store : llcreader.BaseStore
        mask_ds : zarr.Group
            Must contain variables `mask_c`, `mask_w`, `mask_s`
        """
        self.store = store
        self.shape = (self.nz, self.nface, self.nx, self.nx)
        if self.store.shrunk:
            self.masks = self._get_masks()
            from .shrunk_index import all_index_data
            mykey = self.nx if self.domain == 'global' else f'{self.domain}_{self.nx}'
            self.indexes = all_index_data[mykey]
        else:
            self.masks = None
            self.indexes = None

    def _get_masks(self):
        masks = {}
        zgroup = self.store.open_mask_group()
        for point in ['c', 'w', 's']:
            mask_faces = dsa.from_zarr(zgroup['mask_' + point]).astype('bool')
            masks[point] = _faces_to_facets(mask_faces,self.nface)
        return masks

    def _dtype(self,varname=None):
        if isinstance(self.dtype,np.dtype):
            return self.dtype
        elif isinstance(self.dtype,dict):
            return np.dtype(self.dtype[varname])

    def _get_kp1_levels(self,k_levels):
        # determine kp1 levels
        # get borders to all k (center) levels
        # ki used to get Zu, Zl later
        ku = k_levels[1:] + [k_levels[-1] + 1 ]
        kp1 = []
        ki=[]
        for i,(x,y) in enumerate(zip(k_levels,ku)):
            kp1 += [x] if x not in kp1 else []
            kp1 += [y] if y-x==1 else [x+1]


        kp1=np.array(kp1)

        return kp1

    def _make_coords_faces(self, all_iters):

        time = self.delta_t * all_iters
        time_attrs = {'units': self.time_units,
                      'calendar': self.calendar}
        coords = {'face': ('face', np.arange(self.nface)),
                  'i': ('i', np.arange(self.nx)),
                  'i_g': ('i_g', np.arange(self.nx)),
                  'j': ('j', np.arange(self.nx)),
                  'j_g': ('j_g', np.arange(self.nx)),
                  'k': ('k', np.arange(self.nz)),
                  'k_u': ('k_u', np.arange(self.nz)),
                  'k_l': ('k_l', np.arange(self.nz)),
                  'k_p1': ('k_p1', np.arange(self.nz + 1)),
                  'niter': ('time', all_iters),
                  'time': ('time', time, time_attrs)
                 }
        ds = xr.decode_cf(xr.Dataset(coords=coords))

        from ..variables import dimensions
        for d in dimensions:
            if d in ds:
                ds[d].attrs.update(dimensions[d]['attrs'])
        return ds

    def _make_coords_latlon():
        ds = self._make_coords_faces(self)
        return _faces_coords_to_latlon(ds)


    def _dask_array(self, nfacet, varname, iters, klevels, k_chunksize):
        # return a dask array for a single facet
        facet_shape =  _facet_shape(nfacet, self.nx, self.nface)
        time_chunks = (len(iters) * (1,),) if iters is not None else ()
        k_chunks = (tuple([len(c)
                          for c in _chunks(klevels, k_chunksize)]),)
        chunks = time_chunks + k_chunks + tuple([(s,) for s in facet_shape])

        # manually build dask graph
        dsk = {}
        token = tokenize(varname, self.store, nfacet)
        name = '-'.join([varname, token])
        dtype = self._dtype(varname)

        # iters == None for grid variables
        def _key_and_task(n_k, these_klevels, n_iter=None, iternum=None):
            if n_iter is None:
                key = name, n_k, 0, 0, 0
            else:
                key = name, n_iter, n_k, 0, 0, 0
            task = (_get_facet_chunk, self.store, varname, iternum,
                     nfacet, these_klevels, self.nx, self.nz, self.nface,
                     dtype, self.mask_override, self.domain,
                     self.pad_before, self.pad_after)
            return key, task

        if iters is not None:
            for n_iter, iternum in enumerate(iters):

                for n_k, these_klevels in enumerate(_chunks(klevels, k_chunksize)):
                    key, task = _key_and_task(n_k, these_klevels, n_iter, iternum)
                    dsk[key] = task
        else:
            for n_k, these_klevels in enumerate(_chunks(klevels, k_chunksize)):
                key, task = _key_and_task(n_k, these_klevels)
                dsk[key] = task

        return dsa.Array(dsk, name, chunks, dtype)

    def _dask_array_vgrid(self, varname, klevels, k_chunksize):
        # return a dask array for a 1D vertical grid var

        # single chunk for 1D variables
        chunks = ((len(klevels),),)

        # manually build dask graph
        dsk = {}
        token = tokenize(varname, self.store)
        name = '-'.join([varname, token])
        dtype = self._dtype(varname)

        nz = self.nz if _VAR_METADATA[varname]['dims'] != ['k_p1'] else self.nz+1
        task = (_get_1d_chunk, self.store, varname,
                list(klevels), nz, dtype)

        key = name, 0
        dsk[key] = task

        return dsa.Array(dsk, name, chunks, dtype)

    def _get_facet_data(self, varname, iters, klevels, k_chunksize):
        # needs facets to be outer index of nested lists
        dims = _VAR_METADATA[varname]['dims']

        if len(dims)==2:
            klevels = [0,]

        if _is_vgrid(varname):
            data_facets = self._dask_array_vgrid(varname,klevels,k_chunksize)
        else:
            data_facets = [self._dask_array(nfacet, varname, iters, klevels, k_chunksize)
                       for nfacet in range(5)]

        if len(dims)==2:
            # squeeze depth dimension out of 2D variable
            data_facets = [facet[..., 0, :, :, :] for facet in data_facets]

        return data_facets


    def _check_iter_start(self, iter_start):
        if self.iter_start is not None and self.iter_step is not None:
            if (iter_start - self.iter_start) % self.iter_step:
                msg = "Iteration {} may not exist, you may need to change 'iter_start'".format(iter_start)
                warnings.warn(msg, RuntimeWarning)


    def _check_iter_step(self, iter_step):
        if self.iter_step is not None:
            if iter_step % self.iter_step:
                msg = "'iter_step' is not a multiple of {}, meaning some expected timesteps may not be returned".format(self.iter_step)
                warnings.warn(msg, RuntimeWarning)

    def _check_iters(self, iters):
        if self.iters is not None:
            if not set(iters) <= set(self.iters):
                msg = "Some requested iterations may not exist, you may need to change 'iters'"
                warnings.warn(msg, RuntimeWarning)
        
        elif self.iter_start is not None and self.iter_step is not None:
            for iter in iters:
                if (iter - self.iter_start) % self.iter_step:
                    msg = "Some requested iterations may not exist, you may need to change 'iters'"
                    warnings.warn(msg, RuntimeWarning)
                    break
            

    def get_dataset(self, varnames=None, iter_start=None, iter_stop=None,
                    iter_step=None, iters=None, k_levels=None, k_chunksize=1,
                    type='faces', read_grid=True, grid_vars_to_coords=True):
        """
        Create an xarray Dataset object for this model.

        Parameters
        ----------
        *varnames : list of strings, optional
            The variables to include, e.g. ``['Salt', 'Theta']``. Otherwise
            include all known variables.
        iter_start : int, optional
            Starting iteration number. Otherwise use model default.
            Follows standard `range` conventions. (inclusive)
        iter_stop : int, optional
            Stopping iteration number. Otherwise use model default.
            Follows standard `range` conventions. (exclusive)
        iter_step : int, optional
            Iteration number stepsize. Otherwise use model default.
        iters : list of ints, optional
            Specific iteration numbers in a list, possibly with nonuniform spacing.
            Either provide this or the iter parameters above.
        k_levels : list of ints, optional
            Vertical levels to extract. Default is to get them all
        k_chunksize : int, optional
            How many vertical levels per Dask chunk.
        type : {'faces', 'latlon'}, optional
            What type of dataset to create
        read_grid : bool, optional
            Whether to read the grid info
        grid_vars_to_coords : bool, optional
            Whether to promote grid variables to coordinate status

        Returns
        -------
        ds : xarray.Dataset
        """

        def _if_not_none(a, b):
            if a is None:
                return b
            else:
                return a

        user_iter_params = [iter_start, iter_stop, iter_step]
        attribute_iter_params = [self.iter_start, self.iter_stop, self.iter_step]

        # If the user has specified some iter params:
        if any([a is not None for a in user_iter_params]):
            # If iters is also set we have a problem
            if iters is not None:
                raise ValueError("Only `iters` or the parameters `iter_start`, `iters_stop`, "
                                 "and `iter_step` can be provided. Both were provided")
            
            # Otherwise we can override any missing values
            iter_start = _if_not_none(iter_start, self.iter_start)
            iter_stop = _if_not_none(iter_stop, self.iter_stop)
            iter_step = _if_not_none(iter_step, self.iter_step)
            iter_params = [iter_start, iter_stop, iter_step]
            if any([a is None for a in iter_params]):
                raise ValueError("The parameters `iter_start`, `iter_stop`, "
                                 "and `iter_step` must be defined either by the "
                                 "model class or as argument. Instead got %r "
                                 % iter_params)
        
        # Otherwise try loading from the user set iters
        elif iters is not None:
            pass

        # Now have a go at using the attribute derived iteration parameters 
        elif all([a is not None for a in attribute_iter_params]):
            iter_params = attribute_iter_params

        # Now try using the attribute derived iters
        elif self.iters is not None:
            iters = self.iters

        # Now give up
        else:
            raise ValueError("The parameters `iter_start`, `iter_stop`, "
                             "and `iter_step`, or `iters` must be defined either by the "
                             "model class or as argument")
        
        # Check the iter_start and iter_step
        if iters is None:
            self._check_iter_start(iter_params[0])
            self._check_iter_step(iter_params[2])
            iters = np.arange(*iter_params)
        else:
            self._check_iters(iters)
            iters = np.array(iters)

        varnames = varnames or self.varnames

        # grid stuff
        read_grid = read_grid and len(self.grid_varnames)!=0
        if read_grid and self.store.grid_path is None:
            raise TypeError('Cannot read grid if grid_path is not specified in filestore (e.g. llcreader.known_models)')
        grid_vars_to_coords = grid_vars_to_coords and read_grid
        grid_varnames = self.grid_varnames if read_grid else []

        ds = self._make_coords_faces(iters)
        if type=='latlon':
            if self.domain=='aste':
                raise TypeError('Swapping to lat/lon not available for ASTE. Must regrid or interpolate.')
            ds = _faces_coords_to_latlon(ds)

        k_levels = k_levels or list(range(self.nz))
        kp1_levels = self._get_kp1_levels(k_levels)

        ds = ds.sel(k=k_levels, k_l=k_levels, k_u=k_levels, k_p1=kp1_levels)

        # get the data in facet form
        data_facets = {vname:
                       self._get_facet_data(vname, iters, k_levels, k_chunksize)
                       for vname in varnames}

        # get the grid in facet form
        # do separately for vertical coords on kp1_levels
        grid_facets = {}
        for vname in grid_varnames:
            my_k_levels = k_levels if _VAR_METADATA[vname]['dims'] !=['k_p1'] else kp1_levels
            grid_facets[vname] = self._get_facet_data(vname, None, my_k_levels, k_chunksize)

        # transform it into faces or latlon
        data_transformers = {'faces': _all_facets_to_faces,
                             'latlon': _all_facets_to_latlon}

        transformer = data_transformers[type]
        data = transformer(data_facets, _VAR_METADATA, self.nface)

        # separate horizontal and vertical grid variables
        hgrid_facets = {key: grid_facets[key]
                for key in grid_varnames if not _is_vgrid(key)}
        vgrid_facets = {key: grid_facets[key]
                for key in grid_varnames if _is_vgrid(key)}

        # do not transform vertical grid variables
        data.update(transformer(hgrid_facets, _VAR_METADATA, self.nface))
        data.update(vgrid_facets)

        variables = {}
        gridlist = ['Zl','Zu'] if read_grid else []
        for vname in varnames+grid_varnames:
            meta = _VAR_METADATA[vname]
            dims = meta['dims']
            if type=='faces':
                dims = _add_face_to_dims(dims)
            dims = ['time',] + dims if vname not in grid_varnames else dims
            attrs = meta['attrs']

            # Handle grid names different from filenames
            fname = vname
            vname = meta['real_name'] if 'real_name' in meta else vname
            if fname in grid_varnames:
                gridlist.append(vname)

            variables[vname] = xr.Variable(dims, data[fname], attrs)

        # handle vertical coordinate after the fact
        if read_grid and 'RF' in grid_varnames:
            ki = np.array([list(kp1_levels).index(x) for x in k_levels])
            for zv,sl in zip(['Zl','Zu'],[ki,ki+1]):
                variables[zv] = xr.Variable(_VAR_METADATA[zv]['dims'],
                                            data['RF'][sl],
                                            _VAR_METADATA[zv]['attrs'])

        ds = ds.update(variables)

        if grid_vars_to_coords:
            ds = ds.set_coords(gridlist)

        return ds
