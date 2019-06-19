import numpy as np
import dask
import dask.array as dsa
from dask.base import tokenize, normalize_token
import xarray as xr
import warnings

from .duck_array_ops import concatenate
from .shrunk_index import all_index_data

def _get_var_metadata():
    # The LLC run data comes with zero metadata. So we import metadata from
    # the xmitgcm package.
    from ..variables import state_variables, package_state_variables
    from ..utils import parse_available_diagnostics
    from ..default_diagnostics import diagnostics
    from io import StringIO

    diag_file = StringIO(diagnostics)
    available_diags = parse_available_diagnostics(diag_file)
    var_metadata = state_variables
    var_metadata.update(package_state_variables)
    var_metadata.update(available_diags)

    # even the file names from the LLC data differ from standard MITgcm output
    aliases = {'Eta': 'ETAN', 'PhiBot': 'PHIBOT', 'Salt': 'SALT',
               'Theta': 'THETA'}
    for a, b in aliases.items():
        var_metadata[a] = var_metadata[b]

    return var_metadata

_VAR_METADATA = _get_var_metadata()

def _get_variable_point(vname):
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



_facet_strides = ((0,3), (3,6), (6,7), (7,10), (10,13))
# whether to reshape each face
_facet_reshape = (False, False, False, True, True)
_nfaces = 13
_nfacets = 5

def _uncompressed_facet_index(nfacet, nside):
    face_size = nside**2
    start = _facet_strides[nfacet][0] * face_size
    end = _facet_strides[nfacet][1] * face_size
    return start, end

def _facet_shape(nfacet, nside):
    facet_length = _facet_strides[nfacet][1] - _facet_strides[nfacet][0]
    if _facet_reshape[nfacet]:
        facet_shape = (1, nside, facet_length*nside)
    else:
        facet_shape = (1, facet_length*nside, nside)
    return facet_shape

def _facet_to_faces(data, nfacet):
    shape = data.shape
    # facet dimension
    nf, ny, nx = shape[-3:]
    other_dims = shape[:-3]
    assert nf == 1
    facet_length = _facet_strides[nfacet][1] - _facet_strides[nfacet][0]
    if _facet_reshape[nfacet]:
        new_shape = other_dims +  (ny, facet_length, nx / facet_length)
        data_rs = data.reshape(new_shape)
        data_rs = np.moveaxis(data_rs, -2, -3) # dask-safe
    else:
        new_shape = other_dims + (facet_length, ny / facet_length, nx)
        data_rs = data.reshape(new_shape)
    return data_rs

def _facets_to_faces(facets):
    all_faces = []
    for nfacet, data_facet in enumerate(facets):
        data_rs = _facet_to_faces(data_facet, nfacet)
        all_faces.append(data_rs)
    return concatenate(all_faces, axis=-3)

def _faces_to_facets(data, facedim=-3):
    assert data.shape[facedim] == _nfaces
    facets = []
    for nfacet, (strides, reshape) in enumerate(zip(_facet_strides, _facet_reshape)):
        face_data = [data[(...,) + (slice(nface, nface+1), slice(None), slice(None))]
                     for nface in range(*strides)]
        if reshape:
            concat_axis = facedim + 2
        else:
            concat_axis = facedim + 1
        # todo: use duck typing for concat
        facet_data = concatenate(face_data, axis=concat_axis)
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


def _faces_to_latlon_scalar(data):
    data_facets = _faces_to_facets(data)
    return _facets_to_latlon_scalar(data_facets)


# dask's pad function doesn't work
# it does weird things to non-pad dimensions
# need to roll our own
def shift_and_pad(a):
    a_shifted = a[..., 1:]
    pad_array = dsa.zeros_like(a[..., -2:-1])
    return concatenate([a_shifted, pad_array], axis=-1)

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
             + [transform_u_to_v(facet) for facet in facets_u_drop[-2:]])

    u = concatenate(u_rot, axis=-1)
    v = concatenate(v_rot, axis=-1)
    return u, v

def _faces_to_latlon_vector(u_faces, v_faces, metric=False):
    u_facets = _faces_to_facets(u_faces)
    v_facets = _faces_to_facets(v_faces)
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
            vector_pairs.append((vname, mate))
            vnames.remove(mate)
        except KeyError:
            pass

    all_vector_components = [inner for outer in (vector_pairs + metric_vector_pairs)
                             for inner in outer]
    scalars = [vname for vname in vnames if vname not in all_vector_components]
    data_vars = {}

    for vname in scalars:
        if vname=='face' or vname in ds_new:
            continue
        if 'face' in ds[vname].dims:
            data = _faces_to_latlon_scalar(ds[vname].data)
            dims = _drop_facedim(ds[vname].dims)
        else:
            data = ds[vname].data
            dims = ds[vname].dims
        data_vars[vname] = xr.Variable(dims, data, ds[vname].attrs)

    for vname_u, vname_v in vector_pairs:
        data_u, data_v = _faces_to_latlon_vector(ds[vname_u].data, ds[vname_v].data)
        data_vars[vname_u] = xr.Variable(_drop_facedim(ds[vname_u].dims), data_u, ds[vname_u].attrs)
        data_vars[vname_v] = xr.Variable(_drop_facedim(ds[vname_v].dims), data_v, ds[vname_v].attrs)
    for vname_u, vname_v in metric_vector_pairs:
        data_u, data_v = _faces_to_latlon_vector(ds[vname_u].data, ds[vname_v].data, metric=True)
        data_vars[vname_u] = xr.Variable(_drop_facedim(ds[vname_u].dims), data_u, ds[vname_u].attrs)
        data_vars[vname_v] = xr.Variable(_drop_facedim(ds[vname_v].dims), data_v, ds[vname_v].attrs)


    ds_new = ds_new.update(data_vars)
    ds_new = ds_new.set_coords([c for c in coord_vars if c in ds_new])
    return ds_new


# below are data transformers

def _all_facets_to_faces(data_facets, meta):
    return {vname: _facets_to_faces(data)
            for vname, data in data_facets.items()}


def _all_facets_to_latlon(data_facets, meta):

    vector_pairs = []
    scalars = []
    vnames = list(data_facets)
    for vname in vnames:
        try:
            mate = meta[vname]['attrs']['mate']
            vector_pairs.append((vname, mate))
            vnames.remove(mate)
        except KeyError:
            pass

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


def _get_facet_chunk(store, varname, iternum, nfacet, klevels, nx, nz, dtype):
    fs, path = store.get_fs_and_full_path(varname, iternum)
    file = fs.open(path)

    assert (nfacet >= 0) & (nfacet < _nfacets)

    try:
        # workaround for ecco data portal
        file = fs.open(path, size_policy='get')
    except TypeError:
        file = fs.open(path)

    # insert singleton axis for time and k level
    facet_shape = (1, 1,) + _facet_shape(nfacet, nx)

    level_data = []

    # TODO: get index
    # the store tells us whether we need a mask or not
    point = _get_variable_point(varname)
    if store.shrunk:
        index = all_index_data[nx][point]
        zgroup = store.open_mask_group()
        mask = zgroup['mask_' + point].astype('bool')
    else:
        index = None
        mask = None

    for k in klevels:
        assert (k >= 0) & (k < nz)

        # figure out where in the file we have to read to get the data
        # for this level and facet
        if index:
            i = np.ravel_multi_index((k, nfacet), (nz, _nfacets))
            start = index[i]
            end = index[i+1]
        else:
            level_start = k * nx**2 * _nfaces
            facet_start, facet_end = _uncompressed_facet_index(nfacet, nx)
            start = level_start + facet_start
            end = level_start + facet_end

        read_offset = start * dtype.itemsize # in bytes
        read_length  = (end - start) * dtype.itemsize # in bytes
        file.seek(read_offset)
        buffer = file.read(read_length)
        data = np.frombuffer(buffer, dtype=dtype)
        assert len(data) == (end - start)

        if mask:
            mask_level = mask[k]
            mask_facets = _faces_to_facets(mask_level)
            this_mask = mask_facets[nfacet]
            data = _decompress(data, this_mask, dtype)

        # this is the shape this facet is supposed to have
        data.shape = facet_shape
        level_data.append(data)

    return np.concatenate(level_data, axis=1)


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
    varnames : list
        List of variable names contained in the dataset
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
    varnames = []

    def __init__(self, store):
        """Initialize model

        Parameters
        ----------
        store : llcreader.BaseStore
        mask_ds : zarr.Group
            Must contain variables `mask_c`, `masc_w`, `mask_s`
        """
        self.store = store
        self.shape = (self.nz, self.nface, self.nx, self.nx)
        if self.store.shrunk:
            self.masks = self._get_masks()
            from .shrunk_index import all_index_data
            self.indexes = all_index_data[self.nx]
        else:
            self.masks = None
            self.indexes = None


    def _get_masks(self):
        masks = {}
        zgroup = self.store.open_mask_group()
        for point in ['c', 'w', 's']:
            mask_faces = dsa.from_zarr(zgroup['mask_' + point]).astype('bool')
            masks[point] = _faces_to_facets(mask_faces)
        return masks


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
        return xr.decode_cf(xr.Dataset(coords=coords))


    def _make_coords_latlon():
        ds = self._make_coords_faces(self)
        return _faces_coords_to_latlon(ds)


    def _get_mask_and_index_for_variable(self, vname):
        if self.masks is None:
            return None, None

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
            # this is not a 2D variable
            return None, None

        mask = self.masks[point]
        index = self.indexes[point]
        return mask, index


    def _dask_array(self, nfacet, varname, iters, klevels, k_chunksize):
        # return a dask array for a single facet
        facet_shape =  _facet_shape(nfacet, self.nx)
        time_chunks = (len(iters) * (1,),)
        k_chunks = (tuple([len(c)
                          for c in _chunks(klevels, k_chunksize)]),)
        chunks = time_chunks + k_chunks + tuple([(s,) for s in facet_shape])

        # manually build dask graph
        dsk = {}
        token = tokenize(varname, self.store, nfacet)
        name = '-'.join([varname, token])
        for n_iter, iternum in enumerate(iters):
            for n_k, these_klevels in enumerate(_chunks(klevels, k_chunksize)):
                key = name, n_iter, n_k, 0, 0, 0
                task = (_get_facet_chunk, self.store, varname, iternum,
                         nfacet, these_klevels, self.nx, self.nz, self.dtype)
                dsk[key] = task

        return dsa.Array(dsk, name, chunks, self.dtype)


    def _get_facet_data(self, varname, iters, klevels, k_chunksize):
        mask, index = self._get_mask_and_index_for_variable(varname)
        # needs facets to be outer index of nested lists
        dims = _VAR_METADATA[varname]['dims']

        if len(dims)==2:
            klevels = [0,]

        data_facets = [self._dask_array(nfacet, varname, iters, klevels, k_chunksize)
                       for nfacet in range(5)]

        if len(dims)==2:
            # squeeze depth dimension out of 2D variable
            data_facets = [facet[..., 0, :, :, :] for facet in data_facets]

        return data_facets


    def get_dataset(self, varnames=None, iter_start=None, iter_stop=None,
                    iter_step=None, k_levels=None, k_chunksize=1,
                    type='faces'):
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
        iter_start : int, optional
            Stopping iteration number. Otherwise use model default.
            Follows standard `range` conventions. (exclusive)
        iter_step : int, optional
            Iteration number stepsize. Otherwise use model default.
        k_levels : list of ints, optional
            Vertical levels to extract. Default is to get them all
        k_chunksize : int, optional
            How many vertical levels per Dask chunk.
        type : {'faces', 'latlon'}, optional
            What type of dataset to create

        Returns
        -------
        ds : xarray.Dataset
        """

        def _if_not_none(a, b):
            if a is None:
                return b
            else:
                return a

        iter_start = _if_not_none(iter_start, self.iter_start)
        iter_stop = _if_not_none(iter_stop, self.iter_stop)
        iter_step = _if_not_none(iter_step, self.iter_step)
        iter_params = [iter_start, iter_stop, iter_step]
        if any([a is None for a in iter_params]):
            raise ValueError("The parameters `iter_start`, `iter_stop` "
                             "and `iter_step` must be defined either by the "
                             "model class or as argument. Instead got %r "
                             % iter_params)
        iters = np.arange(*iter_params)

        varnames = varnames or self.varnames

        ds = self._make_coords_faces(iters)
        if type=='latlon':
            ds = _faces_coords_to_latlon(ds)

        k_levels = k_levels or np.arange(self.nz)
        ds = ds.sel(k=k_levels, k_l=k_levels, k_u=k_levels, k_p1=k_levels)

        # get the data in facet form
        data_facets = {vname:
                       self._get_facet_data(vname, iters, k_levels, k_chunksize)
                       for vname in varnames}

        # transform it into faces or latlon
        data_transformers = {'faces': _all_facets_to_faces,
                             'latlon': _all_facets_to_latlon}

        transformer = data_transformers[type]
        data = transformer(data_facets, _VAR_METADATA)

        variables = {}
        for vname in varnames:
            meta = _VAR_METADATA[vname]
            dims = meta['dims']
            if type=='faces':
                dims = _add_face_to_dims(dims)
            dims = ['time',] + dims
            attrs = meta['attrs']
            variables[vname] = xr.Variable(dims, data[vname], attrs)

        ds = ds.update(variables)
        return ds
