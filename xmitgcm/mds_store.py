"""
Class to represent MITgcm mds file storage format.
"""
# python 3 compatiblity
from __future__ import print_function, division

from glob import glob
import os
import re
import numpy as np
import warnings
from io import StringIO
import inspect
import xarray as xr
import dask.array as da
import sys

# we keep the metadata in its own module to keep this one cleaner
from .variables import dimensions, \
    horizontal_coordinates_spherical, horizontal_coordinates_cartesian, \
    horizontal_coordinates_curvcart, horizontal_coordinates_llc, \
    horizontal_coordinates_cs, \
    vertical_coordinates, horizontal_grid_variables, vertical_grid_variables, \
    volume_grid_variables, state_variables, aliases, package_state_variables, \
    extra_grid_variables, mask_variables
# would it be better to import mitgcm_variables and then automate the search
# for variable dictionaries

from .utils import parse_meta_file, read_mds, parse_available_diagnostics,\
    get_extra_metadata

from .file_utils import listdir, listdir_startswith, listdir_endswith, \
    listdir_startsandendswith, listdir_fnmatch

# Python2/3 compatibility
if (sys.version_info > (3, 0)):
    stringtypes = [str]
else:
    stringtypes = [str, unicode]

# xarray>=0.12.0 compatiblity
try:
    from xarray.core.pycompat import OrderedDict
except ImportError:
    from collections import OrderedDict

# should we hard code this?
LLC_NUM_FACES = 13
CS_NUM_FACES = 6
FACE_DIMNAME = 'face'


def open_mdsdataset(data_dir, grid_dir=None,
                    iters='all', prefix=None, read_grid=True,
                    delta_t=1, ref_date=None, calendar='gregorian',
                    levels=None, geometry='sphericalpolar',
                    grid_vars_to_coords=True, swap_dims=None,
                    endian=">", chunks=None,
                    ignore_unknown_vars=False, default_dtype=None,
                    nx=None, ny=None, nz=None,
                    llc_method="smallchunks", extra_metadata=None):
    """Open MITgcm-style mds (.data / .meta) file output as xarray datset.

    Parameters
    ----------
    data_dir : string
        Path to the directory where the mds .data and .meta files are stored
    grid_dir : string, optional
        Path to the directory where the mds .data and .meta files are stored, if
        different from ``data_dir``.
    iters : list, optional
        The iterations numbers of the files to be read. If ``None``, no data
        files will be read. If ``'all'`` (default), all iterations will be read.
    prefix : list, optional
        List of different filename prefixes to read. Default (``None``) is to
        read all available files.
    read_grid : bool, optional
        Whether to read the grid data
    delta_t : number, optional
        The timestep used in the model. (Can't be inferred.)
    ref_date : string, optional
        An iSO date string corresponding to the zero timestep,
        e.g. "1990-1-1 0:0:0" (See CF conventions [1]_)
    calendar : string, optional
        A calendar allowed by CF conventions [1]_
    levels : list or slice, optional
        A list or slice of the indexes of the grid levels to read
        Same syntax as in the data.diagnostics file
    geometry : {'sphericalpolar', 'cartesian', 'llc', 'curvilinear', 'cs'}
        MITgcm grid geometry specifier
    grid_vars_to_coords : boolean, optional
        Whether to promote grid variables to coordinate status
    swap_dims : boolean, optional
        Whether to swap the logical dimensions for physical ones. If ``None``,
        will be set to ``False`` for ``geometry==llc`` and ``True`` otherwise.
    endian : {'=', '>', '<'}, optional
        Endianness of variables. Default for MITgcm is ">" (big endian)
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask arrays.
    ignore_unknown_vars : boolean, optional
        Don't raise an error if unknown variables are encountered while reading
        the dataset.
    default_dtype : numpy.dtype, optional
        A datatype to fall back on if the metadata can't be read.
    nx, ny, nz : int, optional
        The numerical dimensions of the model. These will be inferred from
        ``XC.meta`` and ``RC.meta`` if they are not specified. If
        ``geometry==llc``, ``ny`` does not have to specified.
    llc_method : {"smallchunks", "bigchunks"}, optional
        Which routine to use for reading LLC data. "smallchunks" splits the file
        into a individual dask chunk of size (nx x nx) for each face of each
        level (i.e. the total number of chunks is 13 * nz). "bigchunks" loads
        the whole raw data file (either into memory or as a numpy.memmap),
        splits it into faces, and concatenates those faces together using
        ``dask.array.concatenate``. The different methods will have different
        memory and i/o performance depending on the details of the system
        configuration.
    extra_metadata : dict, optional
        Allow to pass information on llc type grid (global or regional).
        The additional metadata is typically such as :

        aste = {'has_faces': True, 'ny': 1350, 'nx': 270,
                'ny_facets': [450,0,270,180,450],
                'pad_before_y': [90,0,0,0,0],
                'pad_after_y': [0,0,0,90,90],
                'face_facets': [0, 0, 2, 3, 4, 4],
                'facet_orders' : ['C', 'C', 'C', 'F', 'F'],
                'face_offsets' : [0, 1, 0, 0, 0, 1],
                'transpose_face' : [False, False, False,
                                    True, True, True]}

        For global llc grids, no extra metadata is required and code
        will set up to global llc default configuration.

    Returns
    -------
    dset : xarray.Dataset
        Dataset object containing all coordinates and variables.

    References
    ----------
    .. [1] http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch04s04.html
    """

    # get frame info for history
    frame = inspect.currentframe()
    _, _, _, arg_values = inspect.getargvalues(frame)
    del arg_values['frame']
    function_name = inspect.getframeinfo(frame)[2]

    # auto-detect whether to swap dims
    if swap_dims is None:
        if read_grid == False:
            swap_dims = False
        else:
            swap_dims = False if geometry in (
                'llc', 'cs', 'curvilinear') else True

    # some checks for argument consistency
    if swap_dims and not read_grid:
        raise ValueError("If swap_dims==True, read_grid must be True.")

    # if prefix is passed as a string, force it to be a list
    if type(prefix) in stringtypes:
        prefix = [prefix]
    else:
        pass

    # if levels s a slice or a list, a subset of levels is needed
    if levels is not None and nz is not None:
        warnings.warn('levels has been set, nz will be ignored.')
        nz = None
    if isinstance(levels, slice):
        levels = np.arange(levels.start, levels.stop)

    # We either have a single iter, in which case we create a fresh store,
    # or a list of iters, in which case we combine.
    if iters == 'all':
        iters = _get_all_iternums(data_dir, file_prefixes=prefix)
    if iters is None:
        iternum = None
    else:
        try:
            iternum = int(iters)
        # if not we probably have some kind of list
        except TypeError:
            if len(iters) == 1 and levels is None:
                iternum = int(iters[0])
            else:
                # We have to check to make sure we have the same prefixes at
                # each timestep...otherwise we can't combine the datasets.
                first_prefixes = prefix or _get_all_matching_prefixes(
                                                        data_dir, iters[0])
                for iternum in iters:
                    these_prefixes = _get_all_matching_prefixes(
                        data_dir, iternum, prefix
                    )
                    # don't care about order
                    if set(these_prefixes) != set(first_prefixes):
                        raise IOError("Could not find the expected file "
                                      "prefixes %s at iternum %g. (Instead "
                                      "found %s)" % (repr(first_prefixes),
                                                     iternum,
                                                     repr(these_prefixes)))

                # chunk at least by time
                chunks = chunks or {}

                # recursively open each dataset at a time
                kwargs = dict(
                    grid_dir=grid_dir, delta_t=delta_t, swap_dims=False,
                    prefix=prefix, ref_date=ref_date, calendar=calendar,
                    geometry=geometry,
                    grid_vars_to_coords=False,
                    endian=endian, chunks=chunks,
                    ignore_unknown_vars=ignore_unknown_vars,
                    default_dtype=default_dtype,
                    nx=nx, ny=ny, nz=nz, llc_method=llc_method,
                    levels=levels, extra_metadata=extra_metadata)
                datasets = [open_mdsdataset(
                        data_dir, iters=iternum, read_grid=False, **kwargs)
                    for iternum in iters]
                # now add the grid
                if read_grid:
                    if 'iters' in kwargs:
                        kwargs.pop('iters')
                    if 'read_grid' in kwargs:
                        kwargs.pop('read_grid')
                    if levels is not None:
                        kwargs.pop('nz')
                        kwargs.pop('levels')
                    grid_dataset = open_mdsdataset(data_dir, iters=None,
                                                   read_grid=True, **kwargs)
                    if levels is not None:
                        grid_dataset = grid_dataset.isel(**{coord: levels
                                    for coord in ['k', 'k_l', 'k_u', 'k_p1']})
                    datasets.insert(0, grid_dataset)
                # apply chunking
                if sys.version_info[0] < 3:
                    ds = xr.auto_combine(datasets)
                elif xr.__version__ < '0.15.2':
                    ds = xr.combine_by_coords(datasets)
                else:
                    ds = xr.combine_by_coords(datasets, compat='override', coords='minimal', combine_attrs='drop')

                if swap_dims:
                    ds = _swap_dimensions(ds, geometry)
                if grid_vars_to_coords:
                    ds = _set_coords(ds)
                return ds

    store = _MDSDataStore(data_dir, grid_dir, iternum, delta_t, read_grid,
                          prefix, ref_date, calendar,
                          geometry, endian,
                          ignore_unknown_vars=ignore_unknown_vars,
                          default_dtype=default_dtype,
                          nx=nx, ny=ny, nz=nz, llc_method=llc_method,
                          levels=levels, extra_metadata=extra_metadata)
    ds = xr.Dataset.load_store(store)
    if swap_dims:
        ds = _swap_dimensions(ds, geometry)
    if grid_vars_to_coords:
        ds = _set_coords(ds)

    if 'time' in ds:
        ds['time'] = xr.decode_cf(ds[['time']])['time']

    # do we need more fancy logic (like open_dataset), or is this enough
    if chunks is not None:
        ds = ds.chunk(chunks)

    # set attributes for CF conventions
    ds.attrs['Conventions'] = "CF-1.6"
    ds.attrs['title'] = "netCDF wrapper of MITgcm MDS binary data"
    ds.attrs['source'] = "MITgcm"
    arg_string = ', '.join(['%s=%s' % (str(k), repr(v))
                            for (k, v) in arg_values.items()])
    ds.attrs['history'] = ('Created by calling '
                           '`%s(%s)`'% (function_name, arg_string))

    return ds


def _set_coords(ds):
    """Turn all variables without `time` dimensions into coordinates."""
    coords = set()
    for vname in ds.variables:
        if ('time' not in ds[vname].dims) or (ds[vname].dims == ('time',)):
            coords.add(vname)
    return ds.set_coords(list(coords))


def _swap_dimensions(ds, geometry, drop_old=True):
    """Replace logical coordinates with physical ones. Does not work for llc or cs.
    """
    keep_attrs = ['axis', 'c_grid_axis_shift']

    # this fixes problems
    ds = ds.reset_coords()

    if geometry.lower() in ('llc', 'cs', 'curvilinear'):
        raise ValueError("Can't swap dimensions if `geometry` is `llc` or `cs`")

    # first squeeze all the coordinates
    for orig_dim in ds.dims:
        if 'swap_dim' in ds[orig_dim].attrs:
            new_dim = ds[orig_dim].attrs['swap_dim']
            coord_var = ds[new_dim]
            for coord_dim in coord_var.dims:
                if coord_dim != orig_dim:
                    # dimension should be the same along all other axes, so just
                    # take the first row / column
                    coord_var = coord_var.isel(**{coord_dim: 0}).drop(coord_dim)
            ds[new_dim] = coord_var
            for key in keep_attrs:
                if key in ds[orig_dim].attrs:
                    ds[new_dim].attrs[key] = ds[orig_dim].attrs[key]
    # then swap dims
    for orig_dim in ds.dims:
        if 'swap_dim' in ds[orig_dim].attrs:
            new_dim = ds[orig_dim].attrs['swap_dim']
            ds = ds.swap_dims({orig_dim:  new_dim})
            if drop_old:
                if sys.version_info[0] < 3:
                    ds = ds.drop(orig_dim)
                else:
                    ds = ds.drop_vars(orig_dim)
    return ds


class _MDSDataStore(xr.backends.common.AbstractDataStore):
    """Representation of MITgcm mds binary file storage format for a specific
    model instance and a specific timestep iteration number."""
    def __init__(self, data_dir, grid_dir=None,
                 iternum=None, delta_t=1, read_grid=True,
                 file_prefixes=None, ref_date=None, calendar=None,
                 geometry='sphericalpolar',
                 endian='>', ignore_unknown_vars=False,
                 default_dtype=np.dtype('f4'),
                 nx=None, ny=None, nz=None, llc_method="smallchunks",
                 levels=None, extra_metadata=None):
        """
        This is not a user-facing class. See open_mdsdataset for argument
        documentation. The only ones which are distinct are.

        Parameters
        ----------
        iternum : int, optional
            The iteration timestep number to read.
        file_prefixes : list
            The prefixes of the data files to be read.
        """

        self.geometry = geometry.lower()
        allowed_geometries = ['cartesian',
                              'sphericalpolar', 'llc', 'cs', 'curvilinear']
        if self.geometry not in allowed_geometries:
            raise ValueError('Unexpected value for parameter `geometry`. '
                             'It must be one of the following: %s' %
                             allowed_geometries)

        # the directory where the files live
        self.data_dir = data_dir
        self.grid_dir = grid_dir if (grid_dir is not None) else data_dir
        self._ignore_unknown_vars = ignore_unknown_vars

        # The endianness of the files
        # By default, MITgcm does big endian
        if endian not in ['>', '<', '=']:
            raise ValueError("Invalid byte order (endian=%s)" % endian)
        self.endian = endian
        if default_dtype is not None:
            self.default_dtype = np.dtype(default_dtype).newbyteorder(endian)
        else:
            self.default_dtype = default_dtype

        # storage dicts for variables and attributes
        self._variables = OrderedDict()
        self._attributes = OrderedDict()
        self._dimensions = []

        # the dimensions are theoretically the same for all datasets
        [self._dimensions.append(k) for k in dimensions]
        self.llc = (self.geometry == 'llc')
        self.cs = (self.geometry == 'cs')

        if nz is None:
            self.nz = _guess_model_nz(self.grid_dir)
        else:
            self.nz = nz

        # if user passes extra_metadata, this should have priority
        user_metadata = True if extra_metadata is not None else False

        # put in local variable to make it more readable
        if extra_metadata is not None and 'has_faces' in extra_metadata:
            has_faces = extra_metadata['has_faces']
        else:
            has_faces = False

        # --------------- LEGACY ----------------------
        if self.llc:
            has_faces = True
            if extra_metadata is None or 'ny_facets' not in extra_metadata:
                # default to llc90, we only need number of facets
                # and we cannot know nx at this point
                llc = get_extra_metadata(domain='llc', nx=90)
                extra_metadata = llc
        if self.cs:
            has_faces = True
            if extra_metadata is None or 'ny_facets' not in extra_metadata:
                # default to llc90, we only need number of facets
                # and we cannot know nx at this point
                cs = get_extra_metadata(domain='cs', nx=32)
                extra_metadata = cs
        # --------------- /LEGACY ----------------------

        # we don't need to know ny if using llc
        if has_faces and (nx is not None):
            ny = nx

        # Now we need to figure out the horizontal dimensions nx, ny
        # nface is the number of llc faces
        if (nx is not None) and (ny is not None):
            # we have been passed enough information to determine the
            # dimensions without reading any files
            self.ny, self.nx = ny, nx
            self.nface = len(extra_metadata['face_facets']) if has_faces \
                else None
        else:
            # have to peek at the grid file metadata
            self.nface, self.ny, self.nx = (
                _guess_model_horiz_dims(self.grid_dir, is_llc=self.llc, is_cs=self.cs))

        # --------------- LEGACY ----------------------
        if self.llc:
            if not user_metadata:
                # if user didn't provide metadata, we default to llc
                llc = get_extra_metadata(domain='llc', nx=self.nx)
                extra_metadata = llc
        if self.cs:
            if not user_metadata:
                # if user didn't provide metadata, we default to llc
                cs = get_extra_metadata(domain='cs', nx=self.nx)
                extra_metadata = cs
        # --------------- /LEGACY ----------------------

        self.layers = _guess_layers(data_dir)

        if has_faces:
            nyraw = self.nx*self.nface
        else:
            nyraw = self.ny
        self.default_shape_3D = (self.nz, nyraw, self.nx)
        self.default_shape_2D = (nyraw, self.nx)
        self.llc_method=llc_method

        # Now set up the corresponding coordinates.
        # Rather than assuming the dimension names, we use Comodo conventions
        # to parse the dimension metdata.
        # http://pycomodo.forge.imag.fr/norm.html
        irange = np.arange(self.nx)
        jrange = np.arange(self.ny)
        if levels is None:
            krange = np.arange(self.nz)
            krange_p1 = np.arange(self.nz+1)
        else:
            krange = levels
            krange_p1 = levels + [levels[-1] + 1]
        # the keys are `standard_name` attribute
        dimension_data = {
            "x_grid_index": irange,
            "x_grid_index_at_u_location": irange,
            "x_grid_index_at_f_location": irange,
            "y_grid_index": jrange,
            "y_grid_index_at_v_location": jrange,
            "y_grid_index_at_f_location": jrange,
            "z_grid_index": krange,
            "z_grid_index_at_lower_w_location": krange,
            "z_grid_index_at_upper_w_location": krange,
            "z_grid_index_at_w_location": krange_p1,
        }

        for dim in self._dimensions:
            dim_meta = dimensions[dim]
            dims = dim_meta['dims']
            attrs = dim_meta['attrs']
            data = dimension_data[attrs['standard_name']]
            dim_variable = xr.Variable(dims, data, attrs)
            self._variables[dim] = dim_variable

        # possibly add the llc dimension
        # seems sloppy to hard code this here
        # TODO: move this metadata to variables.py
        if has_faces:
            self._dimensions.append(FACE_DIMNAME)
            data = np.arange(self.nface)
            attrs = {'standard_name': 'face_index'}
            dims = [FACE_DIMNAME]
            self._variables[FACE_DIMNAME] = xr.Variable(dims, data, attrs)

        # do the same for layers
        for layer_name, n_layer in self.layers.items():
            for suffix, offset in zip(['bounds', 'center', 'interface'],
                                      [0, -1, -2]):
                # e.g. "layer_1RHO_bounds"
                # dimname = 'layer_' + layer_name + '_' + suffix
                # e.g. "l1_b"
                dimname = 'l' + layer_name[0] + '_' + suffix[0]
                self._dimensions.append(dimname)
                data = np.arange(n_layer + offset)
                # we should figure out a way to properly populate the layers
                # attributes
                attrs = {'standard_name':
                         layer_name + '_layer_grid_index_at_layer_' + suffix,
                         'swap_dim': 'layer_' + layer_name + '_' + suffix}
                dim_variable = xr.Variable([dimname], data, attrs)
                self._variables[dimname] = dim_variable

        # maybe add a time dimension
        if iternum is not None:
            self.time_dim_name = 'time'
            self._dimensions.append(self.time_dim_name)
            # a variable for iteration number
            self._variables['iter'] = xr.Variable(
                        (self.time_dim_name,),
                        [iternum],
                        {'standard_name': 'timestep',
                         'long_name': 'model timestep number'})
            self._variables[self.time_dim_name] = _iternum_to_datetime_variable(
                iternum, delta_t, ref_date, calendar, self.time_dim_name
            )

        # build lookup tables for variable metadata
        self._all_grid_variables = _get_all_grid_variables(self.geometry,
                                                           self.grid_dir,
                                                           self.layers)
        self._all_data_variables = _get_all_data_variables(self.data_dir,
                                                           self.grid_dir,
                                                           self.layers)

        # The rest of the data has to be read from disk.
        # The list `prefixes` specifies file prefixes from which to infer
        # The problem with this is that some prefixes are single variables
        # while some are multi-variable diagnostics files.
        prefixes = []
        if read_grid:
            prefixes = prefixes + list(self._all_grid_variables.keys())

        # add data files
        prefixes = (prefixes +
                    _get_all_matching_prefixes(
                        data_dir,
                        iternum,
                        file_prefixes))

        for p in prefixes:
            # use a generator to loop through the variables in each file
            for (vname, dims, data, attrs) in \
                    self.load_from_prefix(p, iternum, extra_metadata):
                # print(vname, dims, data.shape)
                # Sizes of grid variables can vary between mitgcm versions.
                # Check for such inconsistency and correct if so
                (vname, dims, data, attrs) = self.fix_inconsistent_variables(
                    vname, dims, data, attrs)

                # Create masks from hFac variables
                data = self.calc_masks(vname, data)

                thisvar = xr.Variable(dims, data, attrs)
                self._variables[vname] = thisvar
                # print(type(data), type(thisvar._data), thisvar._in_memory)

    def fix_inconsistent_variables(self, vname, dims, data, attrs):
        if vname == 'drC':
            #check to see if the drC variable has the wrong length
            if len(data)==self.nz:
                #create a new array which will replace it
                drc_data = np.zeros(self.nz + 1)
                drc_data[:-1] = np.asarray(data)
                #fill in the missing value
                drc_data[-1] = 0.5 * data[-1]
                data = drc_data
        return vname, dims, data, attrs

    def calc_masks(self, vname, data):
        """Compute mask as True where hFac nonzero, otherwise False"""

        if vname[0:4] == 'mask':
            data = data > 0

        return data

    def load_from_prefix(self, prefix, iternum=None, extra_metadata=None):
        """Read data and look up metadata for grid variable `name`.

        Parameters
        ----------
        name : string
            The name of the grid variable.
        iternume : int (optional)
            MITgcm iteration number

        Yields
        -------
        varname : string
            The name of the variable
        dims : list
            The dimension list
        data : arraylike
            The raw data
        attrs : dict
            The metadata attributes
        """

        fname_base = prefix

        # some special logic is required for grid variables
        if prefix in self._all_grid_variables:
            # grid variables don't have an iteration number suffix
            iternum = None
            # some grid variables have a different filename than their varname
            if 'filename' in self._all_grid_variables[prefix]:
                fname_base = self._all_grid_variables[prefix]['filename']
            ddir = self.grid_dir
        else:
            assert iternum is not None
            ddir = self.data_dir

        basename = os.path.join(ddir, fname_base)
        chunks = "CS" if self.cs else "3D"

        try:
            vardata = read_mds(basename, iternum, endian=self.endian,
                               llc=self.llc, llc_method=self.llc_method,
                               extra_metadata=extra_metadata, chunks=chunks)

        except IOError as ioe:
            # that might have failed because there was no meta file present
            # we can try to get around this by specifying the shape and dtype
            try:
                ndims = len(self._all_data_variables[prefix]['dims'])
            except KeyError:
                ndims = 3
            if ndims == 3 and self.nz > 1:
                data_shape = self.default_shape_3D
            elif ndims == 2 or self.nz == 1:
                data_shape = self.default_shape_2D
            elif self.cs:
                data_shape = None  # handle this inside read_mds
            else:
                raise ValueError("Can't determine shape "
                                 "of variable %s" % prefix)
            vardata = read_mds(basename, iternum, endian=self.endian,
                               dtype=self.default_dtype,
                               shape=data_shape, llc=self.llc,
                               llc_method=self.llc_method,
                               extra_metadata=extra_metadata, chunks=chunks)

        for vname, data in vardata.items():
            # we now have to revert to the original prefix once the file is read
            if fname_base != prefix:
                vname = prefix
            try:
                metadata = (self._all_grid_variables[vname]
                            if vname in self._all_grid_variables
                            else self._all_data_variables[vname])

            except KeyError:
                if self._ignore_unknown_vars:
                    # we didn't find any metadata, so we just skip this var
                    continue
                else:
                    raise KeyError("Couln't find metadata for variable %s "
                                   "and `ignore_unknown_vars`==False." % vname)

            # maybe slice and squeeze the data
            if 'slice' in metadata:
                # if we are slicing, we can assume it is safe to convert dask
                # array to numpy
                data = np.asarray(data)
                sl = metadata['slice']
                # need to promote the variable to higher dimensions in the
                # to handle certain 2D model outputs
                if len(sl) == 3 and data.ndim == 2:
                    data.shape = (1,) + data.shape
                data = np.atleast_1d(data[sl])

            if 'transform' in metadata:
                # transform is a function to be called on the data
                data = metadata['transform'](data)

            # make sure we copy these things
            dims = list(metadata['dims'])
            attrs = dict(metadata['attrs'])

            # Some 2D output squeezes one of the dimensions out (e.g. hFacC).
            # How should we handle this? Can either eliminate one of the dims
            # or add an extra axis to the data. Let's try the former, on the
            # grounds that it is simpler for the user.
            if ((len(dims) == 3 and data.ndim == 2) or
                    ((self.llc or self.cs) and (len(dims) == 3 and data.ndim == 3))):
                # Deleting the first dimension (z) assumes that 2D data always
                # corresponds to x,y horizontal data. Is this really true?
                # The answer appears to be yes: 2D (x|y,z) data retains the
                # missing dimension as an axis of length 1.
                # Also handles https://github.com/xgcm/xmitgcm/issues/140
                # (special case for 2d llc diags)
                dims = dims[1:]
            elif len(dims) == 1 and data.ndim > 1:
                # this is for certain profile data like RC, PHrefC, etc.
                # for some reason, dask arrays don't work here
                # ok to promote to numpy array because data is always 1D
                data = np.atleast_1d(np.asarray(data).squeeze())

            if self.llc:
                dims, data = _reshape_for_llc(dims, data)
            if self.cs:
                dims, data = _reshape_for_cs(dims, data)

            # need to add an extra dimension at the beginning if we have a time
            # variable
            if iternum is not None:
                dims = [self.time_dim_name] + dims
                #newshape = (1,) + data.shape
                #data = data.reshape(newshape)
                data = data[None]

            yield vname, dims, data, attrs

    def get_variables(self):
        return self._variables

    def get_attrs(self):
        return self._attributes

    def get_dimensions(self):
        return self._dimensions

    def close(self):
        for var in list(self._variables):
            del self._variables[var]


def _guess_model_nz(data_dir):
    try:
        rc_meta = parse_meta_file(os.path.join(data_dir, 'RC.meta'))
        if len(rc_meta['dimList']) == 2:
            nz = 1
        else:
            nz = rc_meta['dimList'][2][2]
    except IOError:
        raise IOError("Couldn't find RC.meta file to infer nz.")
    return nz


def _guess_model_horiz_dims(data_dir, is_llc=False, is_cs=False):
    try:
        xc_meta = parse_meta_file(os.path.join(data_dir, 'XC.meta'))
        nx = int(xc_meta['dimList'][0][0])
        ny = int(xc_meta['dimList'][1][0])
    except IOError:
        raise IOError("Couldn't find XC.meta file to infer nx and ny.")
    if is_llc:
        nface = LLC_NUM_FACES
        ny //= nface
    elif is_cs:
        nface = CS_NUM_FACES
        if ny > nx:
            ny //= nface
        else:
            nx //= nface
    else:
        nface = None
    return nface, ny, nx


def _guess_layers(data_dir):
    """Return a dict matching layers suffixes to dimension length."""
    layers_files = listdir_startsandendswith(data_dir, 'layers', '.meta')
    all_layers = {}
    for fname in layers_files:
        # make sure to exclude filenames such as
        # "layers_surfflux.01.0000000001.meta"
        if not re.search('\.\d{10}\.', fname):
            # should turn "foo/bar/layers1RHO.meta" into "1RHO"
            layers_suf = os.path.splitext(os.path.basename(fname))[0][6:]
            meta = parse_meta_file(os.path.join(data_dir, fname))
            Nlayers = meta['dimList'][2][2]
            all_layers[layers_suf] = Nlayers
    return all_layers


def _get_all_grid_variables(geometry, grid_dir=None, layers={}):
    """"Put all the relevant grid metadata into one big dictionary."""
    possible_hcoords = {'cartesian': horizontal_coordinates_cartesian,
                        'llc': horizontal_coordinates_llc,
                        'cs': horizontal_coordinates_cs,
                        'curvilinear': horizontal_coordinates_curvcart,
                        'sphericalpolar': horizontal_coordinates_spherical}
    hcoords = possible_hcoords[geometry]

    # look for extra variables, if they exist in grid_dir
    extravars = _get_extra_grid_variables(grid_dir) if grid_dir is not None else {}

    allvars = [hcoords, vertical_coordinates, horizontal_grid_variables,
               vertical_grid_variables, volume_grid_variables, mask_variables,
               extravars]

    # tortured logic to add layers grid variables
    layersvars = [_make_layers_variables(layer_name)
                  for layer_name in layers]
    allvars += layersvars

    metadata = _concat_dicts(allvars)
    return metadata


def _get_extra_grid_variables(grid_dir):
    """Scan a directory and return all file prefixes for extra grid files.
       Then return the variable information for each of these"""
    extra_grid = {}

    fnames = dict([[val['filename'],key] for key,val in extra_grid_variables.items() if 'filename' in val])

    all_datafiles = listdir_endswith(grid_dir, '.data')
    for f in all_datafiles:
        prefix = os.path.split(f[:-5])[-1]
        # Only consider what we find that matches extra_grid_vars
        if prefix in extra_grid_variables:
            extra_grid[prefix] = extra_grid_variables[prefix]
        elif prefix in fnames:
            extra_grid[fnames[prefix]] = extra_grid_variables[fnames[prefix]]

    return extra_grid


def _make_layers_variables(layer_name):
    """Translate metadata template to actual variable metadata."""
    from .variables import layers_grid_variables
    lvars = OrderedDict()
    layer_num = layer_name[0]
    # should always be int
    assert isinstance(int(layer_num), int)
    layer_id = 'l' + layer_num
    for key, vals in layers_grid_variables.items():
        # replace the name template with the actual name
        # e.g. layer_NAME_bounds -> layer_1RHO_bounds
        varname = key.replace('NAME', layer_name)
        metadata = _recursively_replace(vals, 'NAME', layer_name)
        # now fix dimension
        metadata['dims'] = [metadata['dims'][0].replace('l', layer_id)]
        lvars[varname] = metadata
    return lvars


def _recursively_replace(item, search, replace):
    """Recursively search and replace all strings in dictionary values."""
    if isinstance(item, dict):
        return {key: _recursively_replace(item[key], search, replace)
                for key in item}
    try:
        return item.replace(search, replace)
    except AttributeError:
        # probably no such method
        return item


def _get_all_data_variables(data_dir, grid_dir, layers):
    """"Put all the relevant data metadata into one big dictionary."""
    allvars = [state_variables]
    allvars.append(package_state_variables)

    # add others from available_diagnostics.log
    # search in the data dir
    fnameD = os.path.join(data_dir, 'available_diagnostics.log')
    # and in the grid dir
    fnameG = os.path.join(grid_dir, 'available_diagnostics.log')
    # first look in the data dir
    if os.path.exists(fnameD):
        diag_file = fnameD
    # then in the grid dir
    elif os.path.exists(fnameG):
        diag_file = fnameG
    else:
        warnings.warn("Couldn't find available_diagnostics.log "
                      "in %s or %s. Using default version." % (data_dir, grid_dir))
        from .default_diagnostics import diagnostics
        diag_file = StringIO(diagnostics)
    available_diags = parse_available_diagnostics(diag_file, layers)
    allvars.append(available_diags)
    metadata = _concat_dicts(allvars)

    # Now add the suffix '-T' to every diagnostic. This is a somewhat hacky
    # way to increase the coverage of possible output filenames.
    # But it doesn't work in python3!!!
    extra_metadata = OrderedDict()
    for name, val in metadata.items():
        newname = name + '-T'
        extra_metadata[newname] = val
    metadata = _concat_dicts([metadata, extra_metadata])

    # now fill in aliases
    for alias, original in aliases.items():
        metadata[alias] = metadata[original]

    return metadata


def _concat_dicts(list_of_dicts):
    result = OrderedDict()
    for eachdict in list_of_dicts:
        for k, v in eachdict.items():
            result[k] = v
    return result


def _get_all_iternums(data_dir, file_prefixes=None,
                      file_format='*.??????????.data'):
    """Scan a directory for all iteration number suffixes."""
    iternums = set()
    all_datafiles = listdir_fnmatch(data_dir, file_format)
    istart = file_format.find('?')-len(file_format)
    iend = file_format.rfind('?')-len(file_format)+1
    for f in all_datafiles:
        iternum = int(f[istart:iend])
        prefix = os.path.split(f[:istart-1])[-1]
        if file_prefixes is None:
            iternums.add(iternum)
        else:
            if prefix in file_prefixes:
                iternums.add(iternum)
    iterlist = sorted(iternums)
    return iterlist


def _is_pickup_prefix(prefix):
    if len(prefix) >= 6:
        if prefix[:6] == 'pickup':
            return True
    return False


def _get_all_matching_prefixes(data_dir, iternum, file_prefixes=None,
                               ignore_pickup=True):
    """Scan a directory and return all file prefixes matching a certain
    iteration number."""
    if iternum is None:
        return []
    prefixes = set()
    all_datafiles = listdir_endswith(data_dir, '.%010d.data' % iternum)
    for f in all_datafiles:
        iternum = int(f[-15:-5])
        prefix = os.path.split(f[:-16])[-1]
        if file_prefixes is None:
            if not (ignore_pickup and _is_pickup_prefix(prefix)):
                prefixes.add(prefix)
        else:
            if prefix in file_prefixes:
                prefixes.add(prefix)
    return list(prefixes)


def _iternum_to_datetime_variable(iternum, delta_t, ref_date,
                                  calendar, time_dim_name='time'):
    # create time array
    timedata = np.atleast_1d(iternum)*delta_t
    time_attrs = {'standard_name': 'time', 'long_name': 'Time', 'axis': 'T'}
    if ref_date is not None:
        time_attrs['units'] = 'seconds since %s' % ref_date
    else:
        time_attrs['units'] = 'seconds'
    if calendar is not None:
        time_attrs['calendar'] = calendar
    timevar = xr.Variable((time_dim_name,), timedata, time_attrs)
    return timevar


def _reshape_for_llc(dims, data):
    """Take dims and data and return modified / reshaped dims and data for
    llc geometry."""

    # the data should already come shaped correctly for llc
    # but the dims are not yet correct

    # the only dimensions that get expanded into faces
    expand_dims = ['j', 'j_g']
    for dim in expand_dims:
        if dim in dims:
            # add face dimension to dims
            jdim = dims.index(dim)
            dims.insert(jdim, FACE_DIMNAME)
    assert data.ndim==len(dims), '%r %r' % (data.shape, dims)
    return dims, data

def _reshape_for_cs(dims, data):
    """Take dims and data and return modified / reshaped dims and data for
    cs geometry."""

    # the data should already come shaped correctly for cs
    # but the dims are not yet correct

    # the only dimensions that get expanded into faces
    expand_dims = ['j', 'j_g']
    for dim in expand_dims:
        if dim in dims:
            # add face dimension to dims
            jdim = dims.index(dim)
            dims.insert(jdim+1, FACE_DIMNAME)
    assert data.ndim == len(dims), '%r %r' % (data.shape, dims)
    return dims, data
