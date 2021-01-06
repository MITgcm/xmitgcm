import fsspec
import os
import zarr


class BaseStore:
    """Basic storage class for LLC data.

    Parameters
    ----------
    fs : fsspec.AbstractFileSystem
    base_path : str, optional
        Where to find the data within the filesystem
    shrunk : bool, optional
        Whether the data files have been tagged with `.shrunk`
    mask_fs, grid_fs : fsspec.AbstractFileSystem, optional
        Where to find the mask or grid datasets to decode the compression
    mask_path, grid_path : str, optional
        Path to the mask or grid datasets on the ``mask_fs`` or ``grid_fs`` filesystem
    shrunk_grid : bool, optional
        Whether the grid files have been tagged with `.shrunk`
        not always the same as for data variables
    join_char : str or None
        Character to use to join paths. Falls back on os.path.join if None.
    """

    def __init__(self, fs, base_path='/', shrunk=False,
                 mask_fs=None, mask_path=None, 
                 grid_fs=None, grid_path=None,
                 shrunk_grid=False, join_char=None):
        self.base_path = base_path
        self.fs = fs
        self.shrunk = shrunk
        self.mask_fs = mask_fs or self.fs
        self.mask_path = mask_path
        self.grid_fs = grid_fs or self.fs
        self.grid_path = grid_path
        self.shrunk_grid = shrunk_grid
        self.join_char = join_char
        if shrunk and (mask_path is None):
            raise ValueError("`mask_path` can't be None if `shrunk` is True")


    def _directory(self, varname, iternum):
        if iternum is not None:
            return self.base_path
        else:
            return self.grid_path

    def _fname(self, varname, iternum):

        if iternum is not None:
            fname = varname + '.%010d.data' % iternum
            if self.shrunk:
                fname += '.shrunk'
        else:
            fname = varname + '.data'
            if self.shrunk_grid:
                fname = varname + '.shrunk'

        return fname

    def _join(self, *args):
        if self.join_char:
            return self.join_char.join(args)
        else:
            return os.path.join(*args)

    def _full_path(self, varname, iternum):
        return self._join(self._directory(varname, iternum),
                            self._fname(varname, iternum))

    def get_fs_and_full_path(self, varname, iternum):
        """Return references to a filesystem and path within it for a specific
        variable and iteration number.

        Parameters
        ----------
        varname : str
        iternum : int

        Returns
        -------
        fs : fsspec.AbstractFileSystem
            The filesytem where the file can be found
        path : str
            The path to open
        """
        return self.fs, self._full_path(varname, iternum)

    def open_data_file(self, varname, iternum):
        """Open the file for a specific variable and iteration number.

        Parameters
        ----------
        varname : str
        iternum : int

        Returns
        -------
        fobj : file-like object
        """
        fs, path = self.get_fs_and_full_path(varname, iternum)
        return fs.open(path)

    def open_mask_group(self):
        """Open the zarr group that contains the masks

        Returns
        -------
        mask_group : zarr.Group
        """

        mapper = self.mask_fs.get_mapper(self.mask_path)
        zgroup = zarr.open_consolidated(mapper)
        return zgroup


class NestedStore(BaseStore):
    """Store where the variable are stored in subdirectories according to
    iteration number."""

    def _directory(self, varname, iternum):
        if iternum is not None:
            return self._join(self.base_path, '%010d' % iternum)
        else:
            return self.grid_path
