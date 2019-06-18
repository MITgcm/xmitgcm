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
    mask_fs : fsspec.AbstractFileSystem, optional
        Where to find the mask datasets to decode the compression
    mask_path : str, optional
        Path the the mask datasets on the ``mask_fs`` filesystem
    """

    def __init__(self, fs, base_path='/', shrunk=False,
                 mask_fs=None, mask_path=None):
        self.base_path = base_path
        self.fs = fs
        self.shrunk = shrunk
        self.mask_fs = mask_fs or self.fs
        self.mask_path = mask_path
        if shrunk and (mask_path is None):
            raise ValueError("`mask_path` can't be None if `shrunk` is True")


    def _directory(self, varname, iternum):
        return self.base_path

    def _fname(self, varname, iternum):
        fname = varname + '.%010d.data' % iternum
        if self.shrunk:
            fname += '.shrunk'
        return fname

    def _full_path(self, varname, iternum):
        return os.path.join(self._directory(varname, iternum),
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
        return os.path.join(self.base_path, '%010d' % iternum)
