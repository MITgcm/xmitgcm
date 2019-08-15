import os

from .llcmodel import BaseLLCModel
from . import stores


def _requires_pleiades(func):
    def wrapper(*args, **kwargs):
        # is there a better choice
        test_path = '/home6/dmenemen'
        if not os.path.exists(test_path):
            raise OSError("Can't find %s. We must not be on Pleiades." % test_path)
        func(*args, **kwargs)
    return wrapper


class LLC90Model(BaseLLCModel):
    nx = 90
    nz = 50
    delta_t = 3600
    time_units = 'seconds since 1948-01-01 12:00:00'
    calendar = 'gregorian'


class LLC2160Model(BaseLLCModel):
    nx = 2160
    nz = 90
    delta_t = 45
    iter_start = 92160
    iter_stop = 1586400 + 1
    iter_step = 80
    time_units='seconds since 2011-01-17'
    calendar = 'gregorian'
    varnames = ['Eta', 'KPPhbl', 'oceFWflx', 'oceQnet', 'oceQsw', 'oceSflux',
                'oceTAUX', 'oceTAUY', 'PhiBot', 'Salt', 'SIarea', 'SIheff',
                'SIhsalt', 'SIhsnow', 'SIuice', 'SIvice', 'Theta', 'U', 'V', 'W']


class LLC4320Model(BaseLLCModel):
    nx = 4320
    nz = 90
    delta_t = 25
    iter_start = 10368
    iter_stop = 1310544 + 1
    iter_step = 144
    time_units='seconds since 2011-09-10'
    calendar = 'gregorian'
    varnames = ['Eta', 'KPPhbl', 'oceFWflx', 'oceQnet', 'oceQsw', 'oceSflux',
                'oceTAUX', 'oceTAUY', 'PhiBot', 'Salt', 'SIarea', 'SIheff',
                'SIhsalt', 'SIhsnow', 'SIuice', 'SIvice', 'Theta', 'U', 'V', 'W']


class ECCOPortalLLC2160Model(LLC2160Model):

    def __init__(self):
        from fsspec.implementations.http import HTTPFileSystem
        fs = HTTPFileSystem(size_policy='get')
        base_path = 'https://data.nas.nasa.gov/ecco/download_data.php?file=/eccodata/llc_2160/compressed'
        mask_path = 'https://storage.googleapis.com/pangeo-ecco/llc/masks/llc_2160_masks.zarr/'
        store = stores.NestedStore(fs, base_path=base_path, mask_path=mask_path,
                                   shrunk=True)
        super(ECCOPortalLLC2160Model, self).__init__(store)


class ECCOPortalLLC4320Model(LLC4320Model):

    def __init__(self):
        from fsspec.implementations.http import HTTPFileSystem
        fs = HTTPFileSystem(size_policy='get')
        base_path = 'https://data.nas.nasa.gov/ecco/download_data.php?file=/eccodata/llc_4320/compressed'
        mask_path = 'https://storage.googleapis.com/pangeo-ecco/llc/masks/llc_4320_masks.zarr/'
        store = stores.NestedStore(fs, base_path=base_path, mask_path=mask_path,
                                   shrunk=True)
        super(ECCOPortalLLC4320Model, self).__init__(store)


class PleiadesLLC2160Model(LLC2160Model):

    @_requires_pleiades
    def __init__(self):
        from fsspec.implementations.local import LocalFileSystem
        fs = LocalFileSystem()
        base_path = '/home6/dmenemen/llc_2160/compressed'
        mask_path = '/nobackup/rpaberna/llc/masks/llc_2160_masks.zarr'
        store = stores.NestedStore(fs, base_path=base_path, mask_path=mask_path,
                                   shrunk=True)
        super(PleiadesLLC2160Model, self).__init__(store)


class PleiadesLLC4320Model(LLC4320Model):

    @_requires_pleiades
    def __init__(self):
        from fsspec.implementations.local import LocalFileSystem
        fs = LocalFileSystem()
        base_path = '/home6/dmenemen/llc_4320/compressed'
        mask_path = '/nobackup/rpaberna/llc/masks/llc_4320_masks.zarr'
        store = stores.NestedStore(fs, base_path=base_path, mask_path=mask_path,
                                   shrunk=True)
        super(PleiadesLLC4320Model, self).__init__(store)
