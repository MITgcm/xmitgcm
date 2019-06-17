from .llcmodel import BaseLLCModel
from . import stores

class LLC90Model(BaseLLCModel):
    nx = 90
    nz = 50
    delta_t = 3600
    time_units = 'seconds since 1948-01-01 12:00:00'
    calendar = 'gregorian'

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


class ECCOPortalLLC4320Model(LLC4320Model):

    def __init__(self):
        from fsspec.implementations.http import HTTPFileSystem
        fs = HTTPFileSystem()
        base_path = 'https://data.nas.nasa.gov/ecco/download_data.php?file=/eccodata/llc_4320/compressed'
        mask_path = 'https://storage.googleapis.com/pangeo-ecco/llc/masks/llc_4320_masks.zarr/'
        store = stores.NestedStore(fs, base_path=base_path, mask_path=mask_path,
                                   shrunk=True)
        super(ECCOPortalLLC4320Model, self).__init__(store)
