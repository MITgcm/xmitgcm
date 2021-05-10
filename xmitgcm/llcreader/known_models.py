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

def _requires_sverdrup(func):
    def wrapper(*args,**kwargs):
        test_path = '/scratch2/heimbach'
        if not os.path.exists(test_path):
            raise OSError("Can't find %s. We must not be on Sverdrup." % test_path)
        func(*args, **kwargs)
    return wrapper


def _make_http_filesystem():
    import fsspec
    from fsspec.implementations.http import HTTPFileSystem
    return HTTPFileSystem()

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
    grid_varnames = ['AngleCS','AngleSN','Depth',
                     'DRC','DRF',
                     'DXC','DXF','DXG',
                     'DYC','DYF','DYG',
                     'hFacC','hFacS','hFacW','PHrefC','PHrefF',
                     'RAC','RAS','RAW',
                     'RC','RF','RhoRef','rLowC','rLowS','rLowW',
                     'rSurfC','rSurfS','rSurfW','XC','YC',
                     'RAZ','XG','YG','DXV','DYU']
    mask_override = {'oceTAUX': 'c', 'oceTAUY': 'c'}


class LLC4320Model(BaseLLCModel):
    nx = 4320
    nz = 90
    delta_t = 25
    iter_start = 10368
    iter_stop = 1495152 + 1
    iter_step = 144


    time_units='seconds since 2011-09-10'
    calendar = 'gregorian'
    varnames = ['Eta', 'KPPhbl', 'oceFWflx', 'oceQnet', 'oceQsw', 'oceSflux',
                'oceTAUX', 'oceTAUY', 'PhiBot', 'Salt', 'SIarea', 'SIheff',
                'SIhsalt', 'SIhsnow', 'SIuice', 'SIvice', 'Theta', 'U', 'V', 'W']
    grid_varnames = ['AngleCS','AngleSN','Depth',
                     'DRC','DRF',
                     'DXC','DXF','DXG',
                     'DYC','DYF','DYG',
                     'hFacC','hFacS','hFacW','PHrefC','PHrefF',
                     'RAC','RAS','RAW','RC','RF',
                     'RhoRef','XC','YC','RAZ','XG','YG','DXV','DYU']
    mask_override = {'oceTAUX': 'c', 'oceTAUY': 'c'}

class ASTE270Model(BaseLLCModel):
    nface = 6
    nx = 270
    nz = 50
    domain = 'aste'
    pad_before = [90, 0, 0, 0, 0]
    pad_after = [0, 0, 0, 90, 90]
    delta_t = 600
    time_units='seconds since 2002-01-01'
    calendar = 'gregorian'

    iters=[4464, 8496, 12960, 17280, 21744,
           26064, 30528, 34992, 39312, 43776,
           48096, 52560, 57024, 61056, 65520,
           69840, 74304, 78624, 83088, 87552,
           91872, 96336, 100656, 105120, 109584,
           113760, 118224, 122544, 127008, 131328,
           135792, 140256, 144576, 149040, 153360,
           157824, 162288, 166320, 170784, 175104,
           179568, 183888, 188352, 192816, 197136,
           201600, 205920, 210384, 214848, 218880,
           223344, 227664, 232128, 236448, 240912,
           245376, 249696, 254160, 258480, 262944,
           267408, 271440, 275904, 280224, 284688,
           289008, 293472, 297936, 302256, 306720,
           311040, 315504, 319968, 324144, 328608,
           332928, 337392, 341712, 346176, 350640,
           354960, 359424, 363744, 368208, 372672,
           376704, 381168, 385488, 389952, 394272,
           398736, 403200, 407520, 411984, 416304,
           420768, 425232, 429264, 433728, 438048,
           442512, 446832, 451296, 455760, 460080,
           464544, 468864, 473328, 477792, 481824,
           486288, 490608, 495072, 499392, 503856,
           508320, 512640, 517104, 521424, 525888,
           530352, 534528, 538992, 543312, 547776,
           552096, 556560, 561024, 565344, 569808,
           574128, 578592, 583056, 587088, 591552,
           595872, 600336, 604656, 609120, 613584,
           617904, 622368, 626688, 631152, 635616,
           639648, 644112, 648432, 652896, 657216,
           661680, 666144, 670464, 674928, 679248,
           683712, 688176, 692208, 696672, 700992,
           705456, 709776, 714240, 718704, 723024,
           727488, 731808, 736272, 740736, 744912,
           749376, 753696, 758160, 762480, 766944,
           771408, 775728, 780192, 784512, 788976,
           793440, 797472, 801936, 806256, 810720,
           815040, 819504, 823968, 828288, 832752,
           837072, 841536, 841544]
    varnames = ['ADVr_SLT', 'ADVr_TH',  'ADVxHEFF', 'ADVxSNOW', 'ADVx_SLT',
                'ADVx_TH',  'ADVyHEFF', 'ADVySNOW', 'ADVy_SLT', 'ADVy_TH',
                'DETADT2',  'DFrE_SLT', 'DFrE_TH',  'DFrI_SLT', 'DFrI_TH',
                'DFxEHEFF', 'DFxESNOW', 'DFxE_SLT', 'DFxE_TH',  'DFyEHEFF',
                'DFyESNOW', 'DFyE_SLT', 'DFyE_TH',  'ETAN',     'ETANSQ',
                'GM_PsiX',  'GM_PsiY',  'KPPg_SLT', 'KPPg_TH',  'MXLDEPTH',
                'PHIBOT',   'SALT',     'SFLUX',    'SIaaflux', 'SIacSubl',
                'SIarea',   'SIatmFW',  'SIatmQnt', 'SIheff',   'SIhsnow',
                'SIsnPrcp', 'SItflux',  'SIuice',   'SIvice',   'SRELAX',
                'TFLUX',    'THETA',    'TRELAX',   'UVELMASS', 'VVELMASS',
                'WSLTMASS', 'WTHMASS',  'WVELMASS', 'oceFWflx', 'oceQnet',
                'oceQsw',   'oceSPDep', 'oceSPflx', 'oceSPtnd', 'oceSflux',
                'oceTAUX',  'oceTAUY',  'sIceLoad']

    grid_varnames = ['AngleCS', 'AngleSN',   'DRC',       'DRF',       'DXC',
                     'DXG',     'DYC',       'DYG',       'Depth',     'PHrefC',
                     'PHrefF',  'RAC',       'RAS',       'RAW',       'RAZ',
                     'RC',      'RF',        'RhoRef',    'XC',        'XG',
                     'YC',      'YG',        'hFacC',     'hFacS',     'hFacW',
                     'maskC',   'maskCtrlC', 'maskCtrlS', 'maskCtrlW', 'maskInC',
                     'maskInS', 'maskInW',   'maskS',     'maskW']

    dtype={"ADVr_SLT":">f8", "ADVr_TH":">f8", "ADVxHEFF":">f8",
           "ADVxSNOW":">f8", "ADVx_SLT":">f8", "ADVx_TH":">f8",
           "ADVyHEFF":">f8", "ADVySNOW":">f8", "ADVy_SLT":">f8",
           "ADVy_TH":">f8", "AngleCS":">f8", "AngleSN":">f8",
           "DETADT2":">f4", "DFrE_SLT":">f8", "DFrE_TH":">f8",
           "DFrI_SLT":">f8", "DFrI_TH":">f8", "DFxEHEFF":">f8",
           "DFxESNOW":">f8", "DFxE_SLT":">f8", "DFxE_TH":">f8",
           "DFyEHEFF":">f8", "DFyESNOW":">f8", "DFyE_SLT":">f8",
           "DFyE_TH":">f8", "DRC":">f8", "DRF":">f8",
           "DXC":">f8", "DXG":">f8", "DYC":">f8",
           "DYG":">f8", "Depth":">f8", "ETAN":">f4",
           "ETANSQ":">f4", "GM_PsiX":">f4", "GM_PsiY":">f4",
           "KPPg_SLT":">f8", "KPPg_TH":">f8", "MXLDEPTH":">f4",
           "PHIBOT":">f4", "PHrefC":">f8", "PHrefF":">f8",
           "RAC":">f8", "RAS":">f8", "RAW":">f8",
           "RAZ":">f8", "RC":">f8", "RF":">f8",
           "RhoRef":">f8", "SALT":">f4", "SFLUX":">f8",
           "SIaaflux":">f8", "SIacSubl":">f8", "SIarea":">f4",
           "SIatmFW":">f8", "SIatmQnt":">f8", "SIheff":">f4",
           "SIhsnow":">f4", "SIsnPrcp":">f8", "SItflux":">f8",
           "SIuice":">f4", "SIvice":">f4", "SRELAX":">f8",
           "TFLUX":">f8", "THETA":">f4", "TRELAX":">f8",
           "UVELMASS":">f8", "VVELMASS":">f8", "WSLTMASS":">f8",
           "WTHMASS":">f8", "WVELMASS":">f8", "XC":">f8",
           "XG":">f8", "YC":">f8", "YG":">f8",
           "hFacC":">f8", "hFacS":">f8", "hFacW":">f8",
           "maskC":">f8", "maskCtrlC":">f8", "maskCtrlS":">f8",
           "maskCtrlW":">f8", "maskInC":">f8", "maskInS":">f8",
           "maskInW":">f8", "maskS":">f8", "maskW":">f8",
           "oceFWflx":">f8", "oceQnet":">f8", "oceQsw":">f8",
           "oceSPDep":">f4", "oceSPflx":">f8", "oceSPtnd":">f8",
           "oceSflux":">f8", "oceTAUX":">f4", "oceTAUY":">f4",
           "sIceLoad":">f4"}

class ECCOPortalLLC2160Model(LLC2160Model):

    def __init__(self):
        fs = _make_http_filesystem()
        base_path = 'https://data.nas.nasa.gov/ecco/download_data.php?file=/eccodata/llc_2160/compressed'
        grid_path = 'https://data.nas.nasa.gov/ecco/download_data.php?file=/eccodata/llc_2160/grid'
        mask_path = 'https://storage.googleapis.com/pangeo-ecco/llc/masks/llc_2160_masks.zarr/'
        store = stores.NestedStore(fs, base_path=base_path, mask_path=mask_path,
                                   grid_path=grid_path, shrunk=True, join_char='/')
        super(ECCOPortalLLC2160Model, self).__init__(store)


class ECCOPortalLLC4320Model(LLC4320Model):

    def __init__(self):
        fs = _make_http_filesystem()
        base_path = 'https://data.nas.nasa.gov/ecco/download_data.php?file=/eccodata/llc_4320/compressed'
        grid_path = 'https://data.nas.nasa.gov/ecco/download_data.php?file=/eccodata/llc_4320/grid'
        mask_path = 'https://storage.googleapis.com/pangeo-ecco/llc/masks/llc_4320_masks.zarr/'
        store = stores.NestedStore(fs, base_path=base_path, mask_path=mask_path,
                                   grid_path=grid_path, shrunk=True, join_char='/')
        super(ECCOPortalLLC4320Model, self).__init__(store)


class PleiadesLLC2160Model(LLC2160Model):

    @_requires_pleiades
    def __init__(self):
        from fsspec.implementations.local import LocalFileSystem
        fs = LocalFileSystem()
        base_path = '/home6/dmenemen/llc_2160/compressed'
        grid_path = '/home6/dmenemen/llc_2160/grid'
        mask_path = '/nobackup/rpaberna/llc/masks/llc_2160_masks.zarr'
        store = stores.NestedStore(fs, base_path=base_path, mask_path=mask_path,
                                   shrunk=True,grid_path=grid_path)
        super(PleiadesLLC2160Model, self).__init__(store)


class PleiadesLLC4320Model(LLC4320Model):

    @_requires_pleiades
    def __init__(self):
        from fsspec.implementations.local import LocalFileSystem
        fs = LocalFileSystem()
        base_path = '/home6/dmenemen/llc_4320/compressed'
        grid_path = '/home6/dmenemen/llc_4320/grid'
        mask_path = '/nobackup/rpaberna/llc/masks/llc_4320_masks.zarr'
        store = stores.NestedStore(fs, base_path=base_path, mask_path=mask_path,
                                   shrunk=True,grid_path=grid_path)
        super(PleiadesLLC4320Model, self).__init__(store)

class CRIOSPortalASTE270Model(ASTE270Model):

    def __init__(self):
        fs = _make_http_filesystem()
        base_path = 'https://aste-release1.s3.us-east-2.amazonaws.com/diags'
        grid_path = 'https://aste-release1.s3.us-east-2.amazonaws.com/grid'
        mask_path = 'https://aste-release1.s3.us-east-2.amazonaws.com/masks.zarr'
        store = stores.NestedStore(fs, base_path=base_path, grid_path=grid_path,
                                   mask_path=mask_path,
                                   shrunk=True, join_char='/')

        super(CRIOSPortalASTE270Model, self).__init__(store)

class SverdrupASTE270Model(ASTE270Model):

    @_requires_sverdrup
    def __init__(self):
        from fsspec.implementations.local import LocalFileSystem
        fs = LocalFileSystem()
        base_path = '/scratch2/shared/aste-release1/diags'
        grid_path = '/scratch2/shared/aste-release1/grid'
        mask_path = '/scratch2/shared/aste-release1/masks.zarr'
        store = stores.NestedStore(fs, base_path=base_path, grid_path=grid_path,
                                   mask_path=mask_path,
                                   shrunk=True, join_char='/')

        super(SverdrupASTE270Model, self).__init__(store)
