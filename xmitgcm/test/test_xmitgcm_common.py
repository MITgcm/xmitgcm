import pytest
import os
import fnmatch
import tarfile
import numpy as np
import dask
from contextlib import contextmanager
import py
import tempfile
import hashlib
try:
    import urllib.request as req
except ImportError:
    # urllib in python2 has different structure
    import urllib as req

from xmitgcm.file_utils import clear_cache

@contextmanager
def hide_file(origdir, *basenames):
    """Temporarily hide files within the context."""
    # make everything a py.path.local
    tmpdir = py.path.local(tempfile.mkdtemp())
    origdir = py.path.local(origdir)
    oldpaths = [origdir.join(basename) for basename in basenames]
    newpaths = [tmpdir.join(basename) for basename in basenames]

    # move the files
    for oldpath, newpath in zip(oldpaths, newpaths):
        oldpath.rename(newpath)
    # clear the cache if it exists
    clear_cache()
    try:
        yield str(tmpdir)
    finally:
        # move them back
        for oldpath, newpath in zip(oldpaths, newpaths):
            newpath.rename(oldpath)


dlroot = 'https://ndownloader.figshare.com/files/'

# parameterized fixture are complicated
# http://docs.pytest.org/en/latest/fixture.html#fixture-parametrize

# dictionary of archived experiments and some expected properties
_experiments = {
    'global_oce_latlon': {'geometry': 'sphericalpolar',
                          'dlink': dlroot + '14066630',
                          'md5': '0a846023d01cbcc16bed4963431968cf',
                          'shape': (15, 40, 90), 'test_iternum': 39600,
                          'expected_values': {'XC': ((0, 0), 2)},
                          'dtype': np.dtype('f4'),
                          'layers': {'1RHO': 31},
                          'diagnostics': ('DiagGAD-T',
                                          ['TOTTTEND', 'ADVr_TH',
                                           'ADVx_TH', 'ADVy_TH',
                                           'DFrE_TH', 'DFxE_TH',
                                           'DFyE_TH', 'DFrI_TH',
                                           'UTHMASS', 'VTHMASS', 'WTHMASS'])},
    'barotropic_gyre': {'geometry': 'cartesian',
                        'dlink': dlroot + '14066618',
                        'md5': '5200149791bfd24989ad8b98c18937dc',
                        'shape': (1, 60, 60), 'test_iternum': 10,
                        'dtype': np.dtype('f4'),
                        'expected_values': {'XC': ((0, 0), 10000.0)},
                        'all_iters': [0, 10],
                        'delta_t': 3600,
                        'prefixes': ['T', 'S', 'Eta', 'U', 'V', 'W']},
    'internal_wave': {'geometry': 'sphericalpolar',
                      'dlink': dlroot + '14066642',
                      'md5': 'eedfab1aec365fd8c17d3bc0f86a1431',
                      'shape': (20, 1, 30), 'test_iternum': 100,
                      'dtype': np.dtype('f8'),
                      'expected_values': {'XC': ((0, 0), 109.01639344262296)},
                      'all_iters': [0, 100, 200],
                      'ref_date': "1990-1-1",
                      'delta_t': 60,
                      'expected_time': [
                          (0, np.datetime64('1990-01-01T00:00:00.000000000')),
                          (1, np.datetime64('1990-01-01T01:40:00.000000000'))],
                      # these diagnostics won't load because not all levels
                      # were output...no idea how to overcome that bug
                      # 'diagnostics': ('diagout1', ['UVEL', 'VVEL']),
                      'prefixes': ['T', 'S', 'Eta', 'U', 'V', 'W']},
    'global_oce_llc90': {'geometry': 'llc',
                         'dlink': dlroot + '14066567',
                         'md5': '6c309416f91ae9baaf1fb21b3dc50e49',
                         'ref_date': "1948-01-01 12:00:00",
                         'delta_t': 3600,
                         'expected_time': [
                             (0, np.datetime64('1948-01-01T12:00:00.000000000')),
                             (1, np.datetime64('1948-01-01T20:00:00.000000000'))],
                         'shape': (50, 13, 90, 90), 'test_iternum': 8,
                         'dtype': np.dtype('f4'),
                         'expected_values': {'XC': ((2, 3, 5), -32.5)},
                         'diagnostics': ('state_2d_set1', ['ETAN',
                                                           'SIarea',
                                                           'SIheff',
                                                           'SIhsnow',
                                                           'DETADT2',
                                                           'PHIBOT',
                                                           'sIceLoad',
                                                           'MXLDEPTH',
                                                           'oceSPDep',
                                                           'SIatmQnt',
                                                           'SIatmFW',
                                                           'oceQnet',
                                                           'oceFWflx',
                                                           'oceTAUX',
                                                           'oceTAUY',
                                                           'ADVxHEFF',
                                                           'ADVyHEFF',
                                                           'DFxEHEFF',
                                                           'DFyEHEFF',
                                                           'ADVxSNOW',
                                                           'ADVySNOW',
                                                           'DFxESNOW',
                                                           'DFyESNOW',
                                                           'SIuice',
                                                           'SIvice'])},
    'curvilinear_leman': {'geometry': 'curvilinear',
                          'dlink': dlroot + '14066621',
                          'md5': 'c3203ae1fb0d6a61174dd680b7660894',
                          'delta_t': 20,
                          'ref_date': "2013-11-12 12:00",
                          'shape': (35, 64, 340),
                          'test_iternum': 6,
                          'dtype': np.dtype('f4'),
                          'expected_values': {'XC': ((0, 0), 501919.21875)},
                          'all_iters': [0, 3, 6],
                          'expected_time': [
                              (0, np.datetime64('2013-11-12T12:00:00.000000000')),
                              (1, np.datetime64('2013-11-12T12:02:00.000000000'))],
                          'prefixes': ['THETA']},

    'global_oce_cs32': {'geometry': 'cs',
                        'dlink': dlroot + '14066390',
                        'md5': '209193f4a183307b89337bb18e2e9451',
                        'shape': (15, 32, 6, 32), 'test_iternum': 72020,
                        'dtype': np.dtype('f4'),
                        'expected_values': {'XC': ((2, 3, 5), -29.98921)},
                        'diagnostics': ('oceDiag', ['DRHODR',
                                                    'RHOAnoma',
                                                    'CONVADJ',
                                                    'GM_Kwx',
                                                    'GM_Kwy',
                                                    'GM_Kwz',
                                                    'GM_PsiX',
                                                    'GM_PsiY'])},

    'ideal_2D_oce': {'geometry': 'sphericalpolar',
                     'dlink': dlroot + '17288255',
                     'md5': 'd8868731ff6a8fd951babefbc5ea69ba',
                     'expected_namelistvals': {'eosType': 'LINEAR',
                                               'viscAh': 12e5,
                                               'niter0': 36000,
                                               'delX': [3.],
                                               'fileName': ['surfDiag', 'dynDiag',
                                                            'oceDiag', 'flxDiag'],
                                               'levels': [[1.0], [],
                                                          [2., 3., 4., 5., 6.,
                                                           7., 8., 9., 10., 11.,
                                                           12., 13.]],
                                               'useDiagnostics': True},
                     'diag_levels': {'surfDiag': ([1], (0, -50)),
                                     'oceDiag': (slice(2, 14), (0, -122.5))},
                     'expected_values': {'XC': ((0, 0), 1.5)},
                     'shape': (15, 56, 1),
                     'test_iternum': 36020,
                     'dtype': np.dtype('f4')}
}


_grids = {
    'grid_llc90': {'geometry': 'llc', 'domain': 'llc',
                   'dlink': dlroot + '14072594',
                   'md5': 'f66c3195a62790d539debe6ca8f3a851',
                   'gridfile': 'tile<NFACET>.mitgrid',
                   'nx': 90, 'shape': (13, 90, 90)},

    'grid_aste270': {'geometry': 'llc', 'domain': 'aste',
                     'dlink': dlroot + '14072591',
                     'md5': '92b28c65e0dfb54b253bfcd0a249359b',
                     'gridfile': 'tile<NFACET>.mitgrid',
                     'nx': 270, 'shape': (6, 270, 270)},

    'grid_cs32': {'geometry': 'cs', 'domain': 'cs',
                  'dlink': dlroot + '14072597',
                  'md5': '848cd5b6daab5b069e96a0cff67d4b57',
                  'gridfile': 'grid_cs32.face<NFACET>.bin',
                  'nx': 32, 'shape': (6, 32, 32)}
}


def setup_mds_dir(tmpdir_factory, request, db):
    """Helper function for setting up test cases."""
    expt_name = request.param
    expected_results = db[expt_name]
    target_dir = str(tmpdir_factory.mktemp('mdsdata'))
    try:
        # user-defined directory for test datasets
        data_dir = os.environ["XMITGCM_TESTDATA"]
    except KeyError:
        # default to HOME/.xmitgcm-test-data/
        data_dir = os.environ["HOME"] + '/.xmitgcm-test-data'
    # create the directory if it doesn't exixt
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    datafile = os.path.join(data_dir, expt_name + '.tar.gz')
    # download if does not exist locally
    if not os.path.exists(datafile):
        print('File does not exist locally, downloading...')
        download_archive(expected_results['dlink'], datafile)
        localmd5 = file_md5_checksum(datafile)
        if localmd5 != expected_results['md5']:
            os.remove(datafile)
            msg = """
            MD5 checksum does not match, try downloading dataset again.
            """
            raise IOError(msg)

    return untar(data_dir, expt_name, target_dir), expected_results


def download_archive(url, filename):
    """ download file from url into datafile

    PARAMETERS:

    url: str
        url to retrieve
    filename: str
        file to save on disk
    """

    req.urlretrieve(url, filename)
    return None


def untar(data_dir, basename, target_dir):
    """Unzip a tar file into the target directory. Return path to unzipped
    directory."""
    datafile = os.path.join(data_dir, basename + '.tar.gz')
    if not os.path.exists(datafile):
        raise IOError('Could not find data file %s' % datafile)
    tar = tarfile.open(datafile)
    tar.extractall(target_dir)
    tar.close()
    # subdirectory where file should have been untarred.
    # assumes the directory is the same name as the tar file itself.
    # e.g. testdata.tar.gz --> testdata/
    fulldir = os.path.join(target_dir, basename)
    if not os.path.exists(fulldir):
        raise IOError('Could not find tar file output dir %s' % fulldir)
    # the actual data lives in a file called testdata
    # clean up ugly weird hidden files that mac-os sometimes puts in the archive
    # https://unix.stackexchange.com/questions/9665/create-tar-archive-of-a-directory-except-for-hidden-files
    # https://superuser.com/questions/259703/get-mac-tar-to-stop-putting-filenames-in-tar-archives
    bad_files = [f for f in os.listdir(fulldir)
                 if fnmatch.fnmatch(f, '._*') ]
    for f in bad_files:
        os.remove(os.path.join(fulldir, f))

    return fulldir


def file_md5_checksum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


# find the tar archive in the test directory
# http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
@pytest.fixture(scope='module', params=_experiments.keys())
def all_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request, _experiments)


@pytest.fixture(scope='module', params=['barotropic_gyre', 'internal_wave'])
def multidim_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request, _experiments)


@pytest.fixture(scope='module', params=['global_oce_latlon',
                                        'global_oce_llc90'])
def mds_datadirs_with_diagnostics(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request, _experiments)


@pytest.fixture(scope='module', params=['internal_wave', 'global_oce_llc90'])
def mds_datadirs_with_refdate(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request, _experiments)


@pytest.fixture(scope='module', params=['global_oce_latlon'])
def layers_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request, _experiments)


@pytest.fixture(scope='module', params=['global_oce_llc90'])
def llc_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request, _experiments)


@pytest.fixture(scope='module', params=['global_oce_cs32'])
def cs_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request, _experiments)


@pytest.fixture(scope='module', params=_grids.keys())
def all_grid_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request, _grids)


@pytest.fixture(scope='module', params=['ideal_2D_oce'])
def mds_datadirs_with_inputfiles(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request, _experiments)
