import pytest
import os
import tarfile
import numpy as np
import dask
from contextlib import contextmanager
import py
import tempfile


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

    try:
        yield str(tmpdir)
    finally:
        # move them back
        for oldpath, newpath in zip(oldpaths, newpaths):
            newpath.rename(oldpath)


# parameterized fixture are complicated
# http://docs.pytest.org/en/latest/fixture.html#fixture-parametrize

# dictionary of archived experiments and some expected properties
_experiments = {
    'global_oce_latlon': {'geometry': 'sphericalpolar',
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
                        'shape': (1, 60, 60), 'test_iternum': 10,
                        'dtype': np.dtype('f4'),
                        'expected_values': {'XC': ((0, 0), 10000.0)},
                        'all_iters': [0, 10],
                        'prefixes': ['T', 'S', 'Eta', 'U', 'V', 'W']},
    'internal_wave': {'geometry': 'sphericalpolar',
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
                        'shape': (15, 32, 6, 32), 'test_iternum': 72020,
                        'dtype': np.dtype('f4'),
                        'expected_values': {'XC': ((2, 3, 5), -29.9892)},
                        'diagnostics': ('oceDiag', ['DRHODR',
                                                    'RHOAnoma',
                                                    'CONVADJ',
                                                    'GM_Kwx',
                                                    'GM_Kwy',
                                                    'GM_Kwz',
                                                    'GM_PsiX',
                                                    'GM_PsiY'])}
}


def setup_mds_dir(tmpdir_factory, request):
    """Helper function for setting up test cases."""
    expt_name = request.param
    expected_results = _experiments[expt_name]
    target_dir = str(tmpdir_factory.mktemp('mdsdata'))
    data_dir = os.path.dirname(request.module.__file__)
    return untar(data_dir, expt_name, target_dir), expected_results


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
    return fulldir


# find the tar archive in the test directory
# http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
@pytest.fixture(scope='module', params=_experiments.keys())
def all_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)


@pytest.fixture(scope='module', params=['barotropic_gyre', 'internal_wave'])
def multidim_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)


@pytest.fixture(scope='module', params=['global_oce_latlon',
                                        'global_oce_llc90'])
def mds_datadirs_with_diagnostics(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)


@pytest.fixture(scope='module', params=['internal_wave', 'global_oce_llc90'])
def mds_datadirs_with_refdate(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)


@pytest.fixture(scope='module', params=['global_oce_latlon'])
def layers_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)


@pytest.fixture(scope='module', params=['global_oce_llc90'])
def llc_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)
