import pytest

from xmitgcm import file_utils

@pytest.fixture(scope="session")
def directory_with_files(tmpdir_factory):
    temppath = tmpdir_factory.mktemp("xmitgcm_test_data")
    temppath.join('bar.0000000001.meta').ensure(file=True)
    temppath.join('baz.data').ensure(file=True)
    return temppath

def test_listdir(directory_with_files):
    path = str(directory_with_files)
    assert sorted(file_utils.listdir(path)) == sorted(['bar.0000000001.meta', 'baz.data'])

def test_listdir_startswith(directory_with_files):
    path = str(directory_with_files)
    assert file_utils.listdir_startswith(path, 'bar') == ['bar.0000000001.meta']

def test_listdir_endswith(directory_with_files):
    path = str(directory_with_files)
    assert file_utils.listdir_endswith(path, '.data') == ['baz.data']

def test_listdir_startsandendswith(directory_with_files):
    path = str(directory_with_files)
    assert file_utils.listdir_startsandendswith(path, 'bar', '.meta') == ['bar.0000000001.meta']

def test_listdir_fnmatch(directory_with_files):
    path = str(directory_with_files)
    assert file_utils.listdir_fnmatch(path, '*.??????????.meta') == ['bar.0000000001.meta']

def test_clear_cache():
    file_utils.clear_cache()
