
import json
import numpy as np
import pytest
import tempfile

from xmitgcm.mds_tilez import VarZ

@pytest.fixture(scope="session")
def var_lat(tmp_path_factory):
    return tmp_path_factory.mktemp("lat")


def test_var_zattrs(var_lat):
    v = VarZ(var_lat, 'lat')

    attrs = v['.zattrs']
    attrs = json.loads(attrs)
    assert "_ARRAY_DIMENSIONS" in attrs
    assert attrs["_ARRAY_DIMENSIONS"] == ["lat"]


def test_var_zarray(var_lat):
    v = VarZ(var_lat, 'lat')

    attrs = v['.zarray']
    attrs = json.loads(attrs)
    assert "chunks" in attrs

def test_var_data(var_lat):
    v = VarZ(var_lat, 'lat')

    data = v['0']
    data = np.frombuffer(data)
    assert np.all(data == np.array([10., 11., 12.]))

