import os
import re
import pytest
from dask.array.core import Array as dsa

llcreader = pytest.importorskip("xmitgcm.llcreader")

from .test_llcreader import (EXPECTED_VARS, GRID_VARNAMES, EXPECTED_COORDS)

onPleiades = pytest.mark.skipif(
    not os.path.exists('/home6/dmenemen'), reason="Not on Pleiades")
onSverdrup = pytest.mark.skipif(
    not os.path.exists('/scratch2/heimbach'), reason="Not on Sverdrup")


########### ECCO Pleiades Test ##################################################
@pytest.fixture(scope='module', params=[2160, 4320])
def ecco_pleiades_model(request):
    if request.param==2160:
        return llcreader.PleiadesLLC2160Model()
    else:
        return llcreader.PleiadesLLC4320Model()

@onPleiades
def test_ecco_pleiades_faces(ecco_pleiades_model):
    # just get three timesteps
    iter_stop = ecco_pleiades_model.iter_start + 2 * ecco_pleiades_model.iter_step + 1
    ds_faces = ecco_pleiades_model.get_dataset(iter_stop=iter_stop)
    nx = ecco_pleiades_model.nx
    assert ds_faces.dims == {'face': 13, 'i': nx, 'i_g': nx, 'j': nx,
                              'j_g': nx, 'k': 90, 'k_u': 90, 'k_l': 90,
                              'k_p1': 91, 'time': 3}
    assert set(EXPECTED_VARS) == set(ds_faces.data_vars)
    assert set(EXPECTED_COORDS[nx]).issubset(set(ds_faces.coords))

    # make sure vertical coordinates are in one single chunk
    for fld in ds_faces[['Z','Zl','Zu','Zp1']].coords:
        if isinstance(ds_faces[fld].data,dsa):
            assert len(ds_faces[fld].data.chunks)==1
            assert (len(ds_faces[fld]),)==ds_faces[fld].data.chunks[0]

@onPleiades
def test_ecco_pleiades_iterations(ecco_pleiades_model):
    with pytest.warns(RuntimeWarning, match=r"Iteration .* may not exist, you may need to change 'iter_start'"):
        ecco_pleiades_model.get_dataset(varnames=['Eta'], iter_start=ecco_pleiades_model.iter_start + 1, read_grid=False)

    with pytest.warns(RuntimeWarning, match=r"'iter_step' is not a multiple of .*, meaning some expected timesteps may not be returned"):
        ecco_pleiades_model.get_dataset(varnames=['Eta'], iter_step=ecco_pleiades_model.iter_step - 1, read_grid=False)

    with pytest.warns(RuntimeWarning, match=r"Some requested iterations may not exist, you may need to change 'iters'"):
        iters = [ecco_pleiades_model.iter_start, ecco_pleiades_model.iter_start + 1]
        ecco_pleiades_model.get_dataset(varnames=['Eta'], iters=iters, read_grid=False)

    with pytest.warns(None) as record:
        ecco_pleiades_model.get_dataset(varnames=['Eta'], read_grid=False)
    assert not record


@onPleiades
def test_ecco_pleiades_load(ecco_pleiades_model):
    # an expensive test because it actually loads data
    iter_stop = ecco_pleiades_model.iter_start + 2 * ecco_pleiades_model.iter_step + 1
    ds_faces = ecco_pleiades_model.get_dataset(varnames=['Eta'], iter_stop=iter_stop)
    # a lookup table
    expected = {2160: -1.3054643869400024, 4320: -1.262018084526062}
    assert ds_faces.Eta[0, 0, -1, -1].values.item() == expected[ecco_pleiades_model.nx]

@onPleiades
def test_ecco_pleiades_latlon(ecco_pleiades_model):
    iter_stop = ecco_pleiades_model.iter_start + 2 * ecco_pleiades_model.iter_step + 1
    ds_ll = ecco_pleiades_model.get_dataset(iter_stop=iter_stop, type='latlon')
    nx = ecco_pleiades_model.nx
    assert ds_ll.dims == {'i': 4*nx, 'k_u': 90, 'k_l': 90, 'time': 3,
                             'k': 90, 'j_g': 3*nx, 'i_g': 4*nx, 'k_p1': 91,
                             'j': 3*nx, 'face': 13}
    assert set(EXPECTED_VARS) == set(ds_ll.data_vars)
    assert set(EXPECTED_COORDS[nx]).issubset(set(ds_ll.coords))

    # make sure vertical coordinates are in one single chunk
    for fld in ds_ll[['Z','Zl','Zu','Zp1']].coords:
        if isinstance(ds_ll[fld].data,dsa):
            assert len(ds_ll[fld].data.chunks)==1
            assert (len(ds_ll[fld]),)==ds_ll[fld].data.chunks[0]


########### ASTE Sverdrup Test ##################################################
@pytest.fixture(scope='module')
def aste_sverdrup_model():
    return llcreader.SverdrupASTE270Model()

@onSverdrup
def test_aste_sverdrup(aste_sverdrup_model):
    # just get three timesteps
    iters = aste_sverdrup_model.iters[:3]
    ds_faces = aste_sverdrup_model.get_dataset(iters=iters)
    nx = aste_sverdrup_model.nx
    assert ds_faces.dims == {'face': 6, 'i': nx, 'i_g': nx, 'j': nx,
                              'j_g': nx, 'k': 50, 'k_u': 50, 'k_l': 50,
                              'k_p1': 51, 'time': 3}
    assert set(aste_sverdrup_model.varnames) == set(ds_faces.data_vars)
    assert set(EXPECTED_COORDS['aste_270']).issubset(set(ds_faces.coords))

    # make sure vertical coordinates are in one single chunk
    for fld in ds_faces[['Z','Zl','Zu','Zp1']].coords:
        if isinstance(ds_faces[fld].data,dsa):
            assert len(ds_faces[fld].data.chunks)==1
            assert (len(ds_faces[fld]),)==ds_faces[fld].data.chunks[0]

@onSverdrup
def test_aste_sverdrup_iterations(aste_sverdrup_model):
    with pytest.warns(RuntimeWarning, match=r"Some requested iterations may not exist, you may need to change 'iters'"):
        #iters = [ecco_sverdrup_model.iter_start, ecco_sverdrup_model.iter_start + 1]
        iters = aste_sverdrup_model.iters[:2]
        iters[1] = iters[1] + 1
        aste_sverdrup_model.get_dataset(varnames=['ETAN'], iters=iters, read_grid=False)

    with pytest.warns(None) as record:
        iters = aste_sverdrup_model.iters[:2]
        aste_sverdrup_model.get_dataset(varnames=['ETAN'], iters=iters, read_grid=False)
    assert not record

@onSverdrup
def test_aste_sverdrup_load(aste_sverdrup_model):
    # an expensive test because it actually loads data
    iters = aste_sverdrup_model.iters[:3]
    ds_faces = aste_sverdrup_model.get_dataset(varnames=['ETAN'], iters=iters)
    expected = 0.641869068145752
    assert ds_faces.ETAN[0, 1, 0, 0].values.item() == expected

@onSverdrup
def test_aste_sverdrup_latlon(aste_sverdrup_model):
    iters = aste_sverdrup_model.iters[:3]
    with pytest.raises(TypeError):
        ds_ll = aste_sverdrup_model.get_dataset(iters=iters,type='latlon')
