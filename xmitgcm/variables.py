"""
All of the metadata related to MITgcm variables, grids, naming conventions, etc.
"""
# python 3 compatiblity
from __future__ import print_function, division
import numpy as np

# xarray>=0.12.0 compatiblity
try:
    from xarray.core.pycompat import OrderedDict
except ImportError:
    from collections import OrderedDict

# We are trying to combine the following two things:
# - MITgcm grid
#   http://mitgcm.org/sealion/online_documents/node47.html
# - Comodo conventions
#   http://pycomodo.forge.imag.fr/norm.html
# To do that, we have to set the proper metadata (attributes) for the MITgcm
# variables.

# The spatial dimensions, all 1D
# There is no "data" to go with these. They are just indices.
dimensions = OrderedDict(
    # x direction
    i=dict(dims=['i'], attrs=dict(
        standard_name="x_grid_index", axis='X',
        long_name="x-dimension of the t grid",
        swap_dim='XC')),
    i_g=dict(dims=['i_g'], attrs=dict(
        standard_name="x_grid_index_at_u_location", axis='X',
        long_name="x-dimension of the u grid", c_grid_axis_shift=-0.5,
        swap_dim='XG')),
    # i_z = dict(dims=['i_z'], swap_dim='XG', attrs=dict(
    #             standard_name="x_grid_index_at_f_location", axis='X',
    #             long_name="x-dimension of the f grid", c_grid_axis_shift=-0.5)),
    # y direction
    j=dict(dims=['j'], attrs=dict(
        standard_name="y_grid_index", axis='Y',
        long_name="y-dimension of the t grid", swap_dim='YC')),
    j_g=dict(dims=['j_g'], attrs=dict(
        standard_name="y_grid_index_at_v_location", axis='Y',
        long_name="y-dimension of the v grid", c_grid_axis_shift=-0.5,
        swap_dim='YG')),
    # j_z = dict(dims=['j_z'], swap_dim='YG', attrs=dict(
    #             standard_name="y_grid_index_at_f_location", axis='Y',
    #             long_name="y-dimension of the f grid", c_grid_axis_shift=-0.5)),
    # x direction
    k=dict(dims=['k'], attrs=dict(
        standard_name="z_grid_index", axis="Z",
        long_name="z-dimension of the t grid", swap_dim='Z')),
    k_u=dict(dims=['k_u'], attrs=dict(
        standard_name="z_grid_index_at_upper_w_location",
        axis="Z", long_name="z-dimension of the w grid",
        c_grid_axis_shift=0.5, swap_dim='Zu')),
    k_l=dict(dims=['k_l'], attrs=dict(
        standard_name="z_grid_index_at_lower_w_location",
        axis="Z", long_name="z-dimension of the w grid",
        c_grid_axis_shift=-0.5, swap_dim='Zl')),
    # this is complicated because it is offset in both directions - allowed by comodo?
    k_p1=dict(dims=['k_p1'], attrs=dict(
        standard_name="z_grid_index_at_w_location",
        axis="Z", long_name="z-dimension of the w grid",
        c_grid_axis_shift=(-0.5, 0.5), swap_dim='Zp1'))
)

horizontal_coordinates_spherical = OrderedDict(
    XC=dict(dims=["j", "i"], attrs=dict(
        standard_name="longitude", long_name="longitude",
        units="degrees_east", coordinate="YC XC")),
    YC=dict(dims=["j", "i"], attrs=dict(
        standard_name="latitude", long_name="latitude",
        units="degrees_north", coordinate="YC XC")),
    XG=dict(dims=["j_g", "i_g"], attrs=dict(
        standard_name="longitude_at_f_location", long_name="longitude",
        units="degrees_east", coordinate="YG XG")),
    YG=dict(dims=["j_g", "i_g"], attrs=dict(
        standard_name="latitude_at_f_location", long_name="latitude",
        units="degrees_north", coordinate="YG XG"))
)

horizontal_coordinates_llc = horizontal_coordinates_spherical.copy()
horizontal_coordinates_llc.update(OrderedDict(
    CS=dict(dims=["j", "i"], attrs=dict(standard_name="Cos of grid orientation angle",
                                        long_name="AngleCS", units=" ", coordinate="YC XC"),
            filename='AngleCS'),
    SN=dict(dims=["j", "i"], attrs=dict(standard_name="Sin of grid orientation angle",
                                        long_name="AngleSN", units=" ", coordinate="YC XC"),
            filename='AngleSN')
)
)

horizontal_coordinates_cs = horizontal_coordinates_spherical.copy()
horizontal_coordinates_cs.update(OrderedDict(
    CS=dict(dims=["j", "i"], attrs=dict(standard_name="Cos of grid orientation angle",
                                        long_name="AngleCS", units=" ", coordinate="YC XC"),
            filename='AngleCS'),
    SN=dict(dims=["j", "i"], attrs=dict(standard_name="Sin of grid orientation angle",
                                        long_name="AngleSN", units=" ", coordinate="YC XC"),
            filename='AngleSN')
)
)

horizontal_coordinates_curvcart = OrderedDict(
    XC=dict(dims=["j", "i"], attrs=dict(
        standard_name="plane_x_coordinate", long_name="x coordinate",
        units="m", coordinate="YC XC")),
    YC=dict(dims=["j", "i"], attrs=dict(
        standard_name="plane_y_coordinate", long_name="y coordinate",
        units="m", coordinate="YC XC")),
    XG=dict(dims=["j_g", "i_g"], attrs=dict(
        standard_name="plane_x_coordinate_at_f_location",
        long_name="x coordinate", units="m", coordinate="YG XG")),
    YG=dict(dims=["j_g", "i_g"], attrs=dict(
        standard_name="plane_y_coordinate_at_f_location",
        long_name="y coordinate", units="m", coordinate="YG XG")),
    CS=dict(dims=["j", "i"], attrs=dict(standard_name="Cos of grid orientation angle",
                                        long_name="AngleCS", units=" ", coordinate="YC XC"),
            filename='AngleCS'),
    SN=dict(dims=["j", "i"], attrs=dict(standard_name="Sin of grid orientation angle",
                                        long_name="AngleSN", units=" ", coordinate="YC XC"),
            filename='AngleSN')
)

horizontal_coordinates_cartesian = OrderedDict(
    XC=dict(dims=["j", "i"], attrs=dict(
        standard_name="plane_x_coordinate", long_name="x coordinate",
        units="m", coordinate="YC XC")),
    YC=dict(dims=["j", "i"], attrs=dict(
        standard_name="plane_y_coordinate", long_name="y coordinate",
        units="m", coordinate="YC XC")),
    XG=dict(dims=["j_g", "i_g"], attrs=dict(
        standard_name="plane_x_coordinate_at_f_location",
        long_name="x coordinate", units="m", coordinate="YG XG")),
    YG=dict(dims=["j_g", "i_g"], attrs=dict(
        standard_name="plane_y_coordinate_at_f_location",
        long_name="y coordinate", units="m", coordinate="YG XG"))
)

vertical_coordinates = OrderedDict(
    Z=dict(dims=["k"], attrs=dict(
        standard_name="depth",
        long_name="vertical coordinate of cell center",
        units="m", positive="down"),
        filename="RC", slice=(slice(None), 0, 0)),
    Zp1=dict(dims=["k_p1"], attrs=dict(
        standard_name="depth_at_w_location",
        long_name="vertical coordinate of cell interface",
        units="m", positive="down"),
        filename="RF", slice=(slice(None), 0, 0)),
    Zu=dict(dims=["k_u"], attrs=dict(
        standard_name="depth_at_upper_w_location",
        long_name="vertical coordinate of upper cell interface",
        units="m", positive="down"),
        filename="RF", slice=(slice(1, None), 0, 0)),
    Zl=dict(dims=["k_l"], attrs=dict(
        standard_name="depth_at_lower_w_location",
        long_name="vertical coordinate of lower cell interface",
        units="m", positive="down"),
        filename="RF", slice=(slice(None, -1), 0, 0))
)

# I got these variable names from the default MITgcm netcdf output
horizontal_grid_variables = OrderedDict(
    # tracer cell
    rA=dict(dims=["j", "i"], attrs=dict(standard_name="cell_area",
                                        long_name="cell area", units="m2", coordinate="YC XC"),
            filename='RAC'),
    dxG=dict(dims=["j_g", "i"], attrs=dict(
        standard_name="cell_x_size_at_v_location",
        long_name="cell x size", units="m", coordinate="YG XC"),
        filename='DXG'),
    dyG=dict(dims=["j", "i_g"], attrs=dict(
        standard_name="cell_y_size_at_u_location",
        long_name="cell y size", units="m", coordinate="YC XG"),
        filename='DYG'),
    Depth=dict(dims=["j", "i"], attrs=dict(standard_name="ocean_depth",
                                           long_name="ocean depth", units="m", coordinate="XC YC")),
    # vorticity cell
    rAz=dict(dims=["j_g", "i_g"], attrs=dict(
        standard_name="cell_area_at_f_location",
        long_name="cell area", units="m", coordinate="YG XG"),
        filename='RAZ'),
    dxC=dict(dims=["j", "i_g"], attrs=dict(
        standard_name="cell_x_size_at_u_location",
        long_name="cell x size", units="m", coordinate="YC XG"),
        filename='DXC'),
    dyC=dict(dims=["j_g", "i"], attrs=dict(
        standard_name="cell_y_size_at_v_location",
        long_name="cell y size", units="m", coordinate="YG XC"),
        filename='DYC'),
    # u cell
    rAw=dict(dims=["j", "i_g"], attrs=dict(
        standard_name="cell_area_at_u_location",
        long_name="cell area", units="m2", coordinate="YG XC"),
        filename='RAW'),
    # v cell
    rAs=dict(dims=["j_g", "i"], attrs=dict(
        standard_name="cell_area_at_v_location",
        long_name="cell area", units="m2", coordinate="YG XC"),
        filename='RAS'),
)

vertical_grid_variables = OrderedDict(
    drC=dict(dims=['k_p1'], attrs=dict(
        standard_name="cell_z_size_at_w_location",
        long_name="cell z size", units="m"), filename='DRC'),
    drF=dict(dims=['k'], attrs=dict(
        standard_name="cell_z_size",
        long_name="cell z size", units="m"), filename='DRF'),
    PHrefC=dict(dims=['k'], attrs=dict(
        standard_name="cell_reference_pressure",
        long_name='Reference Hydrostatic Pressure', units='m2 s-2')),
    PHrefF=dict(dims=['k_p1'], attrs=dict(
        standard_name="cell_reference_pressure",
        long_name='Reference Hydrostatic Pressure', units='m2 s-2'))
)

volume_grid_variables = OrderedDict(
    hFacC=dict(dims=['k', 'j', 'i'], attrs=dict(
        standard_name="cell_vertical_fraction",
        long_name="vertical fraction of open cell")),
    hFacW=dict(dims=['k', 'j', 'i_g'], attrs=dict(
        standard_name="cell_vertical_fraction_at_u_location",
        long_name="vertical fraction of open cell")),
    hFacS=dict(dims=['k', 'j_g', 'i'], attrs=dict(
        standard_name="cell_vertical_fraction_at_v_location",
        long_name="vertical fraction of open cell"))
)

# Mask files denoting wet points
# set filename to hFac then compute mask as 1 where hFacC nonzero
mask_variables = OrderedDict(
    maskC=dict(dims=['k', 'j', 'i'], attrs=dict(
        standard_name="sea_binary_mask_at_t_location",
        long_name="mask denoting wet point at center"),
        filename="hFacC"),
    maskW=dict(dims=['k', 'j', 'i_g'], attrs=dict(
        standard_name="cell_vertical_fraction_at_u_location",
        long_name="mask denoting wet point at interface"),
        filename="hFacW"),
    maskS=dict(dims=['k', 'j_g', 'i'], attrs=dict(
        standard_name="cell_vertical_fraction_at_v_location",
        long_name="mask denoting wet point at interface"),
        filename="hFacS")
)

# this a template: NAME gets replaced with the layer name (e.g. 1RHO)
#                  dim gets prepended with the layer number (e.g. l1_b)
layers_grid_variables = OrderedDict(
    layer_NAME_bounds=dict(dims=['l_b'], attrs=dict(
        standard_name="ocean_layer_coordinate_NAME_bounds",
        long_name="boundaries points of layer NAME",
        axis="NAME", c_grid_axis_shift=-0.5),
        filename="layersNAME", slice=(slice(None), 0, 0)),
    layer_NAME_center=dict(dims=['l_c'], attrs=dict(
        standard_name="ocean_layer_coordinate_NAME_center",
        long_name="center points of layer NAME",
        axis="NAME"),
        filename="layersNAME", slice=(slice(None), 0, 0),
        # if we don't convert to array, dask can't tokenize
        # https://github.com/pydata/xarray/issues/1014
        transform=(lambda x: np.asarray(0.5*(x[1:] + x[:-1])))),
    layer_NAME_interface=dict(dims=['l_i'], attrs=dict(
        standard_name="ocean_layer_coordinate_NAME_interface",
        long_name="interface points of layer NAME",
        axis="NAME", c_grid_axis_shift=-0.5),
        filename="layersNAME", slice=(slice(1, -1), 0, 0))
)

# _grid_special_mapping = {
# # name: (file_name, slice_to_extract, expecting_3D_field)
#     'Z': ('RC', (slice(None),0,0), 3),
#     'Zp1': ('RF', (slice(None),0,0), 3),
#     'Zu': ('RF', (slice(1,None),0,0), 3),
#     'Zl': ('RF', (slice(None,-1),0,0), 3),
#     # this will create problems with some curvillinear grids
#     # whate if X and Y need to be 2D?
#     'X': ('XC', (0,slice(None)), 2),
#     'Y': ('YC', (slice(None),0), 2),
#     'Xp1': ('XG', (0,slice(None)), 2),
#     'Yp1': ('YG', (slice(None),0), 2),
#     'rA': ('RAC', (slice(None), slice(None)), 2),
#     'HFacC': ('hFacC', 3*(slice(None),), 3),
#     'HFacW': ('hFacW', 3*(slice(None),), 3),
#     'HFacS': ('hFacS', 3*(slice(None),), 3),
# }


# also try to match CF standard names
# http://cfconventions.org/Data/cf-standard-names/28/build/cf-standard-name-table.html

state_variables = OrderedDict(
    # default state variables
    U = dict(dims=['k','j','i_g'], attrs=dict(
                standard_name='sea_water_x_velocity', mate='V',
                long_name='Zonal Component of Velocity', units='m s-1')),
    V = dict(dims=['k','j_g','i'], attrs=dict(
                standard_name='sea_water_y_velocity', mate='U',
                long_name='Meridional Component of Velocity', units='m s-1')),
    W = dict(dims=['k_l','j','i'], attrs=dict(
                standard_name='sea_water_z_velocity',
                long_name='Vertical Component of Velocity', units='m s-1')),
    T = dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_potential_temperature",
                long_name='Potential Temperature', units='degree_Celcius')),
    S = dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_salinity",
                long_name='Salinity', units='g kg-1')),
    PH= dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_dynamic_pressue",
                long_name='Hydrostatic Pressure Pot.(p/rho) Anomaly',
                units='m2 s-2')),
    PHL=dict(dims=['j','i'], attrs=dict(
                standard_name="sea_water_dynamic_pressure_at_sea_floor",
                long_name='Bottom Pressure Pot.(p/rho) Anomaly',
                units='m2 s-2')),
    Eta=dict(dims=['j','i'], attrs=dict(
                standard_name="sea_surface_height_above_geoid",
                long_name='Surface Height Anomaly', units='m')),
    # default tave variables
    uVeltave = dict(dims=['k','j','i_g'], attrs=dict(
                standard_name='sea_water_x_velocity', mate='vVeltave',
                long_name='Zonal Component of Velocity', units='m s-1')),
    vVeltave = dict(dims=['k','j_g','i'], attrs=dict(
                standard_name='sea_water_y_velocity', mate='uVeltave',
                long_name='Meridional Component of Velocity', units='m s-1')),
    wVeltave = dict(dims=['k_l','j','i'], attrs=dict(
                standard_name='sea_water_z_velocity',
                long_name='Vertical Component of Velocity', units='m s-1')),
    Ttave = dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_potential_temperature",
                long_name='Potential Temperature', units='degree_Celcius')),
    Stave = dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_salinity",
                long_name='Salinity', units='g kg-1')),
    PhHytave= dict(dims=['k','j','i'], attrs=dict(
                standard_name="sea_water_dynamic_pressue",
                long_name='Hydrostatic Pressure Pot.(p/rho) Anomaly',
                units='m2 s-2')),
    PHLtave=dict(dims=['j','i'], attrs=dict(
                standard_name="sea_water_dynamic_pressure_at_sea_floor",
                long_name='Bottom Pressure Pot.(p/rho) Anomaly',
                units='m2 s-2')),
    ETAtave=dict(dims=['j','i'], attrs=dict(
                standard_name="sea_surface_height_above_geoid",
                long_name='Surface Height Anomaly', units='m')),
    # TODO: finish encoding this crap!
    Convtave=dict(dims=['k_l','j','i'], attrs=dict(
                standard_name="convective_adjustment_index",
                long_name="Convective Adjustment Index")),
    Eta2tave=dict(dims=['j','i'], attrs=dict(
                standard_name="square_of_sea_surface_height_above_geoid",
                long_name='Square of Surface Height Anomaly', units='m2')),
    PHL2tave=dict(dims=['j','i'], attrs=dict(
                standard_name="square_of_sea_water_dynamic_pressure_at_sea_floor",
                long_name='Square of Bottom Pressure Pot.(p/rho) Anomaly',
                units='m4 s-4')),
    sFluxtave=dict(dims=['j','i'], attrs=dict(
                standard_name="virtual_salt_flux_into_sea_water",
                long_name='total salt flux (match salt-content variations), '
                          '>0 increases salt', units='g m-2 s-1')),
    Tdiftave=dict(dims=['k_l','j','i'], attrs=dict(
                standard_name="upward_potential_temperature_flux_"
                              "due_to_diffusion",
                long_name="Vertical Diffusive Flux of Pot.Temperature",
                units="K m3 s-1")),
    tFluxtave=dict(dims=['j','i'], attrs=dict(
                standard_name="surface_downward_heat_flux_in_sea_water",
                long_name="Total heat flux (match heat-content variations), "
                          ">0 increases theta", units="W m-2")),
    TTtave = dict(dims=['k','j','i'], attrs=dict(
                standard_name="square_of_sea_water_potential_temperature",
                long_name='Square of Potential Temperature',
                units='degree_Celcius')),
    uFluxtave=dict(dims=['j','i_g'], attrs=dict(
                standard_name="surface_downward_x_stress",
                long_name='surface zonal momentum flux, positive -> increase u',
                units='N m-2', mate='vFluxtave')),
    UStave=dict(dims=['k','j','i_g'], attrs=dict(
                standard_name="product_of_sea_water_x_velocity_and_salinity",
                long_name="Zonal Transport of Salinity",
                units="g kg-1 m s-1", mate='VStave')),
    UTtave=dict(dims=['k','j','i_g'], attrs=dict(
                standard_name="product_of_sea_water_x_velocity_and_"
                              "potential_temperature",
                long_name="Zonal Transport of Potential Temperature",
                units="K m s-1", mate='VTtave')),
    UUtave = dict(dims=['k','j','i_g'], attrs=dict(
                standard_name='square_of_sea_water_x_velocity',
                long_name='Square of Zonal Component of Velocity',
                units='m2 s-2')),
    UVtave=dict(dims=['k','j_g','i_g'], attrs=dict(
                standard_name="product_of_sea_water_x_velocity_and_"
                              "sea_water_y_velocity",
                long_name="Product of meridional and zonal velocity",
                units="m2 s-2")),
    vFluxtave=dict(dims=['j_g','i'], attrs=dict(
                standard_name="surface_downward_y_stress",
                long_name='surface meridional momentum flux, '
                          'positive -> increase u',
                units='N m-2', mate='uFluxtave')),
    VStave=dict(dims=['k','j_g','i'], attrs=dict(
                standard_name="product_of_sea_water_y_velocity_and_salinity",
                long_name="Meridional Transport of Salinity",
                units="g kg-1 m s-1", mate='UStave')),
    VTtave=dict(dims=['k','j_g','i'], attrs=dict(
                standard_name="product_of_sea_water_y_velocity_and_"
                              "potential_temperature",
                long_name="Meridional Transport of Potential Temperature",
                units="K m s-1", mate='UTtave')),
    VVtave = dict(dims=['k','j_g','i'], attrs=dict(
                standard_name='square_of_sea_water_y_velocity',
                long_name='Square of Meridional Component of Velocity',
                units='m2 s-2')),
    WStave=dict(dims=['k_l','j','i'], attrs=dict(
                standard_name="product_of_sea_water_z_velocity_and_salinity",
                long_name="Vertical Transport of Salinity",
                units="g kg-1 m s-1")),
    WTtave=dict(dims=['k_l','j','i'], attrs=dict(
                standard_name="product_of_sea_water_z_velocity_and_"
                              "potential_temperature",
                long_name="Vertical Transport of Potential Temperature",
                units="K m s-1")),
)

for n in range(1,100):
   trname = 'PTRACER%02d' % n
   state_variables[trname] = dict(dims=['k','j','i'],
       attrs=dict(standard_name='%s_concentration' % trname,
                  long_name="Concentration of %s" % trname,
                  units="kg m-3")
   )
for n in range(1,100):
   travename = 'PTRtave%02d' % n
   state_variables[travename] = dict(dims=['k','j','i'],
       attrs=dict(standard_name='%s_concentration' % travename,
                  long_name="Concentration of %s" % travename,
                  units="kg m-3")
   )


# these aliases are necessary to deal with the LLC output which was saved with
# unconventional variable names and no meta files
aliases = {'Eta': 'ETAN', 'PhiBot': 'PHIBOT', 'Salt': 'SALT', 'Theta': 'THETA'}

package_state_variables = {
    'KPPviscAz': dict(dims=['k_l', 'j', 'i'], attrs=dict(
        standard_name="KPP_vertical_eddy_visc",
        long_name='KPP vertical eddy viscosity coefficient',
        units='m2 s-1')),
    'KPPdiffKzS': dict(dims=['k_l', 'j', 'i'], attrs=dict(
        standard_name="KPP_diff_tracer_salt",
        long_name='KPP vertical diffusion coefficient for salt and tracers',
        units='m2 s-1')),
    'KPPdiffKzT': dict(dims=['k_l', 'j', 'i'], attrs=dict(
        standard_name="KPP_diff_temp",
        long_name='KPP vertical diffusion coefficient for heat',
        units='m2 s-1')),
    'KPPghat': dict(dims=['k_l', 'j', 'i'], attrs=dict(
        standard_name="KPP_non_local_coeff",
        long_name='non-local transfer coefficient',
        units='s m-2 ')),
    'KPPhbl': dict(dims=['j', 'i'], attrs=dict(
        standard_name="KPP_boundary_layer",
        long_name='KPP boundary layer depth, bulk Ri criterion',
        units='m')),
    'KPPg_TH': dict(dims=['k_l','j','i'], attrs=dict(
        standard_name='KPP_theta_flux',
        long_name='KPP non-local Flux of Pot.Temperature',
        units='degC m3 s-1')),
    'KPPg_SLT': dict(dims=['k_l','j','i'], attrs=dict(
        standard_name='KPP_salt_flux',
        long_name='KPP non-local Flux of Salinity',
        units='g kg-1 m3 s-1')),
    # pkg/thsice variables
    'ice_fract': dict(dims=['j', 'i'], attrs=dict(
        standard_name="sea_ice_area_fraction",
        long_name='THSICE sea fractional ice area',
        units=' ')),
    'ice_iceH': dict(dims=['j', 'i'], attrs=dict(
        standard_name="sea_ice_thickness",
        long_name='THSICE actual sea ice thickness',
        units='m')),
    'ice_snowH': dict(dims=['j', 'i'], attrs=dict(
        standard_name="snow_thickness",
        long_name='THSICE actual snow thickness',
        units='m')),
    'ice_snowAge': dict(dims=['j', 'i'], attrs=dict(
        standard_name="snow_age",
        long_name='THSICE snow age',
        units='s')),
    'ice_Tsrf': dict(dims=['j', 'i'], attrs=dict(
        standard_name="surface_temperature",
        long_name='temperature at surface',
        units='degC')),
    'ice_Tice1': dict(dims=['j', 'i'], attrs=dict(
        standard_name="temperature_of_sea_ice_layer1",
        long_name='THSICE temperature of ice layer 1',
        units='degC')),
    'ice_Tice2': dict(dims=['j', 'i'], attrs=dict(
        standard_name="temperature_of_sea_ice_layer2",
        long_name='THSICE temperature of ice layer 2',
        units='degC')),
    'ice_Qice1': dict(dims=['j', 'i'], attrs=dict(
        standard_name="enthalpy_content_of_sea_ice_layer1",
        long_name='THSICE enthalpy of ice layer 1',
        units='J kg-1')),
    'ice_Qice2': dict(dims=['j', 'i'], attrs=dict(
        standard_name="enthalpy_content_of_sea_ice_layer2",
        long_name='THSICE enthalpy of ice layer 2',
        units='J kg-1')),
    'ice_flxAtm': dict(dims=['j', 'i'], attrs=dict(
        standard_name="surface_net_downward_heat_flux",
        long_name='THSICE heat flux over ice from the atmosphere (+=down)',
        units='W m-2')),
    'ice_frwAtm': dict(dims=['j', 'i'], attrs=dict(
        standard_name="surface_upward_freshwater_flux",
        long_name='freshwater flux (E-P) over ice to the atmosphere (+=up)',
        units='kg m-2 s-1')),
    'ice_tOceMxL': dict(dims=['j', 'i'], attrs=dict(
        standard_name="temperature_of_ocean_mixed_layer",
        long_name='temperature of ocean mixed layer',
        units='degC')),
    'ice_sOceMxL': dict(dims=['j', 'i'], attrs=dict(
        standard_name="salinity_of_ocean_mixed_layer",
        long_name='salinity of ocean mixed layer',
        units=' ')),
    # pkg/seaice variables
    'AREA': dict(dims=['j', 'i'], attrs=dict(
        standard_name="sea_ice_area_fraction",
        long_name='SEAICE sea fractional ice area',
        units=' ')),
    'HEFF': dict(dims=['j', 'i'], attrs=dict(
        standard_name="sea_ice_thickness",
        long_name='SEAICE mean sea ice thickness per grid cell',
        units='m')),
    'HSNOW': dict(dims=['j', 'i'], attrs=dict(
        standard_name="snow_thickness",
        long_name='SEAICE mean snow ice thickness per grid cell',
        units='m')),
    'UICE': dict(dims=['j', 'i_g'], attrs=dict(
        standard_name="sea_ice_x_velocity", mate='VICE',
        long_name='SEAICE ice drift velocity in x/i-direction',
        units='m s-1')),
    'VICE': dict(dims=['j_g', 'i'], attrs=dict(
        standard_name="sea_ice_y_velocity", mate='UICE',
        long_name='SEAICE ice drift velocity in y/j-direction',
        units='m s-1')),
    'Qnet': dict(dims=['j', 'i'], attrs=dict(
        standard_name="surface_net_upward_heat_flux",
        long_name='SEAICE net upward heat flux (+=up)',
        units='W m-2')),
    'Qsw': dict(dims=['j', 'i'], attrs=dict(
        standard_name="surface_net_upward_shortwave_flux",
        long_name='SEAICE upward shortwave heat flux (+=up)',
        units='W m-2')),
    'EmPmR': dict(dims=['j', 'i'], attrs=dict(
        standard_name="surface_net_upward_freshwater_flux",
        long_name='SEAICE upward freshwater flux (+=up)',
        units='kg m-2 s-1')),
    'UWIND': dict(dims=['j', 'i'], attrs=dict(
        standard_name="x_wind", mate='VWIND',
        long_name='SEAICE wind forcing in x/i-direction',
        units='m s-1')),
    'VWIND': dict(dims=['j', 'i'], attrs=dict(
        standard_name="y_wind", mate='UWIND',
        long_name='SEAICE wind forcing in y/j-direction',
        units='m s-1')),
    'FU': dict(dims=['j', 'i_g'], attrs=dict(
        standard_name="downward_eastward_stress_at_sea_ice_base",
        mate='FV',
        long_name='SEAICE surface stress on ocean in x/i-direction',
        units='N m-2')),
    'FV': dict(dims=['j_g', 'i'], attrs=dict(
        standard_name="downward_northward_stress_at_sea_ice_base",
        mate='FU',
        long_name='SEAICE surface stress on ocean in y/j-direction',
        units='N m-2')),
    'AREAtave': dict(dims=['j', 'i'], attrs=dict(
        standard_name="sea_ice_area_fraction",
        long_name='SEAICE time averaged sea fractional ice area',
        units=' ')),
    'HEFFtave': dict(dims=['j', 'i'], attrs=dict(
        standard_name="sea_ice_thickness",
        long_name='SEAICE mean sea ice thickness per grid cell',
        units='m')),
    'UICEtave': dict(dims=['j', 'i_g'], attrs=dict(
        standard_name="sea_ice_x_velocity", mate='VICEtave',
        long_name='SEAICE ice drift velocity in x/i-direction',
        units='m s-1')),
    'VICEtave': dict(dims=['j_g', 'i'], attrs=dict(
        standard_name="sea_ice_y_velocity", mate='UICEtave',
        long_name='SEAICE ice drift velocity in y/j-direction',
        units='m s-1')),
    'FUtave': dict(dims=['j', 'i_g'], attrs=dict(
        standard_name="downward_eastward_stress_at_sea_ice_base",
        mate='FVtave',
        long_name='SEAICE surface stress on ocean in x/i-direction',
        units='N m-2')),
    'FVtave': dict(dims=['j_g', 'i'], attrs=dict(
        standard_name="downward_northward_stress_at_sea_ice_base",
        mate='FUtave',
        long_name='SEAICE surface stress on ocean in y/j-direction',
        units='N m-2')),
    'EmPmRtave': dict(dims=['j', 'i'], attrs=dict(
        standard_name="surface_net_upward_freshwater_flux",
        long_name='SEAICE upward freshwater flux (+=up)',
        units='kg m-2 s-1')),
    'QNETtave': dict(dims=['j', 'i'], attrs=dict(
        standard_name="surface_net_upward_shortwave_flux",
        long_name='SEAICE net upward heat flux (+=up)',
        units='W m-2')),
    'QSWtave': dict(dims=['j', 'i'], attrs=dict(
        standard_name="surface_net_upward_shortwave_flux",
        long_name='SEAICE upward shortwave heat flux (+=up)',
        units='W m-2')),
    # pkg/autodiff variables
    'ADJtaux': dict(dims=['j', 'i_g'], attrs=dict(
        standard_name="ADJtaux",
        long_name='dJ/dtaux: Sensitivity to zonal surface stress',
        mate='ADJtauy',
        units='dJ/(N m-2)')),
    'ADJtauy': dict(dims=['j_g', 'i'], attrs=dict(
        standard_name="ADJtaux",
        long_name='dJ/dtauy: Sensitivity to meridional surface stress',
        mate='ADJtaux',
        units='dJ/(N m-2)')),
    'ADJqnet': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJqnet",
        long_name='dJ/dqnet: Sensitivity to net upward heat flux',
        units='dJ/(W m-2)')),
    'ADJempr': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJempr",
        long_name='dJ/dempr: Sensitivity to upward freshwater flux',
        units='dJ/(kg m-2 s-1)')),
    'ADJqsw': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJqsw",
        long_name='dJ/dqsw: Sensitivity to net shortwave radiation',
        units='dJ/(W m-2)')),
    'ADJggl90tke': dict(dims=['k', 'j', 'i'], attrs=dict(
        standard_name="ADJggl90tke",
        long_name='dJ/dggl90tke: Sensitivity to TKE',
        units='dJ/(m2 s-2)')),
    'ADJdiffkr': dict(dims=['k', 'j', 'i'], attrs=dict(
        standard_name="ADJdiffkr",
        long_name='dJ/ddiffkr: Sensitivity to vertical diffusivity',
        units='dJ/(m2 s-1)')),
    'ADJkapgm': dict(dims=['k', 'j', 'i'], attrs=dict(
        standard_name="ADJkapgm",
        long_name='dJ/dkapgm: Sensitivity to GM Intensity',
        units='dJ/(m2 s-1)')),
    'ADJkapredi': dict(dims=['k', 'j', 'i'], attrs=dict(
        standard_name="ADJkapredi",
        long_name='dJ/dkapredi: Sensitivity to Redi coefficient',
        units='dJ/(m2 s-1)')),
    'ADJeddypsix': dict(dims=['k', 'j', 'i_g'], attrs=dict(
        standard_name="ADJeddypsix",
        long_name='dJ/deddypsix: Sensitivity to zonal eddy streamfunction',
        mate='ADJeddypsiy',
        units='dJ/(m2 s-1)')),
    'ADJeddypsiy': dict(dims=['k', 'j_g', 'i'], attrs=dict(
        standard_name="ADJtaux",
        long_name='dJ/deddypsiy: Sensitivity to meridional eddy streamfunction',
        mate='ADJeddypsix',
        units='dJ/(m2 s-1)')),
    'ADJsst': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJsst",
        long_name='dJ/dsst: Sensitivity to sea surface temperature',
        units='dJ/K')),
    'ADJsss': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJsss",
        long_name='dJ/dsss: Sensitivity to sea surface salinity',
        units='dJ/(g kg-1)')),
    'ADJbottomdrag': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADbottomdrag",
        long_name='dJ/dbottomdrag: Sensitivity to linear bottom drag coeff',
        units='dJ/(m-1)')),
    'ADJustress': dict(dims=['j', 'i_g'], attrs=dict(
        standard_name="ADJustress",
        long_name='dJ/dustress: Sensitivity to zonal wind stress',
        mate='ADJvstress',
        units='dJ/(N m-2)')),
    'ADJvstress': dict(dims=['j_g', 'i'], attrs=dict(
        standard_name="ADJvstress",
        long_name='dJ/dvstress: Sensitivity to meridional wind stress',
        mate='ADJustress',
        units='dJ/(N m-2)')),
    'ADJuwind': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJuwind",
        long_name='dJ/duwind: Sensitivity to zonal wind',
        mate='ADJvwind',
        units='dJ/(m s-1)')),
    'ADJvwind': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJvwind",
        long_name='dJ/dvwind: Sensitivity to meridional wind',
        mate='ADJuwind',
        units='dJ/(m s-1)')),
    'ADJatemp': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJatemp",
        long_name='dJ/datemp: Sensitivity to atmospheric surface temperature',
        units='dJ/K')),
    'ADJaqh': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJaqh",
        long_name='dJ/daqh: Sensitivity to specific surface humidity',
        units='dJ/(kg kg-1)')),
    'ADJswdown': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJswdown",
        long_name='dJ/dswdown: Sensitivity to downward solar radiation',
        units='dJ/(W m-2)')),
    'ADJlwdown': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJlwdown",
        long_name='dJ/dlwdown: Sensitivity to downward longwave radiation',
        units='dJ/(W m-2)')),
    'ADJhflux': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJhflux",
        long_name='dJ/dhflux: Sensitivity to upward heat flux',
        units='dJ/(W m-2)')),
    'ADJsflux': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJsflux",
        long_name='dJ/dsflux: Sensitivity to upward fresh water flux',
        units='dJ/(m s-1)')),
    'ADJprecip': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJprecip",
        long_name='dJ/dprecip: Sensitivity to precipitation flux',
        units='dJ/(m s-1)')),
    'ADJrunoff': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJrunoff",
        long_name='dJ/drunoff: Sensitivity to runoff',
        units='dJ/(m s-1)')),
    'ADJclimsst': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJclimsst",
        long_name='dJ/dclimsst: Sensitivity to restoring surface temperature',
        units='dJ/K')),
    'ADJclimsss': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJclimsss",
        long_name='dJ/dclimsss: Sensitivity to restoring surface salinity',
        units='dJ/(g kg-1)')),
    'ADJarea': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJarea",
        long_name='dJ/darea: Sensitivity to sea ice concentration',
        units='dJ')),
    'ADJheff': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJheff",
        long_name='dJ/dheff: Sensitivity to sea ice thickness',
        units='dJ/m')),
    'ADJhsnow': dict(dims=['j', 'i'], attrs=dict(
        standard_name="ADJhsnow",
        long_name='dJ/dhsnow: Sensitivity to snow thickness',
        units='dJ/m')),
    'ADJuice': dict(dims=['j', 'i_g'], attrs=dict(
        standard_name="ADJuice",
        long_name='dJ/duice: Sensitivity to zonal ice drift',
        mate='ADJvice',
        units='dJ/(m s-1)')),
    'ADJvice': dict(dims=['j_g', 'i'], attrs=dict(
        standard_name="ADJvice",
        long_name='dJ/dvice: Sensitivity to meridional ice drift',
        mate='ADJuice',
        units='dJ/(m s-1)'))
}

extra_grid_variables = OrderedDict(
    # Printed from write_grid when debugLevel>=debugLevC
    rLowC=dict(dims=['j', 'i'], attrs=dict(
        standard_name="depth_r0_to_bottom",
        long_name='Depth fixed r0 to bottom at tracer location',
        units='m')),
    rLowW=dict(dims=['j', 'i_g'], attrs=dict(
        standard_name="depth_r0_to_bottom_at_u_location",
        long_name='Depth from fixed r0 to bottom at u location',
        units='m')),
    rLowS=dict(dims=['j_g', 'i'], attrs=dict(
        standard_name="depth_r0_to_bottom_at_v_location",
        long_name='Depth fixed r0 to bottom at v location',
        units='m')),
    rSurfC=dict(dims=['j', 'i'], attrs=dict(
        standard_name="depth_r0_to_ref_surface",
        long_name='Depth fixed r0 to reference surface level at '
                  'tracer location',
        units='m')),
    rSurfW=dict(dims=['j', 'i_g'], attrs=dict(
        standard_name="depth_r0_to_ref_surface_at_u_location",
        long_name='Depth fixed r0 to reference surface level at u location',
        units='m')),
    rSurfS=dict(dims=['j_g', 'i'], attrs=dict(
        standard_name="depth_r0_to_ref_surface_at_v_location",
        long_name='Depth fixed r0 to reference surface level at v location',
        units='m')),
    # Printed from write_grid when useOBCS==T
    maskInC=dict(dims=['j', 'i'], attrs=dict(
        standard_name="interior_2d_mask",
        long_name='OBCS 2D interior mask at tracer location, zero beyond OB',
        units='')),
    maskInW=dict(dims=['j', 'i_g'], attrs=dict(
        standard_name="interior_2d_mask_at_u_location",
        long_name='OBCS 2D interior mask at u location, zero on & beyond OB',
        units='')),
    maskInS=dict(dims=['j_g', 'i'], attrs=dict(
        standard_name="interior_2d_mask_at_v_location",
        long_name='OBCS 2D interior mask at v location, zero on & beyond OB',
        units='')),
    # Printed from pkg/ctrl/ctrl_init_wet when useCTRL=T
    maskCtrlC=dict(dims=['k', 'j', 'i'], attrs=dict(
        standard_name="ctrl_vector_3d_mask",
        long_name='CTRL 3D mask where ctrl vector is active at '
                  'tracer location',
        units='')),
    maskCtrlW=dict(dims=['k', 'j', 'i_g'], attrs=dict(
        standard_name="ctrl_vector_3d_mask_at_u_location",
        long_name='CTRL 3D mask where ctrl vector is active at u location',
        units='')),
    maskCtrlS=dict(dims=['k', 'j_g', 'i'], attrs=dict(
        standard_name="ctrl_vector_3d_mask_at_v_location",
        long_name='CTRL 3D mask where ctrl vector is active at v location',
        units='')),
    # Reference density profile
    rhoRef=dict(dims=['k'],attrs=dict(
        standard_name="reference_density_profile",
        long_name="1D, vertical reference density profile",
        coordinate="Z",
        units='kg m-3'),
        filename='RhoRef'),
    # Additional grid metrics
    dxF=dict(dims=['j','i'],attrs=dict(
        standard_name="cell_x_size_at_t_location",
        long_name="cell x size", units="m", coordinate="YC XC"),
        filename='DXF'),
    dyF=dict(dims=['j','i'],attrs=dict(
        standard_name="cell_y_size_at_t_location",
        long_name="cell y size", units="m", coordinate="YC XC"),
        filename='DYF'),
    dxV=dict(dims=['j_g','i_g'],attrs=dict(
        standard_name="cell_x_size_at_f_location",
        long_name="cell x size", units="m", coordinate="YG XG"),
        filename='DXV'),
    dyU=dict(dims=['j_g','i_g'],attrs=dict(
        standard_name="cell_y_size_at_f_location",
        long_name="cell y size", units="m", coordinate="YG XG"),
        filename='DYU')
    # Printed from write_grid when sigma coordinates are used
    # AHybSigmF, BHybSigF, ...
    # Unclear where on the grid these variables exist,
    # skipping for now ...
)

# Nptracers=99
# _ptracers = { 'PTRACER%02d' % n :
#                (dims=['k','j','i'], 'PTRACER%02d Concentration' % n, "tracer units/m^3")
#                for n in range(Nptracers)}

# these variable names have hyphens and need a different syntax
# state_variables['GM_Kwx-T'] = dict(
#     dims=['k_l','j','i'], attrs=dict(
#         standard_name="K_31_element_of_GMRedi_tensor",
#         long_name="K_31 element (W.point, X.dir) of GM-Redi tensor",
#         units="m2 s-1"
#     )
# )
#
# state_variables['GM_Kwy-T'] = dict(
#     dims=['k_l','j','i'], attrs=dict(
#         standard_name="K_32_element_of_GMRedi_tensor",
#         long_name="K_32 element (W.point, Y.dir) of GM-Redi tensor",
#         units="m2 s-1"
#     )
# )
#
# state_variables['GM_Kwy-T'] = dict(
#     dims=['k_l','j','i'], attrs=dict(
#         standard_name="K_33_element_of_GMRedi_tensor",
#         long_name="K_33 element (W.point, Z.dir) of GM-Redi tensor",
#         units="m2 s-1"
#     )
# )


# Nptracers=99
# _ptracers = { 'PTRACER%02d' % n :
#                (dims=['k','j','i'], 'PTRACER%02d Concentration' % n, "tracer units/m^3")
#                for n in range(Nptracers)}
