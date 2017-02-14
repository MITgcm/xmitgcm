Example Usage
=============

Once you have loaded your data, you can analyze it using all the capabilities
available in xarray_. Here are a few quick examples.

Land Masks
----------

xmitgcm simply reads the MDS data directly for the disk; it does not attempt to
mask land points, which are usually filled with zero values. To mask land, you
can use xarray's ``where`` function together the the ``hFac`` variables related
to MITgcm's
`partially filled cells <http://mitgcm.org/sealion/online_documents/node49.html>`_.
For example, with the ``global_oce_latlon`` dataset, an unmasked average of
salinity gives::

    ds.S.mean()
    >>> <xarray.DataArray 'S' ()>
    array(18.85319709777832, dtype=float32)

This value is unrealistically low because it includes all of the zeros inside
the land which shold be masked. To take the masked average, instead do::

    ds.S.where(ds.hFacC>0).mean()
    >>> <xarray.DataArray ()>
    array(34.73611831665039)

This is a more correct value.

Volume Weighting
----------------

However, it is still not properly volume weighted.
To take a volume-weighted average, you can do::

    volume = ds.hFacC * ds.drC * ds.rA
    (ds.S * volume).sum() / volume.sum()
    >>> <xarray.DataArray ()>
    array(34.779126627139945)

This represents the correct mean ocean salinity.
A different land mask and volume weighting is required for variables located at
the u and v points.

netCDF conversion
-----------------

Thanks to xarray_, it is trivial to convert our dataset to netCDF::

    ds.to_netcdf('myfile.nc')

It can then be reloaded directly with xarray::

    import xarray as xr
    ds = xr.open_dataset('myfile.nc')

This is an attractive option for archiving MDS data in a self-contained way.

.. _dask: http://dask.pydata.org
.. _xarray: http://xarray.pydata.org
.. _Comodo: http://pycomodo.forge.imag.fr/norm.html
.. _issues: https://github.com/xgcm/xmitgcm/issues
.. _`pull requests`: https://github.com/xgcm/xmitgcm/pulls
.. _MITgcm: http://mitgcm.org/public/r2_manual/latest/online_documents/node277.html
.. _out-of-core: https://en.wikipedia.org/wiki/Out-of-core_algorithm
.. _Anaconda: https://www.continuum.io/downloads
.. _`CF conventions`: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch04s04.html
.. _gcmfaces: http://mitgcm.org/viewvc/*checkout*/MITgcm/MITgcm_contrib/gael/matlab_class/gcmfaces.pdf
