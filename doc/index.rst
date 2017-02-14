xmitgcm
=======

xmitgcm is a python package for reading MITgcm_ binary MDS files into
xarray_ data structures. By storing data in dask_ arrays, xmitgcm enables
parallel, out-of-core_ analysis of MITgcm output data.

.. toctree::
   :maxdepth: 2

   installation
   quick_start
   usage
   examples
   performance
   utils
   development

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
