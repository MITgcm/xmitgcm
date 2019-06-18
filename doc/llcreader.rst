llcreader
=========

This module provides optimized routines for reading LLC data from disk or over the web.
They were motivated explicitly by the need to provide easy access to the NASA-JPL
LLC4320 family of simulations, in support of the NASA SWOT Science Team.

.. warning::
  These routines are new and experimental. APIs are subject to change.
  Eventually the functionality provided by ``llcreader`` may become part of
  the main xmitgcm module.

Manual Dataset Creation
-----------------------

One way to use this module is to manually create the necessary objects.
First we create a ``Store`` object::

  >>> from xmitgcm import llcreader
  >>> from fsspec.implementations.local import LocalFileSystem
  >>> fs = LocalFileSystem()
  >>> store = llcreader.BaseStore(fs, base_path='/path/to/global_oce_llc90')

Then we use this to initialize a ``Model`` object::

  >>> model = llcreader.LLC90Model(store)

From this object, we can create datasets::

  >>> ds_faces = model.get_dataset(varnames=['S', 'T', 'U', 'V', 'Eta'],
                iter_start=0, iter_stop=9, iter_step=8)
  >>> ds_faces
  <xarray.Dataset>
  Dimensions:  (face: 13, i: 90, i_g: 90, j: 90, j_g: 90, k: 50, k_l: 50, k_p1: 50, k_u: 50, time: 2)
  Coordinates:
    * face     (face) int64 0 1 2 3 4 5 6 7 8 9 10 11 12
    * i        (i) int64 0 1 2 3 4 5 6 7 8 9 10 ... 80 81 82 83 84 85 86 87 88 89
    * i_g      (i_g) int64 0 1 2 3 4 5 6 7 8 9 ... 80 81 82 83 84 85 86 87 88 89
    * j        (j) int64 0 1 2 3 4 5 6 7 8 9 10 ... 80 81 82 83 84 85 86 87 88 89
    * j_g      (j_g) int64 0 1 2 3 4 5 6 7 8 9 ... 80 81 82 83 84 85 86 87 88 89
    * k        (k) int64 0 1 2 3 4 5 6 7 8 9 10 ... 40 41 42 43 44 45 46 47 48 49
    * k_u      (k_u) int64 0 1 2 3 4 5 6 7 8 9 ... 40 41 42 43 44 45 46 47 48 49
    * k_l      (k_l) int64 0 1 2 3 4 5 6 7 8 9 ... 40 41 42 43 44 45 46 47 48 49
    * k_p1     (k_p1) int64 0 1 2 3 4 5 6 7 8 9 ... 40 41 42 43 44 45 46 47 48 49
      niter    (time) int64 ...
    * time     (time) datetime64[ns] 1948-01-01T12:00:00 1948-01-01T20:00:00
  Data variables:
      S        (time, k, face, j, i) >f4 dask.array<shape=(2, 50, 13, 90, 90), chunksize=(1, 1, 3, 90, 90)>
      T        (time, k, face, j, i) >f4 dask.array<shape=(2, 50, 13, 90, 90), chunksize=(1, 1, 3, 90, 90)>
      U        (time, k, face, j, i_g) >f4 dask.array<shape=(2, 50, 13, 90, 90), chunksize=(1, 1, 3, 90, 90)>
      V        (time, k, face, j_g, i) >f4 dask.array<shape=(2, 50, 13, 90, 90), chunksize=(1, 1, 3, 90, 90)>
      Eta      (time, face, j, i) >f4 dask.array<shape=(2, 13, 90, 90), chunksize=(1, 3, 90, 90)>

There are many options you can pass to ``get_dataset`` to control the output
and dask chunking. See the class documentation for more details:
:meth:`xmitgcm.llcreader.BaseLLCModel.get_dataset`.

Pre-Defined Models
------------------

Because of the size and complexity of the LLC datasets, we provide some
pre-defined references to existing stores and models that can be used right
away.

ECCO HTTP Data Portal
~~~~~~~~~~~~~~~~~~~~~

NASA has created an experimental data portal to access the LLC data over the
web via standard HTTP calls. More info and a data browser can be found at
https://data.nas.nasa.gov/ecco/.
The ``llcreader`` module provides a way to access this data directly via
xarray and dask::

  >>> import llcreader
  >>> model = llcreader.ECCOPortalLLC4320Model()
  >>> ds_faces = model.get_dataset(iter_start=10368, iter_stop = 11000)


API Documentation
-----------------

.. autoclass:: xmitgcm.llcreader.BaseLLCModel
  :members:
