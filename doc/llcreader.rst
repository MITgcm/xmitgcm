llcreader
=========

This module provides optimized routines for reading LLC data from disk or over the web.
They were motivated explicitly by the need to provide easy access to the NASA-JPL
LLC4320 family of simulations, in support of the NASA SWOT Science Team.

.. warning::
  These routines are new and experimental. APIs are subject to change.
  Eventually the functionality provided by ``llcreader`` may become part of
  the main xmitgcm module.

Limitations
-----------

llcreader uses `Dask <https://dask.org/>`_ to provide a
`lazy <https://en.wikipedia.org/wiki/Lazy_evaluation>`_ view of the data;
no data are actually loaded into memory until required for computation.
The data are represented using a graph, with node corresponding to a specific
chunk of the full array.
Because of the vast size of the LLC datasets, the user is advised to read about
`Dask best practices <https://docs.dask.org/en/latest/best-practices.html>`_.
If you wish to work with the full-size datasets, you must be aware of Dask's
inherent limitations, and first gain experience with smaller problems.
In particular, it is quite easy to create very large Dask graphs that can
overwhelm your computer's memory and CPU capacity, even without loading any
data!

To make this warning more concrete, let's consider a single variable such as
``Theta`` from the LLC4320 simulation.
In "face" form, the dimensions of this array are 9030 timesteps x 90 vertical
levels x 13 faces x 4320 j points x 4320 i points = 197170122240000 values.
With 32-bit precision, this requires 788 TB of memory!
If we divide this array into 1 GB chunks, we will have 788,000 chunks.
The lazy representation of the array requires about 1 kB of memory per chunk,
and processing the graph for computation requires about 200 µs per chunk.
Therefore, the lazy representation will occupy nearly 1GB of memory and take
nearly 3 minutes to even begin computation.
There are 5 such 3D variables in the LLC4320 dataset, plus another 10 2D
variables.
The bottom line is that creating graphs of this size will seriously bog down
a laptop.

There are several strategies that can be used to mitigate this problem:

- **Subset variables**: llc reader allows you to create datasets the contain
  only the specific variables you need, via the ``varnames`` keyword.

- **Subset vertical levels**: if you don't need full depth fields, you can
  select just the depth levels you need using the ``k_levels`` keyword.

- **Subset in time**: you can specific the iteration number start, stop and step
  manually.

- **Use larger chunks**: llcreader allows you to vary the size of the chunks
  in the ``k`` dimension using the ``k_chunksize`` keyword. Using full depth
  chunks (``k_chunksize=90``) will produce chunks up to 20 GB in size, but
  in a much smaller graph.

Because of how the files are compressed and stored on disk, unfortunately it
is **not** possible to subset in space at the time of dataset creation.
However, this can be done later, via xarray.

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
xarray and dask.
In this example, we display the whole dataset laziy, following the advice above
to use a large ``k_chunksize``.
By default, all variables and all timesteps are loaded::

  >>> import llcreader
  >>> model = llcreader.ECCOPortalLLC4320Model()
  >>> ds = model.get_dataset(k_chunksize=90)
  >>> print(ds)
  <xarray.Dataset>
  Dimensions:   (face: 13, i: 4320, i_g: 4320, j: 4320, j_g: 4320, k: 90, k_l: 90, k_p1: 90, k_u: 90, time: 9030)
  Coordinates:
    * face      (face) int64 0 1 2 3 4 5 6 7 8 9 10 11 12
    * i         (i) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * i_g       (i_g) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * j         (j) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * j_g       (j_g) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * k         (k) int64 0 1 2 3 4 5 6 7 8 9 10 ... 80 81 82 83 84 85 86 87 88 89
    * k_u       (k_u) int64 0 1 2 3 4 5 6 7 8 9 ... 80 81 82 83 84 85 86 87 88 89
    * k_l       (k_l) int64 0 1 2 3 4 5 6 7 8 9 ... 80 81 82 83 84 85 86 87 88 89
    * k_p1      (k_p1) int64 0 1 2 3 4 5 6 7 8 9 ... 80 81 82 83 84 85 86 87 88 89
      niter     (time) int64 ...
    * time      (time) datetime64[ns] 2011-09-13 ... 2012-09-23T05:00:00
  Data variables:
      Eta       (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      KPPhbl    (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      oceFWflx  (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      oceQnet   (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      oceQsw    (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      oceSflux  (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      oceTAUX   (time, face, j, i_g) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      oceTAUY   (time, face, j_g, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      PhiBot    (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      Salt      (time, k, face, j, i) >f4 dask.array<shape=(9030, 90, 13, 4320, 4320), chunksize=(1, 90, 3, 4320, 4320)>
      SIarea    (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      SIheff    (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      SIhsalt   (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      SIhsnow   (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      SIuice    (time, face, j, i_g) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      SIvice    (time, face, j_g, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>
      Theta     (time, k, face, j, i) >f4 dask.array<shape=(9030, 90, 13, 4320, 4320), chunksize=(1, 90, 3, 4320, 4320)>
      U         (time, k, face, j, i_g) >f4 dask.array<shape=(9030, 90, 13, 4320, 4320), chunksize=(1, 90, 3, 4320, 4320)>
      V         (time, k, face, j_g, i) >f4 dask.array<shape=(9030, 90, 13, 4320, 4320), chunksize=(1, 90, 3, 4320, 4320)>
      W         (time, k_l, face, j, i) >f4 dask.array<shape=(9030, 90, 13, 4320, 4320), chunksize=(1, 90, 3, 4320, 4320)>

This dataset is useless for computations on a laptop, because the individual
chunks require nearly 20 GB of memory.
Some more more practical examples are the following.

Get a single 2D variable::

  >>> ds = model.get_dataset(varnames=['Eta'])
  >>> print(ds)
  <xarray.Dataset>
  Dimensions:  (face: 13, i: 4320, i_g: 4320, j: 4320, j_g: 4320, k: 90, k_l: 90, k_p1: 90, k_u: 90, time: 9030)
  Coordinates:
    * face     (face) int64 0 1 2 3 4 5 6 7 8 9 10 11 12
    * i        (i) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * i_g      (i_g) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * j        (j) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * j_g      (j_g) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * k        (k) int64 0 1 2 3 4 5 6 7 8 9 10 ... 80 81 82 83 84 85 86 87 88 89
    * k_u      (k_u) int64 0 1 2 3 4 5 6 7 8 9 ... 80 81 82 83 84 85 86 87 88 89
    * k_l      (k_l) int64 0 1 2 3 4 5 6 7 8 9 ... 80 81 82 83 84 85 86 87 88 89
    * k_p1     (k_p1) int64 0 1 2 3 4 5 6 7 8 9 ... 80 81 82 83 84 85 86 87 88 89
      niter    (time) int64 ...
    * time     (time) datetime64[ns] 2011-09-13 ... 2012-09-23T05:00:00
  Data variables:
      Eta      (time, face, j, i) >f4 dask.array<shape=(9030, 13, 4320, 4320), chunksize=(1, 3, 4320, 4320)>

Get a few vertical levels from some 3D variables::

  >>> ds = model.get_dataset(varnames=['Salt', 'Theta'], k_levels=[1, 10, 40])
  >>> print(ds)
  <xarray.Dataset>
  Dimensions:  (face: 13, i: 4320, i_g: 4320, j: 4320, j_g: 4320, k: 3, k_l: 3, k_p1: 3, k_u: 3, time: 9030)
  Coordinates:
    * face     (face) int64 0 1 2 3 4 5 6 7 8 9 10 11 12
    * i        (i) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * i_g      (i_g) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * j        (j) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * j_g      (j_g) int64 0 1 2 3 4 5 6 7 ... 4313 4314 4315 4316 4317 4318 4319
    * k        (k) int64 1 10 40
    * k_u      (k_u) int64 1 10 40
    * k_l      (k_l) int64 1 10 40
    * k_p1     (k_p1) int64 1 10 40
      niter    (time) int64 ...
    * time     (time) datetime64[ns] 2011-09-13 ... 2012-09-23T05:00:00
  Data variables:
      Salt     (time, k, face, j, i) >f4 dask.array<shape=(9030, 3, 13, 4320, 4320), chunksize=(1, 1, 3, 4320, 4320)>
      Theta    (time, k, face, j, i) >f4 dask.array<shape=(9030, 3, 13, 4320, 4320), chunksize=(1, 1, 3, 4320, 4320)>


A list of all available variables can be seen as follows::

  >>> print(model.varnames)
  ['Eta', 'KPPhbl', 'oceFWflx', 'oceQnet', 'oceQsw', 'oceSflux', 'oceTAUX',
  'oceTAUY', 'PhiBot', 'Salt', 'SIarea', 'SIheff', 'SIhsalt', 'SIhsnow',
  'SIuice', 'SIvice', 'Theta', 'U', 'V', 'W']

The full set of options for these commands is enumerated at
:meth:`xmitgcm.llcreader.BaseLLCModel.get_dataset`.



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



API Documentation
-----------------

.. autoclass:: xmitgcm.llcreader.BaseLLCModel
  :members:
