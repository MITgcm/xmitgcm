
Reading MDS Data
================

open_mdsdataset
---------------

All loading of data in xmitgcm occurs through the function ``open_mdsdataset``.
Its full documentation below enumerates all the possible options.

.. autofunction:: xmitgcm.open_mdsdataset

The optional arguments are explained in more detail in the following sections
and examples.

Lazy Evaluation and Dask Chunking
---------------------------------

``open_mdsdataset`` does not actually load all of the data into memory
when it is invoked. Rather, it opens the files using
`numpy.memmap <http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html>`_,
which means that the data is not read until it is actually needed for
a computation. This is a cheap form of
`lazy evaluation <https://en.wikipedia.org/wiki/Lazy_evaluation>`_.

Additional performance benefits can be achieved with
`xarray dask chunking <http://xarray.pydata.org/en/stable/dask.html>`_::

    ds_chunked = open_mdsdataset(data_dir, chunks={'Z':1, 'Zl':1})

In the example above, the each horizontal slice of the model is assigned to its
own chunk; dask_ will automatically try to parellize operations across these
chunks using all your CPU cores. For this small example dataset, no performance
boost is expected (due to the overhead of parallelization), but very large
simulations will certainly benefit from chunking.

When chunking is applied, the data are represetned by `dask`_ arrays, and all
operations are evaluated lazily. No computation actually takes place until you
call ``.load()``::

    # take the mean of the squared zonal velocity
    (ds_chunked.U**2).mean()
    >>> <xarray.DataArray 'U' ()>
    dask.array<mean_ag..., shape=(), dtype=float32, chunksize=()>
    # load the value and execute the dask computational graph
    >>> <xarray.DataArray 'U' ()>
    array(0.00020325234800111502, dtype=float32)

.. note::
    In certain cases, chunking is applied automatically. These cases are

    - If there is more than one timestep to be read (see :ref:`expected-files`)
      the time dimension is automatically chunked.
    - In llc simulations, (see :ref:`geometries`) the ``face`` dimension is
      automatically chunked.

.. _expected-files:

Expected Files
--------------

MITgcm writes MDS output as pairs of files with the suffixes ``*.data`` and
``*.meta``. The model timestep iteration number is represented by a
ten-digit number at the end of the filename, e.g. ``T.0000000090.data`` and
``T.0000000090.meta``. MDS files without an iteration number are grid files.

xmitgcm has certain expectations regarding the files that will be present in
``datadir``. In particular, it assumes ``datadir`` is an MITgcm "run" directory.
By default, `open_mdsdataset` will read the grid files which describe the
geometry of the computational domain. If these files are not present in
``datadir``, this behavior can be turned off by setting ``read_grid=False``.

In order to determine the dimensions of the model domain ``open_mdsdataset``
needs to peek at the metadata in two grid files: ``XC.meta`` and ``RC.meta``.
(even when ``read_grid=False``). If these files are not available, you have the
option to manually specify the parameters ``nx``, ``ny``, and ``nz`` as keyword
arguments to ``open_mdsdataset``. (``ny`` is not required for
``geometry='llc'``).

By default, ``open_mdsdataset`` attempts to read all the data files in
``datadir``. The files and iteration numbers to read are determined in the
following way:

#. First ``datadir`` is scanned to determine all iteration numbers
   present in the directory. To override this behavior and manually specify the
   iteration numbers to read, use the ``iters`` keyword argument, e.g.
   ``iters=[10,20,30]``.

#. To dertmine the file prefixes to read, ``open_mdsdataset`` looks for all
   ``*.data`` filenames which match the *first* iteration number. To override
   this behavior and manually specify the file prefixes via the ``prefix``
   keyword argument, e.g. ``prefix=['UTave', 'VTave']``.

#. ``open_mdsdataset`` then looks for each file prefix at each iteration
   number.

This approach works for the test examples, but perhaps it does not suit your
model configuration. Suggestions are welcome on how to improve the discovery of
files in the form of issues_ and `pull requests`_.

.. warning::
   If you have certain file prefixes that are present at the first iteration
   (e.g. ``T.0000000000.data``) but not at later iterations (e.g
   ``iters=[0,10]``) but there is no ``T.0000000010.data`` file,
   ``open_mdsdataset`` will raise an error because it can't find the expected
   files. To overcome this you need to manually specify ``iters`` and / or
   ``prefix``.

To determine the variable metadata, xmitgcm is able to parse the
model's ``available_diagnostics.log`` file. If you use diagnostic output,
the ``available_diagnostics.log`` file corresponding with your model run should
be present in ``datadir``.

.. note::
    If the ``available_diagnostics.log`` file can't be found, a
    `default version <https://github.com/xgcm/xmitgcm/blob/master/xmitgcm/variables.py>`_
    will be used. This could lead to problems, since you may have custom
    diagnostics enabled in your run that are not present in the default.
    The default ``available_diagnostics.log`` file was taken from the ECCOv4
    ``global_oce_llc90`` experiment.

For non-diagnostic output (e.g. default "state" or "timeave" output), xmitgcm
assigns the variable metadata based on filenames. The additional metadata
makes the internal represtation of the model variables more verbose and
ensures compliance with `CF Conventions`_.

Dimensions and Coordinates
--------------------------

One major advantage of using xarray_ to represent data is that the variable
dimensions are *labeled*, much like netCDF data structures. This labeling
enables much clearer code. For example, to take a time average of a Dataset,
one just says ``ds.mean(dim='time')`` without having to remember which logical
axis is the time dimension.

xmitgcm distinguishes between *logical dimensions* and *physical dimensions* or
coordinates. Open ``open_mdsdataset`` will attempt to assign physical
dimensions to the data. The physical dimensions correspond to the axes of
the MITgcm grids in ``cartesian`` or ``sphericalpolar`` coordinates. The
standard names have been assigned according to `CF Conventions`_.

+----------------------+---------------------------------------+
| name                 | standard_name                         |
+======================+=======================================+
| time                 | time                                  |
+----------------------+---------------------------------------+
| XC                   | longitude                             |
+----------------------+---------------------------------------+
| YC                   | latitude                              |
+----------------------+---------------------------------------+
| XG                   | longitude_at_f_location               |
+----------------------+---------------------------------------+
| YG                   | latitude_at_f_location                |
+----------------------+---------------------------------------+
| Zl                   | depth_at_upper_w_location             |
+----------------------+---------------------------------------+
| Zu                   | depth_at_lower_w_location             |
+----------------------+---------------------------------------+
| Z                    | depth                                 |
+----------------------+---------------------------------------+
| Zp1                  | depth_at_w_location                   |
+----------------------+---------------------------------------+
| layer_1RHO_center    | ocean_layer_coordinate_1RHO_center    |
+----------------------+---------------------------------------+
| layer_1RHO_interface | ocean_layer_coordinate_1RHO_interface |
+----------------------+---------------------------------------+
| layer_1RHO_bounds    | ocean_layer_coordinate_1RHO_bounds    |
+----------------------+---------------------------------------+

The physical dimensions of typical veriables are::

    print(ds.THETA.dims)
    >>> ('time', 'Z', 'YC', 'XC')
    print(ds.UVEL.dims)
    >>> ('time', 'Z', 'YC', 'XG')
    print(ds.VVEL.dims)
    >>> ('time', 'Z', 'YG', 'XC')
    print(ds.WVEl.dims)
    >>> ('time', 'Zl', 'YC', 'XG')

In order for physical dimensions to be assigned ``open_mdsdataset`` must be
involved with ``read_grid=True`` (default). For a more minimalistic approach,
one can use ``read_grid=False`` and assign logcial dimensions. Logical
dimensions can also be chosen by explicitly setting ``swap_dims=False``, even
with ``read_grid=False``.
Physical dimension only work with ``geometry=='cartesian'`` or
``geometry=='sphericalpolar'``. For ``geometry=='llc'`` or ``geometry=='curvilinear'``,
it is not possible to replace the logical dimensions with physical dimensions, and setting
``swap_dims=False`` will raise an error.

Logical dimensions follow the naming conventions of the
`MITgcm numerical grid <http://mitgcm.org/public/pelican/online_documents/node42.html>`_.
The dimension names are augmented with metadata attributes from the Comodo_
conventions. These logical spatial dimensions are

+------+----------------------------------+------+-------------------+
| name | standard_name                    | axis | c_grid_axis_shift |
+======+==================================+======+===================+
| i    | x_grid_index                     | X    |                   |
+------+----------------------------------+------+-------------------+
| i_g  | x_grid_index_at_u_location       | X    | -0.5              |
+------+----------------------------------+------+-------------------+
| j    | y_grid_index                     | Y    |                   |
+------+----------------------------------+------+-------------------+
| j_g  | y_grid_index_at_v_location       | Y    | -0.5              |
+------+----------------------------------+------+-------------------+
| k    | z_grid_index                     | Z    |                   |
+------+----------------------------------+------+-------------------+
| k_u  | z_grid_index_at_lower_w_location | Z    | -0.5              |
+------+----------------------------------+------+-------------------+
| k_l  | z_grid_index_at_upper_w_location | Z    | 0.5               |
+------+----------------------------------+------+-------------------+
| k_p1 | z_grid_index_at_w_location       | Z    | (-0.5, 0.5)       |
+------+----------------------------------+------+-------------------+

As explained in the Comodo_ documentation, the use of different dimensions is
necessary to represent the fact that, in c-grid ocean models, different
variables are staggered in different ways with respect to the model cells.
For example, tracers and velocity components are all have different logical
dimensions::

    print(ds.THETA.dims)
    >>> ('time', 'k', 'j', 'i')
    print(ds.UVEL.dims)
    >>> ('time', 'k', 'j', 'i_g'
    print(ds.VVEL.dims)
    >>> ('time', 'k', 'j_g', 'i')
    print(ds.WVEl.dims)
    >>> ('time', 'k_l', 'j', 'i')

xarray_ distinguishes between "coordinates" and "data_vars". By default,
``open_mdsdataset`` will promote all grid variables to coordinates. To turn off
this behavior and treat grid variables as data_vars, use
``grid_vars_to_coords=False``.

Time
----

``open_mdsdataset`` attemts to determine the time dimension based on ``iters``.
However, addtiional input is required from the user to fully exploit this
capability. If the user specifies ``delta_t``, the numerical timestep used for
the MITgcm simulation, it is used to muliply ``iters`` to determine the time
in seconds. Additionally, if the user specifies ``ref_date`` (an ISO date
string, e.g. ``“1990-1-1 0:0:0”``), the time dimension will be converted into
a datetime index, exposing all sorts of
`useful timeseries functionalty <http://xarray.pydata.org/en/stable/time-series.html>`_
within xarray.

.. _geometries:

Grid Geometries
---------------

The grid geometry is not inferred; it must be specified via the ``geometry``
keyword. xmitgcm currently supports four MITgcm grid geometries: ``cartesian``,
``sphericalpolar``, ``curvilinear``, and ``llc``.  The first two are straightforward.
The ``curvilinear`` is used for curvilinear cartesian grids. The ``llc``
("lat-lon-cap") geometry is more complicated. This grid consists of four
distinct faces of the same size plus a smaller north-polar cap. Each face has a
distinct relatioship between its logical dimensions and its physical
coordinates. Because netCDF and xarray.Dataset data structures do not support
this sort of complex geometry (multiple faces of different sizes), our approach,
inspired by nc-tiles, is to split the domain into 13 equal-sized "faces".
``face`` then becomes an additional dimension of the data.

To download an example llc dataset, run the following shell commands::

    $ curl -L -J -O https://ndownloader.figshare.com/files/6494721
    $ tar -xvzf global_oce_llc90.tar.gz

And to read it, in python::

    ds_llc = open_mdsdataset('./global_oce_llc90/', iters=8, geometry='llc')
    print(ds_llc['S']dims)
    >>> ('time', 'k', 'face', 'j', 'i')

xmitgcm is not nearly as comprehensive as gcmfaces_. It does not offer
sophisticated operations involving exchanges at face boundaries, integrals
across sections, etc. The goal of this package is simply to read the mds data.
However, by outputing an xarray_ data structure, we can use all of xarray's
usual capabilities on the llc data, for example::

    # calculate the mean squared salinity as a function of depth
    (ds_llc.S**2).mean(dim=['face', 'j', 'i'])
    >>> <xarray.DataArray 'S' (time: 1, k: 50)>
    dask.array<mean_ag..., shape=(1, 50), dtype=float32, chunksize=(1, 50)>
    Coordinates:
      * k        (k) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
        iter     (time) int64 8
      * time     (time) int64 8
        Z        (k) >f4 -5.0 -15.0 -25.0 -35.0 -45.0 -55.0 -65.0 -75.005 ...
        PHrefC   (k) >f4 49.05 147.15 245.25 343.35 441.45 539.55 637.65 735.799 ...
        drF      (k) >f4 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.01 10.03 10.11 ...

(Note that this simple example does not perform correct volume weighting or
land masking in the average.)

``open_mdsdataset`` offers two different strategies for reading LLC data,
``method='smallchunks'`` (the default) and ``method='bigchunks'``. The details
and tradeoffs of these strategies are described in :doc:`/performance`.

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
