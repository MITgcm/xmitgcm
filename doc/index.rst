xmitgcm
=======

xmitgcm is a python package for reading MITgcm_ binary MDS files into
xarray_ data structures. By storing data in dask_ arrays, xmitgcm enables
parallel, out-of-core_ analysis of MITgcm output data.

Installation
------------

Requirements
^^^^^^^^^^^^

xmitgcm is compatible with python 3 and python 2.7. It requires xarray_
(>= version 0.8.2) and dask_ (>= version 0.11.2).
These packages are most reliably installed via the
`conda <http://conda.pydata.org/docs/>`_ environment management
system, which is part of the Anaconda_ python distribution. Assuming you have
conda available on your system, the dependencies can be installed with the
command::

    conda install xarray dask

If you are using earlier versions of these packages, you should update before
installing xmitgcm.

Installation via pip
^^^^^^^^^^^^^^^^^^^^

If you just want to use xmitgcm, the easiest way is to install via pip::

    pip install xmitgcm

This will automatically install the latest release from
`pypi <https://pypi.python.org/pypi>`_.

Installation from github
^^^^^^^^^^^^^^^^^^^^^^^^

xmitgcm is under active development. To obtain the latest development version,
you may clone the `source repository <https://github.com/xgcm/xmitgcm>`_
and install it::

    git clone https://github.com/xgcm/xmitgcm.git
    cd xmitgcm
    python setup.py install

Users are encouraged to `fork <https://help.github.com/articles/fork-a-repo/>`_
xmitgcm and submit issues_ _ and `pull requests`_.

Quick Start
-----------

First make sure you understand what an xarray_ Dataset object is. Then find
some MITgcm MDS data. If you don't have any data of your own, you can download
the xmitgcm
`test repositories <https://figshare.com/articles/xmitgcm_test_datasets/4033530>`_
To download the some test data, run the shell commands::

    $ curl -L -J -O https://ndownloader.figshare.com/files/6494718
    $ tar -xvzf global_oce_latlon.tar.gz

This will create a directory called ``global_oce_latlon`` which we will use
for the rest of these examples. If you have your own data, replace this with
the path to your mitgcm files.

To opean MITgcm MDS data as an xarray.Dataset, do the following in python::

    from xmitgcm import open_mdsdataset
    data_dir = './global_oce_latlon'
    ds = open_mdsdataset(data_dir)
    # display the contents of the Dataset
    print(ds)
    >>> <xarray.Dataset>
    Dimensions:               (XC: 90, XG: 90, YC: 40, YG: 40, Z: 15, Zl: 15, Zp1: 16, Zu: 15, layer_1RHO_bounds: 31, layer_1RHO_center: 30, layer_1RHO_interface: 29, time: 1)
    Coordinates:
        iter                  (time) int64 39600
      * time                  (time) int64 39600
      * XC                    (XC) >f4 2.0 6.0 10.0 14.0 18.0 22.0 26.0 30.0 ...
      * YC                    (YC) >f4 -78.0 -74.0 -70.0 -66.0 -62.0 -58.0 -54.0 ...
      * XG                    (XG) >f4 0.0 4.0 8.0 12.0 16.0 20.0 24.0 28.0 32.0 ...
      * YG                    (YG) >f4 -80.0 -76.0 -72.0 -68.0 -64.0 -60.0 -56.0 ...
      * Zl                    (Zl) >f4 0.0 -50.0 -120.0 -220.0 -360.0 -550.0 ...
      * Zu                    (Zu) >f4 -50.0 -120.0 -220.0 -360.0 -550.0 -790.0 ...
      * Z                     (Z) >f4 -25.0 -85.0 -170.0 -290.0 -455.0 -670.0 ...
      * Zp1                   (Zp1) >f4 0.0 -50.0 -120.0 -220.0 -360.0 -550.0 ...
        dxC                   (YC, XG) >f4 92460.4 92460.4 92460.4 92460.4 ...
        rAs                   (YG, XC) >f4 3.43349e+10 3.43349e+10 3.43349e+10 ...
        rAw                   (YC, XG) >f4 4.11097e+10 4.11097e+10 4.11097e+10 ...
        Depth                 (YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        rA                    (YC, XC) >f4 4.11097e+10 4.11097e+10 4.11097e+10 ...
        dxG                   (YG, XC) >f4 77223.1 77223.1 77223.1 77223.1 ...
        dyG                   (YC, XG) >f4 444710.0 444710.0 444710.0 444710.0 ...
        rAz                   (YG, XG) >f4 3.43349e+10 3.43349e+10 3.43349e+10 ...
        dyC                   (YG, XC) >f4 444710.0 444710.0 444710.0 444710.0 ...
        PHrefC                (Z) >f4 245.25 833.85 1667.7 2844.9 4463.55 6572.7 ...
        drC                   (Zp1) >f4 25.0 60.0 85.0 120.0 165.0 215.0 265.0 ...
        PHrefF                (Zp1) >f4 0.0 490.5 1177.2 2158.2 3531.6 5395.5 ...
        drF                   (Z) >f4 50.0 70.0 100.0 140.0 190.0 240.0 290.0 ...
        hFacS                 (Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        hFacC                 (Z, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        hFacW                 (Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
      * layer_1RHO_center     (layer_1RHO_center) float32 20.1999 20.6922 21.169 ...
      * layer_1RHO_interface  (layer_1RHO_interface) >f4 20.4499 20.9345 21.4034 ...
      * layer_1RHO_bounds     (layer_1RHO_bounds) >f4 19.9499 20.4499 20.9345 ...
    Data variables:
        tFluxtave             (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        PHLtave               (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        Stave                 (time, Z, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        UUtave                (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        LaHw1RHO              (time, layer_1RHO_center, YC, XG) >f4 0.0 0.0 0.0 ...
        LaPs1RHO              (time, layer_1RHO_center, YG, XC) >f4 0.0 0.0 0.0 ...
        LaHs1RHO              (time, layer_1RHO_center, YG, XC) >f4 0.0 0.0 0.0 ...
        LaUH1RHO              (time, layer_1RHO_center, YC, XG) >f4 0.0 0.0 0.0 ...
        LaVH1RHO              (time, layer_1RHO_center, YG, XC) >f4 0.0 0.0 0.0 ...
        LaPw1RHO              (time, layer_1RHO_center, YC, XG) >f4 0.0 0.0 0.0 ...
        UVtave                (time, Z, YG, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        uFluxtave             (time, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        VStave                (time, Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        VTtave                (time, Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        TTtave                (time, Z, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        PhHytave              (time, Z, YC, XC) >f4 -8.30019 -8.30019 -8.30019 ...
        sFluxtave             (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        W                     (time, Zl, YC, XC) >f4 -0.0 -0.0 -0.0 -0.0 -0.0 ...
        ETAtave               (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        VVtave                (time, Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        Ttave                 (time, Z, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        PH                    (time, Z, YC, XC) >f4 -8.30019 -8.30019 -8.30019 ...
        vVeltave              (time, Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        UTtave                (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        PHL2tave              (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        UStave                (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        uVeltave              (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        S                     (time, Z, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        Eta                   (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        Eta2tave              (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        DFxE_TH               (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        ADVy_TH               (time, Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        VTHMASS               (time, Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        DFrE_TH               (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        WTHMASS               (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        TOTTTEND              (time, Z, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        ADVr_TH               (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        DFyE_TH               (time, Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        UTHMASS               (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        DFrI_TH               (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        ADVx_TH               (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        surForcT              (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        TFLUX                 (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        surForcS              (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        SFLUX                 (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        V                     (time, Z, YG, XC) >f4 -0.0 -0.0 -0.0 -0.0 -0.0 0.0 ...
        LaTs1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        LTha1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        LaSz1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        LSto1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        LSha1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        LaTz1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        LaSs1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        LaTh1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        LTto1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        LTza1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        LaSh1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        LSza1RHO              (time, layer_1RHO_interface, YC, XC) >f4 0.0 0.0 ...
        vFluxtave             (time, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        WTtave                (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        WStave                (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        GM_Kwz-T              (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        PHL                   (time, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        THETA                 (time, Z, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        UVEL                  (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        VVEL                  (time, Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        WVEL                  (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        SALT                  (time, Z, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        DFrI_SLT              (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        WSLTMASS              (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        ADVx_SLT              (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        ADVr_SLT              (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        TOTSTEND              (time, Z, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        USLTMASS              (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        DFxE_SLT              (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        DFyE_SLT              (time, Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        DFrE_SLT              (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        ADVy_SLT              (time, Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        VSLTMASS              (time, Z, YG, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        Convtave              (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        wVeltave              (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        GM_Kwy-T              (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        U                     (time, Z, YC, XG) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        Tdiftave              (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        GM_Kwx-T              (time, Zl, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...
        T                     (time, Z, YC, XC) >f4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ...

``data_dir``, should be the path (absolute or relative) to an
MITgcm run directory. xmitgcm will automatically scan this directory and
try to determine the file prefixes and iteration numbers to read. In some
configurations, the ``open_mdsdataset`` function may work without further
keyword arguments. In most cases, you will have to specify further details.

open_mdsdataset
---------------

The only user-facing part of the xmitgcm api is the function
``open_mdsdataset``.
Here is its full documentation.

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
involved with ``read_grid=True`` (default). For a moree minimalistic approach,
one can use ``read_grid=False`` and assign logcial dimensions. Logical
dimensions can also be chosen by explicitly setting ``swap_dims=False``, even
with ``read_grid=False``.
Physical dimension only work with ``geometry=='cartesian'`` or
``geometry=='sphericalpolar'``. For ``geometry=='llc'``, it is not possible
to replace the logical dimensions with physical dimensions, and setting
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
keyword. xmitgcm currently supports three MITgcm grid geometries: ``cartesian``,
``sphericalpolar``, and ``llc``.  The first two are straightforward. The ``llc``
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

Note that this simple example does not perform correct volume weighting or
land masking in the average.




Example Usage
-------------

Once you have loaded your data, you can analyze it using all the capabilities
available in xarray_. Here are a few quick examples.

Land Masks
^^^^^^^^^^

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
^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^

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
