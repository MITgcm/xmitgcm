xmitgcm: Read MITgcm mds binary files into xarray
=================================================

|pypi| |Build Status| |codecov| |docs|

xmitgcm is a python package for reading MITgcm_ binary MDS files into
xarray_ data structures. By storing data in dask_ arrays, xmitgcm enables
parallel, out-of-core_ analysis of MITgcm output data.

Links
-----

-  HTML documentation: http://xmitgcm.readthedocs.org
-  Issue tracker: http://github.com/xgcm/xmitgcm/issues
-  Source code: http://github.com/xgcm/xmitgcm

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

.. |DOI| image:: https://zenodo.org/badge/14957/pyqg/pyqg.svg
   :target: https://zenodo.org/badge/latestdoi/14957/pyqg/pyqg
.. |Build Status| image:: https://travis-ci.org/xgcm/xmitgcm.svg?branch=master
   :target: https://travis-ci.org/xgcm/xmitgcm
   :alt: travis-ci build status
.. |codecov| image:: https://codecov.io/github/xgcm/xmitgcm/coverage.svg?branch=master
   :target: https://codecov.io/github/xgcm/xmitgcm?branch=master
   :alt: code coverage
.. |pypi| image:: https://badge.fury.io/py/xmitgcm.svg
   :target: https://badge.fury.io/py/xmitgcm
   :alt: pypi package
.. |docs| image:: http://readthedocs.org/projects/xmitgcm/badge/?version=stable
   :target: http://xmitgcm.readthedocs.org/en/stable/?badge=stable
   :alt: documentation status

.. _dask: http://dask.pydata.org
.. _xarray: http://xarray.pydata.org
.. _Comodo: http://pycomodo.forge.imag.fr/norm.html
.. _issues: https://github.com/xgcm/xmitgcm/issues
.. _`pull requests`: https://github.com/xgcm/xmitgcm/pulls
.. _MITgcm: http://mitgcm.org/public/r2_manual/latest/online_documents/node277.html
.. _out-of-core: https://en.wikipedia.org/wiki/Out-of-core_algorithm
.. _Anaconda: https://www.continuum.io/downloads
