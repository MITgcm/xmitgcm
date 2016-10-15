# xmitgcm

[![Build Status](https://travis-ci.org/xgcm/xmitgcm.svg?branch=master)](https://travis-ci.org/xgcm/xmitgcm)
[![codecov.io](https://codecov.io/github/xgcm/xgcm/coverage.svg?branch=master)](https://codecov.io/github/xgcm/xgcm?branch=master)
[![Documentation Status](https://readthedocs.org/projects/xmitgcm/badge/?version=latest)](http://xmitgcm.readthedocs.io/en/latest/?badge=latest)

xmitgcm is a python package for reading [MITgcm](http://mitgcm.org/) binary
[MDS](http://mitgcm.org/public/r2_manual/latest/online_documents/node277.html)
files into [xarray](http://xarray.pydata.org) data structures. By storing data
in [dask](http://dask.readthedocs.org) arrays, xmitgcm enables parallel,
[out-of-core](https://en.wikipedia.org/wiki/Out-of-core_algorithm) analysis
of MITgcm output data.

For more information, please consult

- [The online documentation](http://xmitgcm.readthedocs.io/en/latest)
- [The GitHub repository](https://github.com/xgcm/xmitgcm)

xmitgcm was developed by [Ryan Abernathey](http://rabernat.github.io).
