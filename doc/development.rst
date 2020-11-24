Development
===========

Release History
---------------


v0.5.0 (2020-24-11)
~~~~~~~~~~~~~~~~~~~

  - Add mask override option to llcreader from Ryan Abernathey (:issue:`191`)
  - Python 3.8 compatibility from Takaya Uchida (:issue:`194`)
  - llcreader: get grid info from ECCO portal from Tim Smith (:issue:`158`, :issue:`161`, :issue:`166`)
  - Fix python 3.6 master build from Tim Smith (:issue:`200`)
  - ECCO portal new iter_stop from Antonio Quintana (:issue:`193`)
  - Added missing grid variables to llcreader known_models from Ryan Abernathey (:issue:`207`)
  - Migrated to GitHub Actions from Tim Smith (:issue:`223`)
  - Dropped python 2 from Tim Smith (:issue:`226`)
  - llcreader klevels bugfix from Ryan Abernathey (:issue:`224`)
  - Incorporated llcreader for ASTE release 1 from Tim Smith (:issue:`231`)
  - Fixed typo for 'coordinate' entry in dimensions dictionary from Ian Fenty (:issue:`236`)
  - Lazy open_mdsdataset from Pascal Bourgault (:issue:`229`)
  - Implemented checking for variable mates from Fraser Goldsworth (:issue:`234`)
  - Added metadata to llcreader dimensions from Ryan Abernathey (:issue:`239`)
  - Cehck iter_start and iter_stop from Fraser Goldsworth (:issue:`235`)
  - Automated release to pypi from from Ryan Abernathey (:issue:`241`)

v0.4.1 (2019-07-11)
~~~~~~~~~~~~~~~~~~~

  - Incorporated llcreader bugfix from Spencer Jones (:issue:`154`)

v0.4.0 (2019-07-11)
~~~~~~~~~~~~~~~~~~~

  - New :doc:`llcreader` module (see
    `blog post <https://medium.com/pangeo/petabytes-of-ocean-data-part-1-nasa-ecco-data-portal-81e3c5e077be>`_
    for more details.)


v0.3.0 (2019-05-19)
~~~~~~~~~~~~~~~~~~~~
  - Ability to read ASTE grids
  - Ability to read seaice and thsice native output
  - Reading of optional grid files
  - Moved test data to figshare
  - Writing of binary files
  - Xarray 0.12 compatibility
  - Ability to read 2D slice diagnostics of 3D fields


v.0.2.2 (2018-07-18)
~~~~~~~~~~~~~~~~~~~~
  - Extend capabilities of read_raw_data (:issue:`84`)
  - Fix the problem with testing type of prefix (:issue:`83`)
  - Cast prefix to list if it isn't already one (:issue:`79`)
  - Generalizes _get_all_iternums in order to handle compressed data (:issue:`77`)
  - Extract version number from git tag (:issue:`72`)
  - Adding .stickler.yml (:issue:`70`)
  - Added functionality to read PTRtave files (:issue:`63`)
  - Update examples.rst (:issue:`65`)
  - fix time encoding (:issue:`61`)
  - Fix llc chunking (:issue:`60`)
  - Test refactor (:issue:`54`)
  - Kpp added properly (:issue:`55`)
  - Tests for ref_date issue (:issue:`53`)
  - Add python 3.6 testing (:issue:`52`)
  - Added layers axis attribute (:issue:`47`)

v.0.2.1 (2017-05-31)
~~~~~~~~~~~~~~~~~~~~
  - Fix to ensure that grid indices are always interger dtype.
  - Fix to keep proper Comodo metadata when swapping dimensions.

v0.2.0 (2017-02-14)
~~~~~~~~~~~~~~~~~~~

This release contains the following feature enhancements:
  - Files are not read until the data are accessed. This helps overcome a common
    "too many open files issue" (:issue:`11`).
  - A workaround for missing ``.meta`` files (:issue:`12`).
  - Option for a separate ``grid_dir`` in case it is different from ``data_dir``
    (:issue:`13`).
  - Refactor of the way LLC data is read which allows for more efficient chunking
    and lower memory usage (:issue:`20`)
  - Bug fix related to the handling of `default_dtype` parameter (:issue:`34`).
    By `Guillaume Sérazin <https://github.com/serazing>`_.
  - Support for older MITgcm versions that write a different lenght ``drC``
    variable (:issue:`8`). By `Liam Brannigan <https://github.com/braaannigan>`_.
  - Support for cartesian curvilinear grids. By
    `Andrea Cimatoribus <https://github.com/sambarluc>`_.
  - Expanded and improved documentation.

Unless otherwise noted, all updates are by
`Ryan Abernathey <http://github.com/rabernat>`_.

v0.1.0 (2016-10-15)
~~~~~~~~~~~~~~~~~~~

Initial release.

Develpment Workflow
-------------------

Anyone interested in helping to develop xmitgcm needs to create their own fork
of our `git repository`. (Follow the github `forking instructions`_. You
will need a github account.)

.. _git repository: https://github.com/MITgcm/xmitgcm
.. _forking instructions: https://help.github.com/articles/fork-a-repo/

Clone your fork on your local machine.

.. code-block:: bash

    $ git clone git@github.com:USERNAME/xmitgcm

(In the above, replace USERNAME with your github user name.)

Then set your fork to track the upstream xmitgcm repo.

.. code-block:: bash

    $ cd xmitgcm
    $ git remote add upstream git://github.com/MITgcm/xmitgcm.git

You will want to periodically sync your master branch with the upstream master.

.. code-block:: bash

    $ git fetch upstream
    $ git rebase upstream/master

Never make any commits on your local master branch. Instead open a feature
branch for every new development task.

.. code-block:: bash

    $ git checkout -b cool_new_feature

(Replace `cool_new_feature` with an appropriate description of your feature.)
At this point you work on your new feature, using `git add` to add your
changes. When your feature is complete and well tested, commit your changes

.. code-block:: bash

    $ git commit -m 'did a bunch of great work'

and push your branch to github.

.. code-block:: bash

    $ git push origin cool_new_feature

At this point, you go find your fork on github.com and create a `pull
request`_. Clearly describe what you have done in the comments. If your
pull request fixes an issue or adds a useful new feature, the team will
gladly merge it.

.. _pull request: https://help.github.com/articles/using-pull-requests/

After your pull request is merged, you can switch back to the master branch,
rebase, and delete your feature branch. You will find your new feature
incorporated into xmitgcm.

.. code-block:: bash

    $ git checkout master
    $ git fetch upstream
    $ git rebase upstream/master
    $ git branch -d cool_new_feature

Virtual Environment
-------------------

This is how to create a virtual environment into which to test-install xmitgcm,
install it, check the version, and tear down the virtual environment.

.. code-block:: bash

    $ conda create --yes -n test_env python=3.5 xarray dask numpy pytest future
    $ source activate test_env
    $ pip install xmitgcm
    $ python -c 'import xmitgcm; print(xmitgcm.__version__);'
    $ source deactivate
    $ conda env remove --yes -n test_env
