Development
===========

Release History
---------------

v.0.3.0 (???)
~~~~~~~~~~~~~
  - Fix to keep proper Comodo metadata when swapping dimensions.

v0.2.0 (2017-02-14)
~~~~~~~~~~~~~~~~~~~~

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

.. _git repository: https://github.com/xgcm/xmitgcm
.. _forking instructions: https://help.github.com/articles/fork-a-repo/

Clone your fork on your local machine.

.. code-block:: bash

    $ git clone git@github.com:USERNAME/xmitgcm

(In the above, replace USERNAME with your github user name.)

Then set your fork to track the upstream xmitgcm repo.

.. code-block:: bash

    $ cd xmitgcm
    $ git remote add upstream git://github.com/xgcm/xmitgcm.git

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
