 #!/usr/bin/env python
import os
import re
import sys
import warnings
import versioneer

from setuptools import setup, find_packages

DISTNAME = 'xmitgcm'
LICENSE = 'Apache'
AUTHOR = 'Ryan Abernathey'
AUTHOR_EMAIL = 'rpa@ldeo.columbia.edu'
URL = 'https://github.com/MITgcm/xmitgcm'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering',
]

INSTALL_REQUIRES = ['xarray >= 0.11.0', 'dask >= 1.0', 'cachetools']
SETUP_REQUIRES = ['pytest-runner']
TESTS_REQUIRE = ['pytest >= 4.0', 'coverage']

DESCRIPTION = "Read MITgcm mds binary files into xarray"
def readme():
    with open('README.rst') as f:
        return f.read()

setup(name=DISTNAME,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      license=LICENSE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=readme(),
      install_requires=INSTALL_REQUIRES,
      setup_requires=SETUP_REQUIRES,
      tests_require=TESTS_REQUIRE,
      url=URL,
      packages=find_packages(),
      python_requires='>=3.7')
