 #!/usr/bin/env python
import os
import re
import sys
import warnings

from setuptools import setup, find_packages

VERSION = '0.2.0'
DISTNAME = 'xmitgcm'
LICENSE = 'Apache'
AUTHOR = 'Ryan Abernathey'
AUTHOR_EMAIL = 'rpa@ldeo.columbia.edu'
URL = 'https://github.com/xgcm/xmitgcm'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Topic :: Scientific/Engineering',
]

INSTALL_REQUIRES = ['xarray >= 0.8.2', 'dask >= 0.12']
SETUP_REQUIRES = ['pytest-runner']
TESTS_REQUIRE = ['pytest >= 2.8', 'coverage']

DESCRIPTION = "Read MITgcm mds binary files into xarray"
def readme():
    with open('README.rst') as f:
        return f.read()

setup(name=DISTNAME,
      version=VERSION,
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
      packages=find_packages())
