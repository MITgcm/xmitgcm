from packaging import version
try:
    import fsspec
    assert version.parse(fsspec.__version__) >= version.parse("0.2.1")
except (ImportError, AssertionError): # pramga: no cover
    raise ImportError('The llcreader module requires fsspec version 0.2.1 or '
                      'or greater to be installed. '
                      'To install it, run `pip install fsspec`. See '
                      'https://filesystem-spec.readthedocs.io/en/latest/ '
                      'for more info.')

from .known_models import *
from .stores import *
from .llcmodel import BaseLLCModel
