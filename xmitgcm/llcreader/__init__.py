try:
    import fsspec
except ImportError:
    raise ImportError('The llcreader module requires fsspec to be installed. '
                      'To install it, run `pip install fsspec`. See '
                      'https://filesystem-spec.readthedocs.io/en/latest/ '
                      'for more info.')

from .known_models import *
from .stores import *
from .llcmodel import BaseLLCModel
