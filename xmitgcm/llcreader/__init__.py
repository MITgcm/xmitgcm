from packaging import version

try:
    import fsspec
    assert version.parse(fsspec.__version__) >= version.parse("0.4.4")
except (ImportError, AssertionError): # pramga: no cover
    raise ImportError('The llcreader module requires fsspec version 0.4.4 or '
                      'or greater to be installed. '
                      'To install it, run `pip install fsspec`. See '
                      'https://filesystem-spec.readthedocs.io/en/latest/ '
                      'for more info.')

try:
    import zarr
    assert version.parse(zarr.__version__) >= version.parse("2.3.1")
except (ImportError, AssertionError): # pramga: no cover
    raise ImportError('The llcreader module requires zarr version 2.3.1 or '
                      'or greater to be installed. '
                      'To install it, run `pip install zarr`. See '
                      'https://zarr.readthedocs.io/en/stable/index.html#installation '
                      'for more info.')


from .known_models import *
from .stores import *
from .llcmodel import BaseLLCModel, faces_dataset_to_latlon
