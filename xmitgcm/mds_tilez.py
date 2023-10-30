
from collections.abc import Container
from collections import UserDict
from dataclasses import dataclass
import json
import os.path

from fsspec.implementations.reference import ReferenceFileSystem
import numpy as np
import xarray as xr

from xmitgcm.utils import parse_meta_file

@dataclass()
class Chunk():
    """
    Handles one chunk of one variable. For instance, it could be:
    S.0000000000.001.001.data
    S.0000000000.001.001.meta

    metafilename = "/Users/castelao/work/projects/others/MIT_tiles/data/mitgcm/S.0000000000.001.001.meta"
    """
    filename: str
    metadata: dict

    def __fspath__(self):
        return os.path.join(self.root, self._fdata)

    def __getitem__(self, key):
        return self._labels[key]

    @property
    def _labels(self):
        size = np.prod([(d[2] - d[1] + 1) for d in self.metadata['dimList']])
        size *= self.metadata['dataprec'].itemsize # f32

        # return {f"{v}/{self.index}": [self.filename, i*size, (i+1)*size] for i,v in enumerate(self.varnames)}
        return {v: [self.filename, i*size, (i+1)*size] for i,v in enumerate(self.varnames)}

    @property
    def varnames(self):
        try:
            return self._varnames
        except:
            if "fldList" in self.metadata:
                self._varnames = self.metadata['fldList']
            else:
                self._varnames = os.path.basename(self.filename).split('.')[0]
            return self._varnames

    @property
    def dtype(self):
        return self.metadata['dataprec'].str

    @property
    def index(self):
        idx = []
        for d in self.metadata['dimList']:
            idx.append(str((d[1] - 1) //  (d[2] - d[1]+1)))

        return ".".join(idx)

    @property
    def missing_value(self):
        if self.metadata['dataprec'].kind == 'i':
            return int(self.metadata['missingValue'])
        else:
            return float(self.metadata['missingValue'])

    @property
    def shape(self):
        return tuple(s[0] for s in x.metadata['dimList'])

    @property
    def chunks(self):
        return tuple(s[2] for s in x.metadata['dimList'])

    @property
    def time_step_number(self):
        return int(self.metadata['timeStepNumber'])

    @staticmethod
    def from_meta(filename):
        assert os.path.exists(filename.replace(".meta", ".data"))
        metadata = parse_meta_file(filename)
        chunk = Chunk(
            filename=filename.replace(".meta", ".data"),
            metadata=metadata)
        return chunk
