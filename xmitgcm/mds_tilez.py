
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


@dataclass()
class VarZ():
    varname: str
    data: dict
    chunks: tuple
    dtype: str
    fill_value: float
    shape: tuple

    def __getitem__(self, key):
        print(f"VarZ.getitem: {key}")

        assert key[:len(self.varname)+1] == f"{self.varname}/"
        k = key[len(self.varname)+1:]
        if k == f".zattrs":
            return self._zattrs()
        elif k == f".zarray":
            return self._zarray()
        else:
            ti, xyi = k.split(".", 1)
            ts = sorted(v.data.keys())[int(ti)]
            return self.data[ts][xyi]

    def __iter__(self):
        print(f"VarZ.__iter__()")
        yield from [
            f"{self.varname}/.zattrs",
            f"{self.varname}/.zarray",
        ]
        idx_t = sorted(self.data.keys())
        for ts in idx_t:
            yield from (f"{self.varname}/{idx_t.index(ts)}.{i}"
                        for i in self.data[ts])

    def _zattrs(self):
        # How to guess this?
        if len(self.shape) == 3:
            dims = ["time", "lon", "lat", "depth"]
        elif len(self.shape) == 2:
            dims = ["time", "lon", "lat"]

        return json.dumps({
            # "_ARRAY_DIMENSIONS": ["lon", "lat", "depth"]
            "_ARRAY_DIMENSIONS": dims
        })

    def _zarray(self):
            return json.dumps({
                "chunks": self.chunks,
                "compressor": None, # fixed
                "dtype": self.dtype,
                "fill_value": self.fill_value,
                "filters": None, # fixed
                "order": "C", # fixed
                "shape": self.shape,
                "zarr_format": 2 # fixed
            })

    def push(self, chunk):
        assert self.varname in chunk.varnames
        assert self.chunks == chunk.chunks
        assert self.dtype == chunk.dtype
        assert self.shape == chunk.shape
        assert self.fill_value == chunk.missing_value

        if chunk.time_step_number not in self.data:
            self.data[chunk.time_step_number] = {}
        self.data[chunk.time_step_number][chunk.index] = chunk[self.varname]

    def push_from_meta(self, filename):
        d = Chunk.from_meta(filename)
        self.push(d)

    @staticmethod
    def from_chunk(chunk, varname=None):
        if varname is None:
            assert len(chunk.varnames) == 1
            varname = chunk.varnames[0]

        assert varname in chunk.varnames

        v = VarZ(varname=varname, data={}, chunks=chunk.chunks,
                 shape=chunk.shape,
                 dtype=chunk.dtype, fill_value=chunk.missing_value)
        v.push(chunk)
        return v

    @staticmethod
    def from_meta(filename, varname=None):
        return VarZ.from_chunk(Chunk.from_meta(filename))
