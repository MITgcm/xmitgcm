
from collections.abc import Container
import json

class VarZ():
    def __init__(self, path: str, varname: str):
        self.path = path
        self.varname = varname
        self._ndims = 1 # From ["time"]

    def __getitem__(self, key):
        print(f"getitem: {key}")
        if key == ".zattrs":
            # How to guess this?
            return json.dumps({
                # "_ARRAY_DIMENSIONS": ["lon", "lat", "depth"]
                "_ARRAY_DIMENSIONS": ["time"]
            })
        elif key == ".zarray":
            return json.dumps({
                # "chunks": [ 20, 30, 2 ],
                "chunks": [ 2 ],
                "compressor": None, # fixed
                # "dtype": ">f4",
                "dtype": "<i8",
                "fill_value": "NaN", # fixed
                "filters": None, # fixed
                "order": "C", # fixed
                #"shape": [ 20, 30, 2 ],
                "shape": [ 2 ],
                "zarr_format": 2 # fixed
            })
        return super().__getitem__(key)
