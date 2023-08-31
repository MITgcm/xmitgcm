
from collections.abc import Container
import json

import numpy as np

class VarZ():
    def __init__(self, path: str, varname: str):
        self.path = path
        self.varname = varname

    def __getitem__(self, key):
        print(f"getitem: {key}")
        if key == ".zattrs":
            # How to guess this?
            return json.dumps({
                # "_ARRAY_DIMENSIONS": ["lon", "lat", "depth"]
                "_ARRAY_DIMENSIONS": ["lat"]
            })
        elif key == ".zarray":
            return json.dumps({
                # "chunks": [ 20, 30, 2 ],
                "chunks": [ 3 ],
                "compressor": None, # fixed
                # "dtype": ">f4",
                "dtype": "<f8",
                "fill_value": "NaN", # fixed
                "filters": None, # fixed
                "order": "C", # fixed
                #"shape": [ 20, 30, 2 ],
                "shape": [ 3 ],
                "zarr_format": 2 # fixed
            })
        elif key == "0":
            lat = np.array([10.,11.,12.]).tobytes()
            return lat

        return super().__getitem__(key)
