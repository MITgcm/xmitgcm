
from collections.abc import Container
import json

import numpy as np

class VarZ():
    def __init__(self, path: str, varname: str):
        self.path = path
        self.varname = varname

    def __getitem__(self, key):
        print(f"VarZ.getitem: {key}")
        if key == "lat/.zattrs":
            return self._zattrs()
        elif key == "lat/.zarray":
            return self._zarray()
        elif key == "lat/0":
            return [ f"{self.path}/lat/0", 0, 24]
            # np.array([11.,12.,13.]).tobytes()

    def __iter__(self):
        print(f"VarZ.__iter__()")
        yield from ['lat/.zattrs', 'lat/.zarray', 'lat/0']

    def _zattrs(self):
        # How to guess this?
        return json.dumps({
            # "_ARRAY_DIMENSIONS": ["lon", "lat", "depth"]
            "_ARRAY_DIMENSIONS": ["lat"]
        })

    def _zarray(self):
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

    def items(self):
        print(f"VarZ.items()")
        yield from [
            # ('lat/.zattrs', ['/Users/castelao/work/projects/others/MIT_tiles/data/example/lat/.zattrs', 0, 50]),
            ('lat/.zattrs', self._zattrs()),
            ('lat/.zarray', self._zarray()),
            ('lat/0', [f"{self.path}/lat/0", 0, 24])
        ]

    def values(self):
        print(f"VarZ.values()")
        yield from [
            self._zattrs(),
            self._zarray(),
            [f"{self.path}/lat/0", 0, 24]
        ]

