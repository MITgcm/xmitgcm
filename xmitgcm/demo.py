
import os

from .utils import parse_meta_file


def collect_meta(path: str):

    output = {}
    for m in (m for m in os.listdir(path) if m[-5:] == ".meta"):
        try:
            meta = parse_meta_file(os.path.join(path, m))
            if meta["basename"] not in output:
                output[meta["basename"]] = []
            meta["filename"] = m
            output[meta["basename"]].append(meta)
        except:
            print(f"Failed parsing: {m}")

    return output


class DataFile(UserDict):
    def get(self, *args, **kwargs):
        print(f"get: {args}")
        # import pdb; pdb.set_trace()
        return super().get(*args)

    def __getitem__(self, key):
        print(f"getitem: {key}")
        # if "S/0.0.0" not in self:
        #    super().__setitem__("S/0.0.0", [f"{path_mitgcm}/S/0.0.0", 0, 4800])

        if key == ".zgroup":
            return json.dumps({"zarr_format": 2})
        elif key == "S/.zattrs":
            # How to guess this?
            return json.dumps({
                "_ARRAY_DIMENSIONS": ["lon", "lat", "depth"]
            })
        elif key == "S/.zarray":
            return json.dumps({
                "chunks": [ 20, 30, 2 ],
                "compressor": None, # fixed
                "dtype": ">f4",
                "fill_value": "NaN", # fixed
                "filters": None, # fixed
                "order": "C", # fixed
                "shape": [ 20, 30, 2 ],
                "zarr_format": 2 # fixed
            })
            # return [f"{path_mitgcm}/S/.zarray", 0, 356]
        elif key == "S/0.0.0":
            import pdb; pdb.set_trace()
            # size can guess from chunks * dtype
            # super().__setitem__("S/0.0.0", [f"{path_mitgcm}/S.0000000000.001.001.data", 0, 4800])
            return [f"{path_mitgcm}/S.0000000000.001.001.data", 0, 4800]
        return super().__getitem__(key)

    def __contains__(self, item):
        print(f"contains: {item}")
        if item in (".zgroup", "S/.zattrs", "S/.zarray", "S/0.0.0"):
            return True
        else:
            print(f"Checking on {item}")
        # if item == "S/0.0.0":
        #     # import pdb; pdb.set_trace()
        #    print(super().__contains__(item))
        return super().__contains__(item)

    def __iter__(self):
        print("iter")
        yield from (".zgroup", "S/.zattrs", "S/.zarray", "S/0.0.0")
