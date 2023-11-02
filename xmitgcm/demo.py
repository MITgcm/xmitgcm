
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
