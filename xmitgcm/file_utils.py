from cachetools import cached
import os

@cached(cache={})
def listdir(path):
    return os.listdir(path)

@cached(cache={})
def listdir_startswith(path, pattern):
    files = listdir(path)
    return [f for f in files if f.startswith(pattern)]

@cached(cache={})
def listdir_endswith(path, pattern):
    files = listdir(path)
    return [f for f in files if f.endswith(pattern)]
