from cachetools import cached
import os
import fnmatch

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

@cached(cache={})
def listdir_startsandendswith(path, start, end):
    files = listdir(path)
    return [f for f in files if f.endswith(end) and f.startswith(start)]

@cached(cache={})
def listdir_fnmatch(path, pattern):
    files = listdir(path)
    return [f for f in files if fnmatch.fnmatch(pattern)]
