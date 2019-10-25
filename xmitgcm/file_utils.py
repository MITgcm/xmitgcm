import cachetools.func
import os
import fnmatch

cache_maxsize = 100
cache_ttl = 600 # tem minutes

@cachetools.func.ttl_cache(maxsize=cache_maxsize, ttl=cache_ttl)
def listdir(path):
    return os.listdir(path)

@cachetools.func.ttl_cache(maxsize=cache_maxsize, ttl=cache_ttl)
def listdir_startswith(path, pattern):
    files = listdir(path)
    return [f for f in files if f.startswith(pattern)]

@cachetools.func.ttl_cache(maxsize=cache_maxsize, ttl=cache_ttl)
def listdir_endswith(path, pattern):
    files = listdir(path)
    return [f for f in files if f.endswith(pattern)]

@cachetools.func.ttl_cache(maxsize=cache_maxsize, ttl=cache_ttl)
def listdir_startsandendswith(path, start, end):
    files = listdir(path)
    return [f for f in files if f.endswith(end) and f.startswith(start)]

@cachetools.func.ttl_cache(maxsize=cache_maxsize, ttl=cache_ttl)
def listdir_fnmatch(path, pattern):
    files = listdir(path)
    return [f for f in files if fnmatch.fnmatch(f, pattern)]

def clear_cache():
    listdir.cache_clear()
    listdir_startswith.cache_clear()
    listdir_endswith.cache_clear()
    listdir_startsandendswith.cache_clear()
    listdir_fnmatch.cache_clear()
