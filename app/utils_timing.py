# coding: utf-8

from functools import wraps
from time import time


def timing_decorator(f):
    @wraps(f)
    def wrap(*args, **kw):
        start_time = time()
        result = f(*args, **kw)
        end_time = time()
        print('timing: %r args: [%r, %r] took: %2.5f sec' % (f.__name__, args, kw, end_time - start_time))
        return result
    return wrap
