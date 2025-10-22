import time
import inspect
from contextlib import contextmanager
from functools import wraps
from loguru import logger


@contextmanager
def timed():
    """Context manager to time a code block"""
    frame = inspect.currentframe().f_back.f_back
    filename = frame.f_code.co_filename.split('/')[-1]
    funcname = frame.f_code.co_name
    lineno = frame.f_lineno

    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.debug(f"{filename}:{funcname}:{lineno} took {elapsed:.4f}s")


def timed_func(func):
    """Decorator to time a function"""
    # Get info from the decorated function
    filename = func.__code__.co_filename.split('/')[-1]
    funcname = func.__name__
    lineno = func.__code__.co_firstlineno

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{filename}:{funcname}:{lineno} took {elapsed:.4f}s")
        return result

    return wrapper