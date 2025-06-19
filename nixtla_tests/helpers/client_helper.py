import os

from contextlib import contextmanager


@contextmanager
def delete_env_var(key):
    original_value = os.environ.get(key)
    rm = False
    if key in os.environ:
        del os.environ[key]
        rm = True
    try:
        yield
    finally:
        if rm:
            os.environ[key] = original_value
