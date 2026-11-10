import pytest


def assert_raises_with_message(func, expected_msg, *args, **kwargs):
    with pytest.raises((AssertionError, ValueError, Exception)) as exc_info:
        func(*args, **kwargs)
    assert expected_msg in str(exc_info.value)
