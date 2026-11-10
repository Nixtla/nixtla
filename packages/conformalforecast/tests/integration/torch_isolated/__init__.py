"""Integration tests that load torch, isolated into their own pytest session.

torch and scikit-learn each vendor a copy of ``libomp``. Loading both into one
process segfaults on macOS arm64 -- and ``KMP_DUPLICATE_LIB_OK`` does not help.
Because pytest imports every collected module before running anything, a marker
cannot prevent the clash: the split has to happen at collection scope, which is
why these files live in their own directory.

Run them with ``make test_integration``, which invokes two pytest sessions:

    pytest tests/integration --ignore=tests/integration/torch_isolated
    pytest tests/integration/torch_isolated
"""
