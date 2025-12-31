import sys
import pytest

# Skip entire directory if Ray is not available or on Windows
pytest.importorskip('ray', reason="Ray is required for distributed tests")
if sys.platform == "win32":
    pytest.skip("Distributed tests are not supported on Windows", allow_module_level=True)

import ray


@pytest.fixture(scope="session", autouse=True)
def ray_session():
    """Initialize Ray once for all tests in this directory."""
    # Initialize Ray with limited resources for CI
    ray.init(
        num_cpus=2,
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir="/tmp/ray"
    )
    yield
    # Shutdown Ray after all tests complete
    ray.shutdown()


@pytest.fixture(autouse=True)
def ray_test_cleanup():
    """Clean up Ray resources after each test."""
    yield
    # Ensure any datasets are cleaned up between tests
    import gc
    gc.collect()
