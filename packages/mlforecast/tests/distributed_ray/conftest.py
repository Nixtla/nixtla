import os
import sys
import pytest

# Skip entire directory if Ray is not available or on Windows
pytest.importorskip("ray", reason="Ray is required for distributed tests")
# CI only runs these on linux, but they work locally on macOS; set
# MLFORECAST_FORCE_RAY_TESTS=1 to run them there while developing.
if sys.platform != "linux" and os.environ.get("MLFORECAST_FORCE_RAY_TESTS") != "1":
    pytest.skip(
        "Distributed interface is only supported on Linux", allow_module_level=True
    )

import ray


@pytest.fixture(scope="session", autouse=True)
def ray_session():
    """Initialize Ray once for all tests in this directory."""
    # Initialize Ray with limited resources for CI
    ray.init(
        num_cpus=2,
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir="/tmp/ray",
    )
    yield
    # Shutdown Ray after all tests complete
    ray.shutdown()


@pytest.fixture(autouse=True)
def ray_test_cleanup():
    """Clean up Ray resources after each test."""
    from ray.util.placement_group import placement_group_table

    # anything already there belongs to a wider scoped fixture, not to this test
    before = set(placement_group_table())
    yield
    # Ensure any datasets are cleaned up between tests
    import gc

    gc.collect()
    _reclaim_placement_groups(keep=before)


def _reclaim_placement_groups(keep=()):
    """Remove placement groups left behind by a failed training run.

    A training failure can leak its placement group, which keeps holding the
    cluster's CPUs. The next test's ``ray.data`` call would then block forever
    rather than fail, which is how a one line dtype error turned into a 6 hour
    CI job (see #713).

    ``keep`` holds the ids that were already around when the test started, so a
    group a session or module scoped fixture legitimately holds isn't taken with
    them.
    """
    from ray._raylet import PlacementGroupID
    from ray.util.placement_group import (
        PlacementGroup,
        placement_group_table,
        remove_placement_group,
    )

    for pg_id, info in placement_group_table().items():
        if pg_id in keep or info["state"] == "REMOVED":
            continue
        try:
            remove_placement_group(PlacementGroup(PlacementGroupID.from_hex(pg_id)))
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass
