"""Client factory shared by the docs asset-generation scripts.

Two ways to reach a model:

* In-process tsfm (default). ``tsfm.serverless.NixtlaClient`` subclasses the real
  client and routes requests through a FastAPI ``TestClient``, so no service and
  no API key are needed. Run from a tsfm checkout with this repository shadowing
  the ``nixtla`` release that tsfm pins::

      PYTHONPATH=/path/to/nixtla uv run --no-sync python \
        /path/to/nixtla/timegpt-docs/scripts/<script>.py

* Over HTTP, by setting ``NIXTLA_BASE_URL`` (plus ``NIXTLA_API_KEY``) to a
  running nixtla-compute service.
"""

import os
from typing import Any


def make_docs_client(**kwargs: Any):
    """Return the client the docs assets should be generated with."""
    base_url = os.environ.get("NIXTLA_BASE_URL")
    if base_url:
        from nixtla import NixtlaClient

        return NixtlaClient(
            base_url=base_url,
            api_key=os.environ.get("NIXTLA_API_KEY", "local"),
            **kwargs,
        )

    try:
        from tsfm.serverless import NixtlaClient as ServerlessNixtlaClient
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "No model backend available. Either install tsfm and run this script "
            "from a tsfm checkout (see this module's docstring), or point "
            "NIXTLA_BASE_URL at a running nixtla-compute service."
        ) from exc

    return ServerlessNixtlaClient(**kwargs)
