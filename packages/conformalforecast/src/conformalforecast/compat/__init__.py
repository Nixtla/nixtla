"""Migration shims for libraries adopting the calibration layer.

Each module here soft-imports its target library and exists to show that
library's maintainers the shortest path from their current in-repo conformal
code to a :class:`~conformalforecast.methods.base.ConformalMethod`. Nothing in
``compat/`` is required by the rest of the package.
"""

from __future__ import annotations

__all__: list[str] = []
