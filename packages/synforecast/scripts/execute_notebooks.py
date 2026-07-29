"""Execute public documentation notebooks in a reproducible environment."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Kernel resolution must not depend on user-level kernelspecs: a stale
# ~/.local/share/jupyter/kernels/python3 whose argv is a bare "python"
# resolves through PATH and can silently launch a different project's
# interpreter (or one without ipykernel). Hiding the user/system data dirs
# makes jupyter_client fall back to the native kernel of the interpreter
# running this script, addressed by absolute path.
os.environ["JUPYTER_DATA_DIR"] = tempfile.mkdtemp(prefix="synforecast-jupyter-")
os.environ["JUPYTER_PATH"] = ""

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "nbs" / "docs"


def _notebooks() -> list[Path]:
    return sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))


def _is_network_notebook(notebook: nbformat.NotebookNode) -> bool:
    metadata = notebook.metadata.get("synforecast", {})
    return bool(metadata.get("requires_network", False))


def _normalize(notebook: nbformat.NotebookNode) -> None:
    """Remove volatile execution metadata while retaining rendered outputs."""
    notebook.metadata.pop("widgets", None)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.execution_count = None
        cell.metadata.pop("execution", None)
        outputs = []
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            if output.get("output_type") == "stream" and any(
                fragment in text
                for fragment in (
                    "datasetsforecast.utils:Successfully decompressed",
                    "Seed set to ",
                    "GPU available:",
                    "IProgress not found",
                    "Missing packages: ['ipywidgets']",
                    "pytorch_lightning/utilities/_pytree.py",
                    "TPU available:",
                    "`Trainer.fit` stopped: `max_steps=",
                )
            ):
                continue
            if output.get("output_type") == "execute_result":
                output["execution_count"] = None
            outputs.append(output)
        cell.outputs = outputs


def execute(path: Path, *, write: bool) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        shutdown_kernel="immediate",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.km = client.create_kernel_manager()
    # Kernelspecs shipped with ipykernel address the interpreter as a bare
    # "python", which resolves through PATH — an activated venv from another
    # project would silently supply the kernel. Pin it to this interpreter.
    client.km.kernel_spec.argv[0] = sys.executable
    if os.name != "nt":
        # Avoid TCP port reuse while many short-lived kernels run in sequence.
        client.km.transport = "ipc"
    client.execute()
    _normalize(notebook)
    if write:
        nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-network",
        action="store_true",
        help="also execute notebooks that download public datasets",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write normalized outputs back to the source notebooks",
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="normalize existing outputs without starting notebook kernels",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    notebooks = _notebooks()
    if not notebooks:
        raise SystemExit(f"No notebooks found below {NOTEBOOK_ROOT}")

    executed = 0
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        if args.normalize_only:
            _normalize(notebook)
            nbformat.write(notebook, path)
            continue
        if _is_network_notebook(notebook) and not args.include_network:
            print(f"SKIP {path.relative_to(ROOT)} (requires network)", flush=True)
            continue
        print(f"RUN  {path.relative_to(ROOT)}", flush=True)
        execute(path, write=args.write)
        executed += 1

    if args.normalize_only:
        print(f"Normalized {len(notebooks)} notebook(s)", flush=True)
    else:
        print(f"Executed {executed} notebook(s)", flush=True)


if __name__ == "__main__":
    main()
