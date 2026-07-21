"""Static compatibility checks for the documentation notebooks."""

import ast
import json
from pathlib import Path
from typing import Any

import pytest

import synforecast.generators as generators

NOTEBOOKS = sorted((Path(__file__).parents[1] / "nbs" / "docs").rglob("*.ipynb"))
INTEGRATION_REQUIREMENTS = {
    "neuralforecast.ipynb": [
        "SynAugment",
        "generate_series",
        "predict(df=train_df)",
        "use_init_models=False",
    ],
    "mlforecast.ipynb": [
        "SynAugment",
        "generate_series",
        "new_df=train_df",
    ],
    "statsforecast.ipynb": [
        "SynAugment",
        "synthetic_history_df",
        'groupby("ds"',
        "np.allclose",
    ],
}
GENERATOR_CLASSES = {
    name: value
    for name, value in vars(generators).items()
    if name.endswith("Generator")
    and isinstance(value, type)
    and hasattr(value, "model_fields")
}


def _dict_items(node: ast.Dict) -> dict[str, ast.expr] | None:
    """Return items from a literal configuration dictionary with string keys."""
    if all(
        isinstance(key, ast.Constant) and isinstance(key.value, str)
        for key in node.keys
    ):
        return {
            key.value: value
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    return None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_generator_configs_match_current_api(path: Path) -> None:
    """Keep executable examples aligned with the Pydantic generator models."""
    notebook = json.loads(path.read_text())
    named_configs: dict[str, dict[str, ast.expr]] = {}

    for cell_number, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue

        tree = ast.parse("".join(cell.get("source", [])))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Dict)
            ):
                items = _dict_items(node.value)
                if items is not None:
                    named_configs[node.targets[0].id] = items

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "balanced_pool"
            ):
                keyword_values = {
                    keyword.arg: keyword.value for keyword in node.keywords
                }
                engine = keyword_values.get("engine")
                assert isinstance(engine, ast.Constant) and engine.value == "polars", (
                    f'{path.name}, cell {cell_number}: set engine="polars" '
                    "because the example uses Polars APIs"
                )

            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id.endswith("Generator")
            ):
                continue

            location = f"{path.name}, cell {cell_number}"
            generator_class: Any = GENERATOR_CLASSES.get(node.func.id)
            assert generator_class is not None, (
                f"{location}: unknown generator {node.func.id}"
            )
            assert not node.args, (
                f"{location}: generator configuration must use keyword arguments"
            )

            config: dict[str, ast.expr] = {}
            for keyword in node.keywords:
                if keyword.arg is not None:
                    config[keyword.arg] = keyword.value
                elif isinstance(keyword.value, ast.Dict):
                    items = _dict_items(keyword.value)
                    assert items is not None, (
                        f"{location}: generator config keys must be strings"
                    )
                    config.update(items)
                elif isinstance(keyword.value, ast.Name):
                    assert keyword.value.id in named_configs, (
                        f"{location}: cannot resolve **{keyword.value.id}"
                    )
                    config.update(named_configs[keyword.value.id])
                else:
                    pytest.fail(f"{location}: generator config must be a literal dict")

            engine = config.get("engine")
            assert isinstance(engine, ast.Constant) and engine.value == "polars", (
                f'{location}: set engine="polars" because the example uses Polars APIs'
            )
            unknown = set(config) - set(generator_class.model_fields)
            assert not unknown, (
                f"{location}: {node.func.id} has unknown fields {sorted(unknown)}"
            )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_has_saved_outputs(path: Path) -> None:
    """Public notebooks retain outputs so docs builds include results and plots."""
    notebook = json.loads(path.read_text())
    outputs = [
        output
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        for output in cell.get("outputs", [])
    ]
    assert outputs, f"{path.relative_to(path.parents[3])} has no saved outputs"


@pytest.mark.parametrize("filename,required", INTEGRATION_REQUIREMENTS.items())
def test_integration_notebooks_cover_training_regimes(
    filename: str, required: list[str]
) -> None:
    """Keep the public integration comparisons complete and leakage-aware."""
    path = Path(__file__).parents[1] / "nbs" / "docs" / "integrations" / filename
    notebook = json.loads(path.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert source.index("train_df =") < source.index("augmented_train_df = SynAugment")
    for token in required:
        assert token in source, f"{filename} is missing {token!r}"
