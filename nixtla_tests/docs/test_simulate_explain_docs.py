import inspect
import json
from pathlib import Path

from nixtla import NixtlaClient


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "timegpt-docs"
SIMULATION_GUIDE = DOCS / "forecasting/probabilistic/simulation.mdx"
COUPLED_SIMULATION_GUIDE = (
    DOCS / "use_cases/coupled_simulation_retail.mdx"
)
HISTORICAL_GUIDE = (
    DOCS / "forecasting/exogenous-variables/causal-explanations.mdx"
)
EXPLANATION_OVERVIEW = DOCS / "forecasting/explanation/introduction.mdx"
SHAP_GUIDE = (
    DOCS / "forecasting/exogenous-variables/interpretability_with_shap.mdx"
)
INTERVENTION_GUIDE = DOCS / "forecasting/explanation/intervention.mdx"
ADVANCED_GUIDE = DOCS / "forecasting/explanation/advanced-explanations.mdx"
SDK_REFERENCE = DOCS / "reference/sdk_reference.mdx"
OPENAPI = DOCS / "openapi.json"
NAVIGATION = DOCS / "docs.json"
CASE_STUDY_ASSETS = [
    DOCS / "images/forecasting/simulation-energy-scenarios.png",
    DOCS / "images/forecasting/simulation-energy-cost-risk.png",
    DOCS / "images/forecasting/simulation-retail-history.png",
    DOCS / "images/forecasting/simulation-retail-paths.png",
    DOCS / "images/forecasting/simulation-retail-coupled-risk.png",
    DOCS / "images/forecasting/explain-retail-shap.png",
    DOCS / "images/forecasting/explain-retail-intervention.png",
    DOCS / "images/forecasting/explain-retail-historical-signals.png",
    DOCS / "images/forecasting/explain-retail-analysis-comparison.png",
    DOCS / "images/forecasting/explain-retail-advanced-allocations.png",
    DOCS / "images/forecasting/explain-retail-window-stability.png",
]


def _sdk_section(name: str, next_name: str) -> str:
    reference = SDK_REFERENCE.read_text()
    return reference.split(f"## NixtlaClient.{name}", 1)[1].split(
        f"## NixtlaClient.{next_name}", 1
    )[0]


def test_guides_are_in_navigation():
    navigation = NAVIGATION.read_text()

    assert "/forecasting/probabilistic/simulation" in navigation
    assert "/use_cases/what_if_forecasting_price_effects_in_retail" in navigation
    assert "/use_cases/coupled_simulation_retail" in navigation
    assert navigation.index(
        "/use_cases/what_if_forecasting_price_effects_in_retail"
    ) < navigation.index("/use_cases/coupled_simulation_retail")
    assert '"group": "Explanation"' in navigation
    assert "/forecasting/explanation/introduction" in navigation
    assert "/forecasting/explanation/intervention" in navigation
    assert "/forecasting/explanation/advanced-explanations" in navigation
    assert "/forecasting/exogenous-variables/causal-explanations" in navigation
    assert "/forecasting/exogenous-variables/interpretability_with_shap" in navigation


def test_guides_link_to_the_sdk_capabilities():
    assert "NixtlaClient.simulate()" in SIMULATION_GUIDE.read_text()
    assert "NixtlaClient.explain()" in HISTORICAL_GUIDE.read_text()
    assert 'feature_contributions_type="intervention"' in (
        INTERVENTION_GUIDE.read_text()
    )
    assert 'feature_contributions_type="granger"' in ADVANCED_GUIDE.read_text()


def test_explanation_overview_starts_with_plain_language_questions():
    overview = EXPLANATION_OVERVIEW.read_text()

    for question in [
        "Why is this forecast high or low?",
        "What is the forecast sensitive to?",
        "Which signals have been useful historically?",
    ]:
        assert question in overview

    assert "If you have a forecast in front of you and are unsure" in overview
    assert "Granger" not in overview
    assert "transfer entropy" not in overview.lower()


def test_case_studies_include_rendered_assets():
    simulation = SIMULATION_GUIDE.read_text()
    explanation = "\n".join(
        page.read_text()
        for page in [
            EXPLANATION_OVERVIEW,
            SHAP_GUIDE,
            INTERVENTION_GUIDE,
            HISTORICAL_GUIDE,
            ADVANCED_GUIDE,
        ]
    )

    assert "Real-data walkthrough: German electricity prices" in simulation
    assert "electricity-short-with-ex-vars.csv" in simulation
    assert "EUR 214.28" in simulation
    assert "EUR 509.86" in simulation
    assert "43.0%" in simulation
    assert "67.0%" in simulation
    coupled_simulation = COUPLED_SIMULATION_GUIDE.read_text()
    # The product roles are planted in the generator, so the guide must state
    # both the planted effect and the recovered one.
    assert "Planted role" in coupled_simulation
    assert "**Substitute**" in coupled_simulation
    assert "**Complement**" in coupled_simulation
    assert "+11.9%" in coupled_simulation
    assert "−11.5%" in coupled_simulation
    assert "+15.0%" in coupled_simulation
    assert "−11.7%" in coupled_simulation
    assert "+5.4%" in coupled_simulation
    assert "+6.0%" in coupled_simulation
    assert "42.4%" in coupled_simulation
    assert "53.8%" in coupled_simulation
    assert "92.01" in SHAP_GUIDE.read_text()
    assert "0.863" in HISTORICAL_GUIDE.read_text()
    assert "Results were generated with TimeGPT 2.1" in explanation

    for asset in CASE_STUDY_ASSETS:
        assert asset.exists()
        assert asset.stat().st_size > 10_000
        assert asset.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert f"/images/forecasting/{asset.name}" in (
            simulation + coupled_simulation + explanation
        )


def test_simulation_documentation_does_not_expose_hidden_option():
    hidden_option = "method"
    guide = (
        SIMULATION_GUIDE.read_text()
        + COUPLED_SIMULATION_GUIDE.read_text()
    ).lower()
    sdk_section = _sdk_section("simulate", "explain").lower()

    spec = json.loads(OPENAPI.read_text())
    raw_api_contract = json.dumps(
        {
            "path": spec["paths"]["/v2/simulate"],
            "schema": spec["components"]["schemas"]["SimulateInput"],
        }
    ).lower()

    assert hidden_option not in guide
    assert hidden_option not in sdk_section
    assert hidden_option not in raw_api_contract
    assert hidden_option not in inspect.signature(NixtlaClient.simulate).parameters


def test_openapi_exposes_simulate_and_explain_contracts():
    spec = json.loads(OPENAPI.read_text())
    schemas = spec["components"]["schemas"]

    assert spec["paths"]["/v2/simulate"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/SimulateInput")
    assert spec["paths"]["/v2/explain"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/ExplainInput")
    assert schemas["SimulateInput"]["properties"].keys() == {
        "series",
        "freq",
        "h",
        "model",
        "finetuned_model_id",
        "clean_ex_first",
        "multivariate",
        "n_paths",
        "quantiles",
        "seed",
    }
    assert schemas["ExplainInput"]["properties"]["method"]["enum"] == [
        "granger",
        "transfer_entropy",
    ]
    assert schemas["ForecastInput"]["properties"]["feature_contributions_type"][
        "enum"
    ] == ["shapley", "intervention", "granger", "transfer_entropy"]


def test_sdk_reference_contains_both_public_signatures():
    reference = SDK_REFERENCE.read_text()

    assert "## NixtlaClient.simulate" in reference
    assert "## NixtlaClient.explain" in reference
    assert 'model="timegpt-2.1"' in _sdk_section("simulate", "explain")
    assert 'method="granger"' in _sdk_section("explain", "cross_validation")
    assert 'feature_contributions_type="shapley"' in _sdk_section(
        "forecast", "simulate"
    )
