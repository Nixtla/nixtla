import inspect
import re
from pathlib import Path

from nixtla import NixtlaClient


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "timegpt-docs"
SIMULATION_GUIDE = DOCS / "forecasting/probabilistic/simulation.mdx"
COUPLED_SIMULATION_GUIDE = DOCS / "use_cases/coupled_simulation_retail.mdx"
HISTORICAL_GUIDE = DOCS / "forecasting/exogenous-variables/causal-explanations.mdx"
EXPLANATION_OVERVIEW = DOCS / "forecasting/explanation/introduction.mdx"
SHAP_GUIDE = DOCS / "forecasting/exogenous-variables/interpretability_with_shap.mdx"
INTERVENTION_GUIDE = DOCS / "forecasting/explanation/intervention.mdx"
ADVANCED_GUIDE = DOCS / "forecasting/explanation/advanced-explanations.mdx"
SDK_REFERENCE = DOCS / "reference/sdk_reference.mdx"
NAVIGATION = DOCS / "docs.json"
EXPECTED_SIMULATION_FIGURES = [
    "EUR 214.28",
    "EUR 509.86",
    "43.0%",
    "67.0%",
]
EXPECTED_COUPLED_PLANTED = [
    "+11.9%",
    "−11.5%",
    "+15.0%",
]
EXPECTED_COUPLED_RECOVERED = [
    "+6.0%",
    "−11.7%",
    "+5.4%",
]
EXPECTED_COUPLED_RISK = [
    "42.4%",
    "53.8%",
]
EXPECTED_EXPLANATION_FIGURES = [
    (SHAP_GUIDE, "92.01"),
    (HISTORICAL_GUIDE, "0.863"),
]
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


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sdk_section(name: str, next_name: str) -> str:
    reference = _read(SDK_REFERENCE)
    return reference.split(f"## NixtlaClient.{name}", 1)[1].split(
        f"## NixtlaClient.{next_name}", 1
    )[0]


def test_guides_are_in_navigation():
    navigation = _read(NAVIGATION)

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
    assert "NixtlaClient.simulate()" in _read(SIMULATION_GUIDE)
    assert "NixtlaClient.explain()" in _read(HISTORICAL_GUIDE)
    assert 'feature_contributions_type="intervention"' in (_read(INTERVENTION_GUIDE))
    assert 'feature_contributions_type="granger"' in _read(ADVANCED_GUIDE)


def test_explanation_overview_starts_with_plain_language_questions():
    overview = _read(EXPLANATION_OVERVIEW)

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
    simulation = _read(SIMULATION_GUIDE)
    explanation = "\n".join(
        _read(page)
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
    for figure in EXPECTED_SIMULATION_FIGURES:
        assert figure in simulation, f"{figure} missing from the simulation guide"

    coupled_simulation = _read(COUPLED_SIMULATION_GUIDE)
    assert "Planted role" in coupled_simulation
    assert "**Substitute**" in coupled_simulation
    assert "**Complement**" in coupled_simulation
    for figure in EXPECTED_COUPLED_PLANTED + EXPECTED_COUPLED_RECOVERED:
        assert figure in coupled_simulation, f"{figure} missing from the tutorial"
    for figure in EXPECTED_COUPLED_RISK:
        assert figure in coupled_simulation, f"{figure} missing from the tutorial"

    for page, figure in EXPECTED_EXPLANATION_FIGURES:
        assert figure in _read(page), f"{figure} missing from {page.name}"
    assert "Results were generated with TimeGPT 2.1" in explanation

    for asset in CASE_STUDY_ASSETS:
        assert asset.exists()
        assert asset.stat().st_size > 10_000
        assert asset.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert f"/images/forecasting/{asset.name}" in (
            simulation + coupled_simulation + explanation
        )


def test_simulate_does_not_expose_a_method_option():
    assert "method" not in inspect.signature(NixtlaClient.simulate).parameters

    for guide in (SIMULATION_GUIDE, COUPLED_SIMULATION_GUIDE):
        text = _read(guide)
        for call in re.findall(r"\.simulate\((.*?)\)", text, flags=re.DOTALL):
            assert "method" not in call, f"{guide.name} passes `method` to simulate"

    sdk_signature = _sdk_section("simulate", "explain")
    assert "method=" not in sdk_signature


def test_sdk_reference_contains_both_public_signatures():
    reference = _read(SDK_REFERENCE)

    assert "## NixtlaClient.simulate" in reference
    assert "## NixtlaClient.explain" in reference
    assert 'model="timegpt-2.1"' in _sdk_section("simulate", "explain")
    assert 'method="granger"' in _sdk_section("explain", "cross_validation")
    assert 'feature_contributions_type="shapley"' in _sdk_section(
        "forecast", "simulate"
    )
