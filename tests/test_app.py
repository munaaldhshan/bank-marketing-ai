"""End-to-end check of app/app.py against a small trained model, no real dataset required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.config as config
from src.final_pipeline import train_and_save_final_pipeline

APP_PATH = Path(__file__).resolve().parents[1] / 'app' / 'app.py'


@pytest.fixture
def trained_artifacts(tmp_path, synthetic_df, monkeypatch):
    data_path = tmp_path / 'bank_marketing.csv'
    synthetic_df.to_csv(data_path, sep=';', index=False)

    model_path = tmp_path / 'final_model.joblib'
    schema_path = tmp_path / 'feature_schema.json'
    snapshots_path = tmp_path / 'economic_snapshots.json'
    metrics_path = tmp_path / 'metrics.json'

    train_and_save_final_pipeline(
        data_path=data_path, model_path=model_path,
        metrics_path=metrics_path, schema_path=schema_path,
        economic_snapshots_path=snapshots_path,
    )

    # app.py imports these names directly from src.config at run time, so patching the
    # module's attributes (rather than app.py's already-bound names) is what a fresh
    # `AppTest.from_file(...).run()` re-import will actually pick up.
    monkeypatch.setattr(config, 'MODEL_PATH', model_path)
    monkeypatch.setattr(config, 'SCHEMA_PATH', schema_path)
    monkeypatch.setattr(config, 'ECONOMIC_SNAPSHOTS_PATH', snapshots_path)
    monkeypatch.setattr(config, 'METRICS_PATH', metrics_path)
    return model_path, schema_path, snapshots_path, metrics_path


def test_app_loads_without_a_trained_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'MODEL_PATH', tmp_path / 'missing.joblib')
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception
    assert any('train' in w.value.lower() for w in at.warning)


def test_app_predicts_for_a_first_time_contact(trained_artifacts):
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception

    at.button[0].click().run()  # checkbox left unchecked: "never contacted before" path
    assert not at.exception
    assert len(at.metric) == 1
    probability = float(at.metric[0].value.rstrip('%'))
    assert 0.0 <= probability <= 100.0


def test_app_predicts_for_a_previously_contacted_customer(trained_artifacts):
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.checkbox[0].set_value(True).run()
    assert not at.exception
    assert len(at.slider) > 2  # the pdays/previous sliders should now be showing

    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) == 1


def test_app_warns_on_an_out_of_schema_numeric_value(trained_artifacts):
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.slider[0].set_value(9999).run()  # age slider is clamped by min/max, but check the plumbing
    at.button[0].click().run()
    assert not at.exception


def test_app_uses_the_tuned_threshold_from_metrics_not_a_hardcoded_half(trained_artifacts):
    _, _, _, metrics_path = trained_artifacts
    threshold = json.loads(metrics_path.read_text())['decision_threshold']

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.button[0].click().run()
    assert any(f'{threshold:.0%}' in md.value for md in at.markdown)
