# tests/test_run_end_to_end.py
from pathlib import Path
from src.run import run_pipeline


def test_run_pipeline_produces_all_seven_outputs(tmp_path):
    output_dir = tmp_path / "run"
    run_pipeline(
        raw_csv_path="tests/fixtures/sample_raw.csv",
        dag_path="../docs/causal-dag.md",
        output_dir=str(output_dir),
        seed=42,
    )
    expected_files = {
        "register.csv", "cohort_breakdown.csv", "sensitivity_tornado.png",
        "scenario_manifest.json", "deferral_statement.md", "causal-dag.md",
        "uncertainty_distribution.csv",
    }
    actual_files = {p.name for p in output_dir.iterdir()}
    assert expected_files.issubset(actual_files)


def test_deferral_statement_contains_both_known_limitations(tmp_path):
    output_dir = tmp_path / "run"
    run_pipeline(
        raw_csv_path="tests/fixtures/sample_raw.csv",
        dag_path="../docs/causal-dag.md",
        output_dir=str(output_dir),
        seed=42,
    )
    text = (output_dir / "deferral_statement.md").read_text()
    assert "bootstrap" in text.lower()
    assert "threshold" in text.lower()


def test_same_seed_produces_identical_register(tmp_path):
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    run_pipeline("tests/fixtures/sample_raw.csv", "../docs/causal-dag.md", str(out_a), seed=7)
    run_pipeline("tests/fixtures/sample_raw.csv", "../docs/causal-dag.md", str(out_b), seed=7)
    assert (out_a / "register.csv").read_text() == (out_b / "register.csv").read_text()
