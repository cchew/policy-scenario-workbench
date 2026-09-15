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


def test_same_seed_produces_identical_seed_dependent_outputs(tmp_path):
    """register.csv isn't actually seed-dependent (the DCM fit is deterministic,
    ABM params are fixed constants) — cohort_breakdown.csv and
    uncertainty_distribution.csv are the outputs that genuinely depend on the
    seed (they're derived from the stochastic diffusion run / bootstrap sweep).
    """
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    run_pipeline("tests/fixtures/sample_raw.csv", "../docs/causal-dag.md", str(out_a), seed=13)
    run_pipeline("tests/fixtures/sample_raw.csv", "../docs/causal-dag.md", str(out_b), seed=13)
    assert (out_a / "cohort_breakdown.csv").read_text() == (out_b / "cohort_breakdown.csv").read_text()
    assert (out_a / "uncertainty_distribution.csv").read_text() == (out_b / "uncertainty_distribution.csv").read_text()


def test_different_seeds_produce_different_cohort_breakdown(tmp_path):
    """Confirms the pipeline's seed is actually load-bearing for the diffusion
    run, not silently ignored.

    NOTE: this asserts on cohort_breakdown.csv, not uncertainty_distribution.csv.
    Empirically verified (two real runs, seed=1 vs seed=2) that
    uncertainty_distribution.csv comes out byte-identical regardless of the
    pipeline's seed argument: `uncertainty_distribution()` hardcodes its own
    internal sweep (`range(n_seeds)`, i.e. seeds 0..29) rather than threading
    the caller's seed through — this is the parked Important #3 finding from
    the final review (see FUTURE.md), not something this fix wave changes.
    cohort_breakdown.csv is the output that's genuinely seed-dependent, since
    DiffusionModel(..., seed=seed) is the one place the caller's seed actually
    flows into the ABM run.
    """
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    run_pipeline("tests/fixtures/sample_raw.csv", "../docs/causal-dag.md", str(out_a), seed=1)
    run_pipeline("tests/fixtures/sample_raw.csv", "../docs/causal-dag.md", str(out_b), seed=2)
    assert (out_a / "cohort_breakdown.csv").read_text() != (out_b / "cohort_breakdown.csv").read_text()
