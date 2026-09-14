import json
import os
import pandas as pd
from src.outputs import write_scenario_manifest, cohort_breakdown, write_tornado_chart


def test_write_scenario_manifest_creates_valid_json(tmp_path):
    write_scenario_manifest(
        params={"n_agents": 300, "k": 6}, seed=42,
        dataset_version="pyyjfthc84-v1", output_dir=str(tmp_path),
    )
    manifest_path = tmp_path / "scenario_manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["seed"] == 42
    assert data["dataset_version"] == "pyyjfthc84-v1"
    assert "git_sha" in data
    assert data["params"]["k"] == 6


def test_cohort_breakdown_groups_by_role():
    run_df = pd.DataFrame({
        "node_id": [0, 1, 2, 3], "adopted": [True, False, True, True],
    })
    respondents_sample = pd.DataFrame({
        "role": ["staff", "student", "staff", "student"],
    })
    breakdown = cohort_breakdown(run_df, respondents_sample)
    assert set(breakdown["role"]) == {"staff", "student"}
    staff_rate = breakdown.loc[breakdown["role"] == "staff", "adoption_rate"].iloc[0]
    assert staff_rate == 1.0  # both staff rows adopted


def test_write_tornado_chart_creates_png(tmp_path):
    tornado = {"k": (0.2, 0.4), "fc_uplift": (0.1, 0.6)}
    write_tornado_chart(tornado, output_dir=str(tmp_path))
    assert (tmp_path / "sensitivity_tornado.png").exists()
