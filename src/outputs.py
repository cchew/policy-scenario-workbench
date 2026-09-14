import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_scenario_manifest(params: dict, seed: int, dataset_version: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest = {
        "params": params,
        "seed": seed,
        "dataset_version": dataset_version,
        "git_sha": _git_sha(),
    }
    (Path(output_dir) / "scenario_manifest.json").write_text(json.dumps(manifest, indent=2))


def cohort_breakdown(run_df: pd.DataFrame, respondents_sample: pd.DataFrame) -> pd.DataFrame:
    joined = run_df.reset_index(drop=True).join(respondents_sample.reset_index(drop=True))
    return (
        joined.groupby("role")["adopted"]
        .mean()
        .reset_index()
        .rename(columns={"adopted": "adoption_rate"})
    )


def write_tornado_chart(tornado: dict, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    params = list(tornado.keys())
    los = [tornado[p][0] for p in params]
    his = [tornado[p][1] for p in params]
    widths = [h - l for l, h in zip(los, his)]

    fig, ax = plt.subplots()
    ax.barh(params, widths, left=los)
    ax.set_xlabel("Final adoption rate")
    ax.set_title("One-at-a-time sensitivity")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "sensitivity_tornado.png")
    plt.close(fig)


DEFERRAL_STATEMENT = """\
# Deferral statement

This is a demo-grade architecture POC, not a validated forecasting tool.

- No individual decisions are made or informed by this output.
- No forecast claim is made about real-world adoption rates.
- This is not a substitute for consultation with affected stakeholders.
- **Bootstrap-resampling limitation:** synthetic agents are resampled from one
  small, university-only survey — they are near-duplicates of the same
  respondent pool, not independent draws from a larger population.
  Sensitivity/tornado outputs likely understate true real-world variance.
- **Dichotomization threshold limitation:** the top-2-box split used to derive
  a binary adopter label from the Likert intention score is a modelling
  choice, not a given — a different threshold (top-1-box, median split)
  would calibrate a different DCM. This was not validated against
  alternatives.
"""


def write_deferral_statement(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "deferral_statement.md").write_text(DEFERRAL_STATEMENT)


def bundle_causal_dag(dag_path: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(dag_path, Path(output_dir) / "causal-dag.md")
