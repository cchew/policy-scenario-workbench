import json
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
