import hashlib
from pathlib import Path

import pandas as pd

from src.preprocess import load_and_clean, check_class_balance
from src.rac import load_eligibility_rule, apply_eligibility
from src.dcm import fit_dcm, dcm_register_rows
from src.abm import DiffusionModel, BASELINE_PARAMS, uncertainty_distribution, one_at_a_time_sensitivity, abm_register_rows
from src.register import Register
from src.outputs import (
    write_scenario_manifest, cohort_breakdown, write_tornado_chart,
    write_deferral_statement, bundle_causal_dag,
)


def _dataset_content_hash(csv_path: str) -> str:
    return hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()[:12]


def run_pipeline(raw_csv_path: str, dag_path: str, output_dir: str, seed: int = 42) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = load_and_clean(raw_csv_path)
    check_class_balance(df)

    rule_doc = load_eligibility_rule(str(Path(__file__).parent.parent / "rules" / "eligibility.json"))
    df = apply_eligibility(df, rule_doc)

    dcm_result = fit_dcm(df)

    register = Register()
    for row in dcm_register_rows(dcm_result):
        register.add(row)
    abm_params = {**BASELINE_PARAMS}
    for row in abm_register_rows(
        k=abm_params["k"], rewiring_p=abm_params["rewiring_p"],
        fc_uplift=abm_params["fc_uplift"],
        peer_influence_weight=abm_params["peer_influence_weight"],
        timesteps=abm_params["timesteps"],
        n_agents=abm_params["n_agents"],
    ):
        register.add(row)
    register.write_csv(str(Path(output_dir) / "register.csv"))

    model = DiffusionModel(df, dcm_result, seed=seed, **abm_params)
    run_df = model.run()
    # model.sample is the exact bootstrap draw the ABM predicted against (same
    # random_state=seed) — read it back instead of re-deriving a second draw that
    # would otherwise have to be kept in lockstep with DiffusionModel's own sampling.
    # run_df already carries its own per-agent `eligible` (post fc_uplift); drop the
    # respondents-side copy before joining so the two frames don't collide on that column.
    breakdown = cohort_breakdown(run_df, model.sample.drop(columns=["eligible"]))
    breakdown.to_csv(Path(output_dir) / "cohort_breakdown.csv", index=False)

    dist = uncertainty_distribution(df, dcm_result, n_seeds=30)
    pd.DataFrame({"adoption_rate": dist}).to_csv(
        Path(output_dir) / "uncertainty_distribution.csv", index=False
    )

    tornado = one_at_a_time_sensitivity(df, dcm_result)
    write_tornado_chart(tornado, output_dir)

    write_scenario_manifest(
        params=abm_params, seed=seed,
        dataset_version=f"pyyjfthc84 (Mendeley, accessed 2026-09); sha256:{_dataset_content_hash(raw_csv_path)}",
        output_dir=output_dir,
    )
    write_deferral_statement(output_dir)
    bundle_causal_dag(dag_path, output_dir)
