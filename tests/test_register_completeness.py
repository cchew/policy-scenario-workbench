# tests/test_register_completeness.py
import numpy as np
import pandas as pd
from src.dcm import fit_dcm, dcm_register_rows
from src.abm import abm_register_rows, BASELINE_PARAMS
from src.register import Register

REQUIRED_PARAM_IDS = {
    "dcm.intercept", "dcm.coef.pe", "dcm.coef.ee", "dcm.coef.si", "dcm.coef.fc",
    "lever.fc_uplift", "abm.network.k", "abm.network.rewiring_p",
    "abm.peer_influence.weight", "abm.timesteps.T",
}


def test_register_covers_every_dag_edge_reference():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "pe_score": rng.uniform(1, 5, 100), "ee_score": rng.uniform(1, 5, 100),
        "si_score": rng.uniform(1, 5, 100), "fc_score": rng.uniform(1, 5, 100),
    })
    df["adopter"] = (rng.uniform(0, 1, 100) < 0.4).astype(int)
    result = fit_dcm(df)

    reg = Register()
    for row in dcm_register_rows(result):
        reg.add(row)
    for row in abm_register_rows(
        k=BASELINE_PARAMS["k"], rewiring_p=BASELINE_PARAMS["rewiring_p"],
        fc_uplift=BASELINE_PARAMS["fc_uplift"],
        peer_influence_weight=BASELINE_PARAMS["peer_influence_weight"],
        timesteps=BASELINE_PARAMS["timesteps"],
    ):
        reg.add(row)

    reg.assert_covers(REQUIRED_PARAM_IDS)
