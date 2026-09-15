# tests/test_register_completeness.py
import numpy as np
import pandas as pd
from src.dcm import fit_dcm, dcm_register_rows, FEATURES, _SHORT_NAMES
from src.abm import abm_register_rows, BASELINE_PARAMS
from src.register import Register

# Explicit mapping from BASELINE_PARAMS key -> register row ID. Kept explicit (not
# auto-derived from the key string) since the ID format doesn't mechanically follow
# from the param name (e.g. "fc_uplift" -> "lever.fc_uplift", not "abm.fc_uplift").
# Deriving REQUIRED_PARAM_IDS from this + BASELINE_PARAMS.keys() below means this test
# fails loudly (KeyError or a missing-parameter assertion) if a future BASELINE_PARAMS
# key is added without both a mapping entry here and a corresponding register row.
_ABM_PARAM_TO_REGISTER_ID = {
    "k": "abm.network.k",
    "rewiring_p": "abm.network.rewiring_p",
    "fc_uplift": "lever.fc_uplift",
    "peer_influence_weight": "abm.peer_influence.weight",
    "timesteps": "abm.timesteps.T",
    "n_agents": "abm.n_agents",
}

REQUIRED_PARAM_IDS = (
    {"dcm.intercept"}
    | {f"dcm.coef.{_SHORT_NAMES[f]}" for f in FEATURES}
    | {_ABM_PARAM_TO_REGISTER_ID[k] for k in BASELINE_PARAMS.keys()}
)


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
        n_agents=BASELINE_PARAMS["n_agents"],
    ):
        reg.add(row)

    reg.assert_covers(REQUIRED_PARAM_IDS)
