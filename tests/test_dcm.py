import numpy as np
import pandas as pd
from src.dcm import fit_dcm, dcm_register_rows, predict_probability, FEATURES


def _synthetic_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "pe_score": rng.uniform(1, 5, n),
        "ee_score": rng.uniform(1, 5, n),
        "si_score": rng.uniform(1, 5, n),
        "fc_score": rng.uniform(1, 5, n),
    })
    # adopter driven mostly by fc_score, so the fitted fc coefficient should be positive
    logit = -6 + 1.5 * df["fc_score"]
    prob = 1 / (1 + np.exp(-logit))
    df["adopter"] = (rng.uniform(0, 1, n) < prob).astype(int)
    return df


def test_fit_dcm_recovers_positive_fc_coefficient():
    df = _synthetic_df()
    result = fit_dcm(df)
    assert result.params["fc_score"] > 0


def test_dcm_register_rows_all_tagged_observed():
    df = _synthetic_df()
    result = fit_dcm(df)
    rows = dcm_register_rows(result)
    assert len(rows) == len(FEATURES) + 1  # + intercept
    assert all(r.evidence_strength == "observed" for r in rows)
    param_ids = {r.parameter for r in rows}
    assert param_ids == {"dcm.intercept", "dcm.coef.pe", "dcm.coef.ee", "dcm.coef.si", "dcm.coef.fc"}


def test_predict_probability_is_between_0_and_1():
    df = _synthetic_df()
    result = fit_dcm(df)
    probs = predict_probability(result, df)
    assert probs.between(0, 1).all()


def test_fit_dcm_is_deterministic():
    df = _synthetic_df()
    result_a = fit_dcm(df)
    result_b = fit_dcm(df)
    pd.testing.assert_series_equal(result_a.params, result_b.params)
