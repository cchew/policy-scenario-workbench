# tests/test_abm_setup.py
import pandas as pd
from src.abm import DiffusionModel, abm_register_rows
from src.dcm import fit_dcm
import numpy as np


def _respondents_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "pe_score": rng.uniform(1, 5, n),
        "ee_score": rng.uniform(1, 5, n),
        "si_score": rng.uniform(1, 5, n),
        "fc_score": rng.uniform(1, 5, n),
    })
    logit = -6 + 1.5 * df["fc_score"]
    prob = 1 / (1 + np.exp(-logit))
    df["adopter"] = (rng.uniform(0, 1, n) < prob).astype(int)
    df["eligible"] = [i % 3 == 0 for i in range(n)]
    return df


def test_model_creates_n_agents_with_valid_baseline_probs():
    df = _respondents_df()
    result = fit_dcm(df)
    model = DiffusionModel(df, result, n_agents=30, k=4, rewiring_p=0.1,
                            fc_uplift=1.0, peer_influence_weight=0.5, timesteps=10, seed=42)
    assert len(model.node_to_agent) == 30
    for agent in model.node_to_agent.values():
        assert 0.0 <= agent.baseline_prob <= 1.0
        assert agent.adopted is False
        assert agent.adopted_step is None


def test_eligible_agents_get_fc_uplift_reflected_in_higher_baseline():
    df = _respondents_df()
    result = fit_dcm(df)
    model_no_uplift = DiffusionModel(df, result, n_agents=30, k=4, rewiring_p=0.1,
                                      fc_uplift=0.0, peer_influence_weight=0.5, timesteps=10, seed=42)
    model_with_uplift = DiffusionModel(df, result, n_agents=30, k=4, rewiring_p=0.1,
                                        fc_uplift=2.0, peer_influence_weight=0.5, timesteps=10, seed=42)
    eligible_no_uplift = [a.baseline_prob for a in model_no_uplift.node_to_agent.values() if a.eligible]
    eligible_with_uplift = [a.baseline_prob for a in model_with_uplift.node_to_agent.values() if a.eligible]
    assert sum(eligible_with_uplift) >= sum(eligible_no_uplift)


def test_same_seed_produces_identical_network():
    df = _respondents_df()
    result = fit_dcm(df)
    model_a = DiffusionModel(df, result, n_agents=20, k=4, rewiring_p=0.1,
                              fc_uplift=1.0, peer_influence_weight=0.5, timesteps=10, seed=7)
    model_b = DiffusionModel(df, result, n_agents=20, k=4, rewiring_p=0.1,
                              fc_uplift=1.0, peer_influence_weight=0.5, timesteps=10, seed=7)
    assert sorted(model_a.graph.edges()) == sorted(model_b.graph.edges())


def test_abm_register_rows_all_tagged_expert_assumption():
    rows = abm_register_rows(k=6, rewiring_p=0.1, fc_uplift=1.0, peer_influence_weight=0.5,
                              timesteps=20, n_agents=300)
    assert all(r.evidence_strength == "expert-assumption" for r in rows)
    param_ids = {r.parameter for r in rows}
    assert param_ids == {
        "abm.network.k", "abm.network.rewiring_p", "lever.fc_uplift",
        "abm.peer_influence.weight", "abm.timesteps.T", "abm.n_agents",
    }
