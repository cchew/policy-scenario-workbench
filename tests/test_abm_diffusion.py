import numpy as np
import pandas as pd
import pytest
from src.abm import DiffusionModel, final_adoption_rate, uncertainty_distribution, one_at_a_time_sensitivity
from src.dcm import fit_dcm


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


def test_neighbour_adopted_fraction_zero_for_isolated_node():
    df = _respondents_df()
    result = fit_dcm(df)
    model = DiffusionModel(df, result, n_agents=20, k=2, rewiring_p=0.0,
                            fc_uplift=1.0, peer_influence_weight=0.5, timesteps=5, seed=1)
    # no one has adopted yet, so every fraction must be 0.0 regardless of degree
    for node_id in model.node_to_agent:
        assert model.neighbour_adopted_fraction(node_id) == 0.0


def test_adoption_probability_clamps_at_one():
    from src.abm import adoption_probability
    # baseline_prob=0.9, weight=5.0, fraction=1.0 -> raw 0.9*6=5.4, must clamp to 1.0
    assert adoption_probability(baseline_prob=0.9, peer_influence_weight=5.0, fraction=1.0) == 1.0


def test_adoption_probability_no_boost_when_fraction_zero():
    from src.abm import adoption_probability
    assert adoption_probability(baseline_prob=0.3, peer_influence_weight=5.0, fraction=0.0) == 0.3


def test_step_once_never_produces_invalid_probability():
    df = _respondents_df()
    result = fit_dcm(df)
    model = DiffusionModel(df, result, n_agents=20, k=6, rewiring_p=0.3,
                            fc_uplift=3.0, peer_influence_weight=5.0, timesteps=1, seed=2)
    # force everyone adopted except one, to maximize peer_influence_weight * fraction
    for agent in model.node_to_agent.values():
        agent.adopted = True
    target = next(iter(model.node_to_agent.values()))
    target.adopted = False
    model.step_once(current_step=0)  # must not raise


def test_run_returns_expected_columns_and_length():
    df = _respondents_df()
    result = fit_dcm(df)
    model = DiffusionModel(df, result, n_agents=15, k=4, rewiring_p=0.1,
                            fc_uplift=1.0, peer_influence_weight=0.5, timesteps=10, seed=3)
    out = model.run()
    assert set(out.columns) == {"node_id", "eligible", "baseline_prob", "adopted", "adopted_step"}
    assert len(out) == 15


def test_same_seed_gives_identical_run_result():
    df = _respondents_df()
    result = fit_dcm(df)
    model_a = DiffusionModel(df, result, n_agents=15, k=4, rewiring_p=0.1,
                              fc_uplift=1.0, peer_influence_weight=0.5, timesteps=10, seed=99)
    model_b = DiffusionModel(df, result, n_agents=15, k=4, rewiring_p=0.1,
                              fc_uplift=1.0, peer_influence_weight=0.5, timesteps=10, seed=99)
    pd.testing.assert_frame_equal(model_a.run(), model_b.run())


def test_final_adoption_rate_is_a_fraction():
    df = _respondents_df()
    result = fit_dcm(df)
    rate = final_adoption_rate(df, result, seed=5)
    assert 0.0 <= rate <= 1.0


def test_uncertainty_distribution_length_matches_n_seeds():
    df = _respondents_df()
    result = fit_dcm(df)
    dist = uncertainty_distribution(df, result, n_seeds=5)
    assert len(dist) == 5


def test_one_at_a_time_sensitivity_covers_all_five_params():
    df = _respondents_df()
    result = fit_dcm(df)
    tornado = one_at_a_time_sensitivity(df, result)
    assert set(tornado.keys()) == {"k", "rewiring_p", "fc_uplift", "peer_influence_weight", "timesteps"}
    for lo, hi in tornado.values():
        assert lo <= hi


def test_step_hazard_recovers_baseline_prob_as_terminal_probability():
    """Regression test for the diffusion-saturation bug: baseline_prob must be
    converted to a per-step hazard ONCE at construction, not re-applied raw every
    timestep (which would make P(adopt by T)=1-(1-p)^T saturate near 1 for any
    baseline_prob above ~0.15 at T=20, swamping the fc_uplift lever and the peer
    network). This asserts the closed-form property that must hold immediately
    after construction, for every agent, independent of any stochastic draw:
    1 - (1 - step_hazard)**timesteps == baseline_prob.
    """
    df = _respondents_df()
    result = fit_dcm(df)
    timesteps = 20
    model = DiffusionModel(df, result, n_agents=25, k=4, rewiring_p=0.1,
                            fc_uplift=1.0, peer_influence_weight=0.0,
                            timesteps=timesteps, seed=11)
    for agent in model.node_to_agent.values():
        assert agent.step_hazard != agent.baseline_prob or agent.baseline_prob == 0.0
        recovered_terminal_prob = 1 - (1 - agent.step_hazard) ** timesteps
        assert recovered_terminal_prob == pytest.approx(agent.baseline_prob, abs=1e-9)
