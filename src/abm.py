import mesa
import networkx as nx
import pandas as pd

from src.dcm import FEATURES, predict_probability
from src.register import RegisterRow
from src.schema import LIKERT_MAX


class AdoptionAgent(mesa.Agent):
    def __init__(self, model, eligible: bool, baseline_prob: float, step_hazard: float):
        super().__init__(model)
        self.eligible = eligible
        self.baseline_prob = baseline_prob
        self.step_hazard = step_hazard
        self.adopted = False
        self.adopted_step = None


class DiffusionModel(mesa.Model):
    def __init__(self, respondents_df: pd.DataFrame, dcm_result, n_agents: int,
                 k: int, rewiring_p: float, fc_uplift: float,
                 peer_influence_weight: float, timesteps: int, seed: int):
        super().__init__(seed=seed)
        self.peer_influence_weight = peer_influence_weight
        self.timesteps = timesteps
        self.graph = nx.watts_strogatz_graph(n_agents, k, rewiring_p, seed=seed)
        self.node_to_agent: dict[int, AdoptionAgent] = {}

        sample = respondents_df.sample(n=n_agents, replace=True, random_state=seed).reset_index(drop=True)
        # Exposed so callers (e.g. run.py's cohort_breakdown join) can read back the
        # exact bootstrap draw this model predicted against, instead of re-deriving a
        # second draw that has to be kept in lockstep by using an identical seed.
        self.sample = sample

        eligible_mask = sample["eligible"].astype(bool)
        fc_scores = sample["fc_score"].mask(
            eligible_mask, (sample["fc_score"] + fc_uplift).clip(upper=LIKERT_MAX)
        )
        # One batch prediction for all n_agents instead of one statsmodels call per
        # agent — this loop runs ~55-60 times per pipeline call (uncertainty sweep +
        # sensitivity sweep), so per-agent calls meant tens of thousands of individual
        # predict() calls.
        X = sample[FEATURES].assign(fc_score=fc_scores)
        baseline_probs = predict_probability(dcm_result, X)

        for node_id, row in sample.iterrows():
            eligible = bool(row["eligible"])
            baseline_prob = float(baseline_probs.loc[node_id])
            # Convert the DCM's cumulative/terminal adoption probability into a genuine
            # per-step hazard, once, at construction time. `_SENSITIVITY_RANGES["timesteps"]`
            # never includes 0, so no zero-division guard is needed here.
            step_hazard = 1 - (1 - baseline_prob) ** (1 / self.timesteps)
            agent = AdoptionAgent(self, eligible=eligible, baseline_prob=baseline_prob,
                                   step_hazard=step_hazard)
            self.node_to_agent[node_id] = agent

    def neighbour_adopted_fraction(self, node_id: int) -> float:
        neighbours = list(self.graph.neighbors(node_id))
        if not neighbours:
            return 0.0
        adopted = sum(1 for n in neighbours if self.node_to_agent[n].adopted)
        return adopted / len(neighbours)

    def step_once(self, current_step: int) -> None:
        for node_id, agent in self.node_to_agent.items():
            if agent.adopted:
                continue
            fraction = self.neighbour_adopted_fraction(node_id)
            prob = adoption_probability(agent.step_hazard, self.peer_influence_weight, fraction)
            if self.random.random() < prob:
                agent.adopted = True
                agent.adopted_step = current_step

    def run(self) -> pd.DataFrame:
        for t in range(self.timesteps):
            self.step_once(t)
        return pd.DataFrame([
            {"node_id": nid, "eligible": a.eligible, "baseline_prob": a.baseline_prob,
             "adopted": a.adopted, "adopted_step": a.adopted_step}
            for nid, a in self.node_to_agent.items()
        ])


def abm_register_rows(k: int, rewiring_p: float, fc_uplift: float,
                       peer_influence_weight: float, timesteps: int,
                       n_agents: int) -> list[RegisterRow]:
    return [
        RegisterRow("abm.network.k", k, "expert assumption", "expert-assumption",
                    "Watts-Strogatz nearest-neighbour degree; no data justifies a specific value"),
        RegisterRow("abm.network.rewiring_p", rewiring_p, "expert assumption", "expert-assumption",
                    "Watts-Strogatz rewiring probability"),
        RegisterRow("lever.fc_uplift", fc_uplift, "expert assumption", "expert-assumption",
                    "Assumed facilitating-conditions uplift for eligible agents; the program is invented, no real-world effect data exists"),
        RegisterRow("abm.peer_influence.weight", peer_influence_weight, "expert assumption", "expert-assumption",
                    "Multiplier weight on neighbour-adopted fraction"),
        RegisterRow("abm.timesteps.T", timesteps, "expert assumption", "expert-assumption",
                    "Fixed diffusion horizon"),
        RegisterRow("abm.n_agents", n_agents, "expert assumption", "expert-assumption",
                    "Population size for the ABM simulation; expert assumption, not derived from data."),
    ]


def adoption_probability(baseline_prob: float, peer_influence_weight: float, fraction: float) -> float:
    return min(baseline_prob * (1 + peer_influence_weight * fraction), 1.0)


BASELINE_PARAMS = dict(n_agents=300, k=6, rewiring_p=0.1, fc_uplift=1.0,
                        peer_influence_weight=0.5, timesteps=20)


def final_adoption_rate(respondents_df, dcm_result, seed: int, **overrides) -> float:
    params = {**BASELINE_PARAMS, **overrides}
    model = DiffusionModel(respondents_df, dcm_result, seed=seed, **params)
    return model.run()["adopted"].mean()


def uncertainty_distribution(respondents_df, dcm_result, n_seeds: int = 30) -> list[float]:
    """Sweeps its own internal seed range (0..n_seeds-1) — deliberately independent
    of any caller-supplied pipeline seed, so this distribution is comparable across
    scenario runs made with different seeds. The scenario manifest's recorded `seed`
    therefore governs the point-estimate diffusion run only, not this sweep."""
    return [final_adoption_rate(respondents_df, dcm_result, seed=s) for s in range(n_seeds)]


_SENSITIVITY_RANGES = {
    "k": [3, 4, 6, 8, 10],
    "rewiring_p": [0.0, 0.05, 0.1, 0.2, 0.3],
    "fc_uplift": [0.0, 0.5, 1.0, 1.5, 2.0],
    "peer_influence_weight": [0.0, 0.25, 0.5, 0.75, 1.0],
    "timesteps": [10, 15, 20, 25, 30],
    "n_agents": [100, 200, 300, 400, 500],
}


def one_at_a_time_sensitivity(respondents_df, dcm_result) -> dict:
    """Sweeps each parameter at a fixed seed=42, independent of any caller-supplied
    pipeline seed — same rationale as `uncertainty_distribution`: comparability
    across scenario runs, at the cost of this sweep not reflecting the manifest's
    recorded seed."""
    tornado = {}
    for param, values in _SENSITIVITY_RANGES.items():
        rates = [final_adoption_rate(respondents_df, dcm_result, seed=42, **{param: v}) for v in values]
        tornado[param] = (min(rates), max(rates))
    return tornado
