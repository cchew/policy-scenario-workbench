import mesa
import networkx as nx
import pandas as pd

from src.dcm import FEATURES, predict_probability
from src.register import RegisterRow
from src.schema import LIKERT_MAX


class AdoptionAgent(mesa.Agent):
    def __init__(self, model, eligible: bool, baseline_prob: float):
        super().__init__(model)
        self.eligible = eligible
        self.baseline_prob = baseline_prob
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
        for node_id, row in sample.iterrows():
            fc = row["fc_score"]
            eligible = bool(row["eligible"])
            if eligible:
                fc = min(fc + fc_uplift, LIKERT_MAX)
            features = {f: row[f] for f in FEATURES}
            features["fc_score"] = fc
            X = pd.DataFrame([features])[FEATURES]
            baseline_prob = float(predict_probability(dcm_result, X).iloc[0])
            agent = AdoptionAgent(self, eligible=eligible, baseline_prob=baseline_prob)
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
            prob = adoption_probability(agent.baseline_prob, self.peer_influence_weight, fraction)
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
                       peer_influence_weight: float, timesteps: int) -> list[RegisterRow]:
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
    return [final_adoption_rate(respondents_df, dcm_result, seed=s) for s in range(n_seeds)]


_SENSITIVITY_RANGES = {
    "k": [3, 4, 6, 8, 10],
    "rewiring_p": [0.0, 0.05, 0.1, 0.2, 0.3],
    "fc_uplift": [0.0, 0.5, 1.0, 1.5, 2.0],
    "peer_influence_weight": [0.0, 0.25, 0.5, 0.75, 1.0],
    "timesteps": [10, 15, 20, 25, 30],
}


def one_at_a_time_sensitivity(respondents_df, dcm_result) -> dict:
    tornado = {}
    for param, values in _SENSITIVITY_RANGES.items():
        rates = [final_adoption_rate(respondents_df, dcm_result, seed=42, **{param: v}) for v in values]
        tornado[param] = (min(rates), max(rates))
    return tornado
