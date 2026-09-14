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
