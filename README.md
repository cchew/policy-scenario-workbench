# Policy Scenario Workbench

_Blog post: [Simulating Policy Adoption with Rules-as-Code, DCM and ABM](https://www.herdmentality.xyz/blog/policy-scenario-workbench)_

A three-layer pipeline for simulating behavioural response to a policy lever, built to keep what's measured (a statistical model fit on real survey data) separate from what's assumed (the causal links wiring it together). POC scope, toy policy scenario, real survey data.

1. **Rules-as-code** (`src/rac.py`) — eligibility as a [JSON-Logic](https://jsonlogic.com/) document (`rules/eligibility.json`), evaluated by a small generic rule engine, not a hardcoded conditional.
2. **Discrete choice model** (`src/dcm.py`) — a `statsmodels` model of one person's adoption probability, fit on real survey responses.
3. **Agent-based model** (`src/abm.py`) — a `mesa` population resampled from those same respondents, connected on a small-world network, stepped forward in time as adoption spreads.

The toy scenario is calibrated against a real survey of AI adoption at a Vietnamese university (59 responses, scored against [UTAUT](https://en.wikipedia.org/wiki/Unified_theory_of_acceptance_and_use_of_technology)). The policy lever is an invented facilitating-conditions support program (a training/support budget), eligibility gated by role. See `docs/causal-dag.md` for the full causal chain from lever to diffusion, and the write-up linked above for why this evidence/assumption split beats LLM-agent simulation for this kind of question.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Download the dataset per `data/raw/README.md`.

## Run

```bash
python3 -c "from src.run import run_pipeline; run_pipeline('data/raw/ai_adoption_vietnam.csv', '../docs/causal-dag.md', 'outputs/demo_run', seed=42)"
```

Outputs land in `outputs/demo_run/`:

| File | What it is |
|---|---|
| `register.csv` | Assumption-and-evidence register — every causal edge in `causal-dag.md`, tagged as measured or assumed |
| `cohort_breakdown.csv` | Adoption outcomes by eligibility cohort |
| `uncertainty_distribution.csv` | Spread of the point estimate across a fixed-seed repeat sweep |
| `sensitivity_tornado.png` | One-at-a-time sensitivity sweep across baseline parameters |
| `scenario_manifest.json` | Run provenance — seed, parameters, dataset content hash |
| `deferral_statement.md` | What this POC deliberately doesn't model |
| `causal-dag.md` | Copy of the causal DAG used for this run |

## Tests

```bash
pytest -v
```
