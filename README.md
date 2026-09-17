# Policy Scenario Workbench

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

Outputs land in `outputs/demo_run/`: `register.csv`, `cohort_breakdown.csv`,
`uncertainty_distribution.csv`, `sensitivity_tornado.png`,
`scenario_manifest.json`, `deferral_statement.md`, `causal-dag.md`.

## Tests

```bash
pytest -v
```
