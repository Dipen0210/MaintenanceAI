# Evaluation and reproduction

The public repository includes reference predictive benchmarks and explicitly synthetic risk/scheduling scenarios used by code and regression tests. Local model weights and raw datasets must be supplied separately; see [local setup](../deployment/LOCAL_DEMO.md).

| Evidence | Scope |
| --- | --- |
| `audio_fan.json`, `audio_pump.json`, `audio_valve.json` | Audio baseline recording hashes, partitions and measured results |
| `legacy_baseline.json` | Preserved CWRU and C-MAPSS results and limitations |
| `rul_training_smoke.json` | One-epoch training smoke; not a new full training result |
| `synthetic-models.json` | Actual fitting and evaluation on generated conveyor data |
| `risk-scenarios.json`, `risk-benchmark.json` | Authored risk fixtures and software expectations |
| `scheduling-benchmark.json` | Solver comparison on authored maintenance scenarios |

The audio baseline has weak discrimination. Legacy model selection/splits limit scientific validity, and synthetic scores do not establish real-machine diagnostic accuracy. Expert review remains pending; technical checks do not replace that requirement or authorize maintenance.

## Reproduce locally

Run from the repository root with the verified artifacts and legitimate datasets available:

```sh
venv/bin/python -m scripts.evaluate_legacy_models
venv/bin/python -m scripts.evaluate_risk_scenarios
venv/bin/python -m scripts.evaluate_scheduling
venv/bin/python -m scripts.evaluate_copilot --mode all
```

Training commands and model prerequisites are documented in the deployment guide. Evaluation scripts write local outputs here. Generated Copilot conversations, review packets, test reports, screenshots, debug traces and source snapshots are excluded from public commits by default. The demo imports existing actual Copilot reports only when their configuration matches; a fresh clone needs its own evaluation before those results can appear.

Locally retained evidence is grouped into `phases/`, `test-results/`, `snapshots/`, `screenshots/`, `media/`, `field-work/` and `machine-onboarding/`. Moving these records does not create new evaluation results. Historical JSON/XML content is preserved unchanged.
