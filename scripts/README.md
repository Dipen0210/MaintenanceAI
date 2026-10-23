# Project commands

Run from the repository root with `venv/bin/python -m scripts.<name>`. Entry points stay stable for existing commands and measured configuration identities.

| Purpose | Modules |
| --- | --- |
| Local application | `run_demo`, `serve_demo_e2e` |
| Artifact management | `model_artifacts`, `build_model_reference`, `download_language_models` |
| Training | `train_audio`, `train_synthetic_models` |
| Predictive validation | `evaluate_legacy_models`, `verify_rul_training` |
| Copilot validation | `verify_language_models`, `verify_copilot_flow`, `evaluate_copilot` |
| Release evidence | `prepare_review_packet`, `check_copilot_release`, `verify_copilot_release_regressions` |
| Risk and scheduling | `evaluate_risk_scenarios`, `evaluate_scheduling` |
| Demo evidence import | `seed_evaluation_reports` |
| Industrial integration | `opcua_gateway`, `verify_opcua` |

See [local setup](../docs/deployment/LOCAL_DEMO.md) and [machine onboarding](../docs/deployment/MACHINE_ONBOARDING.md) for prerequisites and configuration. Locally generated reports are not published automatically. Do not replace measured evidence with fabricated results when artifacts are unavailable.
