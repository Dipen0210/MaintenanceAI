# Copilot evaluation and release controls

Phase 5 uses the same transactional application database for traces, immutable configuration identities, measured evaluation runs, declared human reviews and release decisions. It does not claim a deployed Langfuse/LangSmith service. `/operations/observability` is the authorized review interface.

## Reproducibility

`backend/copilot/configuration.py` hashes the active prompt, response/tool schemas, retrieval settings, model manifests, relevant inference/decision/access implementation, evaluation runner and evaluator. Each request records its configuration ID. A changed configuration cannot reuse an older evaluation as release evidence. Model files remain external and are verified by the provider before inference.

`fixtures/evaluation/copilot-v1.json` contains twelve authored synthetic scenarios and explicit task expectations. The fixture is **not expert-reviewed**. Its hash/version accompanies every run. Each scenario uses an isolated database, seeded synthetic equipment, actual trained conveyor inference, actual neural retrieval and actual local language-model inference where applicable. Corrupted model records, conflicting scalar data and malicious documents are explicitly described as injected scenario conditions, never measured industrial observations. Refusal and unsupported-RUL cases intentionally invoke no model.

```sh
venv/bin/python -m scripts.evaluate_copilot
venv/bin/python -m scripts.check_copilot_release docs/evaluation/copilot-evaluation-tools_rag.json --technical-only
venv/bin/python -m scripts.verify_copilot_release_regressions
```

The runner writes three JSON reports and persists them in the local demo database. It refuses production mode and nonlocal providers. Each report retains questions, scenario setup, answers, citation snapshots, retrieval/tool traces, actual tokens/latency, deterministic checks and configuration identity. IDs/timestamps are newly generated; the underlying synthetic signal seed, model revision, prompt and expectations are fixed. Local hardware cost is unknown, not zero.

## Comparisons and limits

The three configurations use the same constrained evidence-selection architecture: no retrieval, documentation only, and documentation plus equipment tools. This is a controlled retrieval ablation. The no-retrieval configuration cannot cite measurements; documentation alone cannot establish a current machine fault. Their diagnostic abstention is partly a deliberate consequence of the safety policy. Results must not be represented as a general benchmark against unconstrained chatbots or proof of improved real-world reliability.

Checks independently resolve citations, compare claim text, validate evidence roles and equipment scope, screen unsafe content, enforce human review, verify abstention/refusal and probe authorization of all eleven tools plus four forbidden tool names. Task success checks the authored expected status and required content groups; it is not expert confirmation of a root cause. Latency is end-to-end service time after one-time language-model initialization; process startup/model loading is excluded.

The initial full retrieval evaluation exposed an empty-array schema-enforcement defect. The backend safely rejected the model's attempted wrong-role citation, but the expected abstention became unavailable. The failed measured report is retained as `copilot-evaluation-initial-failure.json`. Prompt v3 and omission of unsupported schema fields fix this path; a real-model regression test covers it.

## Release decisions

Technical gates require all twelve scenarios, current dataset/configuration identities, actual inference provenance, all mandatory critical checks and all authorization probes. They reject unavailable responses, task success below 90%, and any task-success regression against an accepted baseline. The threshold is a software demonstration criterion, not an industrial safety guarantee. Aggregates are recalculated from case results; a green summary cannot override a failing critical check.

The production release gate also requires a qualified person's concrete review. The UI records reviewer identity, declared qualification, decision, findings and explicit coverage of every case. The application cannot independently certify the reviewer's expertise. Automated tests use isolated test identities and never create a real expert review.

For a CLI/CI release, a real reviewer supplies `docs/evaluation/copilot-expert-review.json` with `run_id`, `reviewer`, `qualification`, `decision`, `reviewed_case_ids` and findings for the exact preserved run. No such review is fabricated or auto-generated. Then run:

```sh
venv/bin/python -m scripts.check_copilot_release docs/evaluation/copilot-evaluation-tools_rag.json --review docs/evaluation/copilot-expert-review.json --output docs/evaluation/copilot-release-gate.json
```

Missing review or failed checks produce a nonzero exit status. `--technical-only` is explicitly not release approval. The regression verifier mutates copies of measured reports and proves rejection of safety, citation, scope, authorization, quality and configuration regressions; its injection evidence is separate from actual model metrics.

## CI and operational boundaries

The regular hosted workflow runs deterministic release-control tests and frontend checks. It does not claim full inference without external artifacts. The manually dispatched release workflow requires a trusted `maintenanceai-evaluation` runner with the project environment and legitimate model/data artifacts provisioned. It has two stages: measure and upload evidence; then preserve and review that exact report before running release checks. A release check does not regenerate the reviewed run. No runner or remote CI execution is currently claimed as configured or successful.

The observability API enforces role, plant, asset, source and document access. Engineers inspect their own accessible traces; reviewers can inspect authorized traces and synthetic evaluation results. Revoked private evidence is withheld. Release acceptance records a software gate decision; it does not deploy code, connect OT systems or authorize maintenance work.
