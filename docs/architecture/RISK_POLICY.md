# Auditable maintenance risk policy

The engine evaluates immutable observations and compatible recent predictions against versioned plant policies and asset profiles. All demo thresholds are synthetic site configuration, not OEM-approved limits for real machinery.

For valid required channels, an observation below the warning limit contributes zero degradation, warning-to-critical contributes 0.5–1, and a critical excursion contributes 1. A compatible current anomaly contributes its normal-calibration percentile only when flagged; a non-normal classification contributes a fixed 0.75 proxy. These are proxies, never calibrated failure probabilities. Historical, stale, unsupported, drift-flagged, revoked-binding or invalid model results are excluded. An older result cannot silently replace an invalid latest result from the same family.

The maximum supported degradation proxy is multiplied by the weighted criticality consequence and 100. Default consequence weights are safety 0.4, production 0.3, maintenance 0.15 and equipment damage 0.15; each criticality is configured on 1–5. Production dependencies are explicit context for the human-assigned production criticality. Cycles and unauthorized dependencies are rejected.

Default score thresholds: planned 25, high 50, critical 85. A safety criticality of at least 4 combined with a critical observed limit requires critical review regardless of other evidence. A strong compatible anomaly on a low-consequence asset still requires planned inspection. Required sensor evidence that is absent, stale, bad, rejected, unit-incompatible or outside approved operating states yields **Insufficient evidence** and no numerical score. Critical observed hazards retain mandatory escalation even with other missing channels.

Ranking order is confirmed critical review, confirmed high priority, insufficient evidence requiring verification, planned maintenance, and monitor. Scores break ties within a category; asset identifier breaks remaining ties reproducibly. A monitor result never authorizes operation.

Each stored assessment retains policy/profile versions, observation/model references, source permissions, missing evidence, explanation and mandatory human verification. Responses flag assessments stale when evidence expires or the active policy/profile changes. Existing assessments are not rewritten. A user cannot retrieve an assessment containing source evidence they cannot access.

Administrator, reliability engineer and maintenance manager roles can update criticality and policy with an audit reason. Engineers can evaluate authorized assets, but cannot change decision policy. Evaluating a batch is atomic with respect to authorization failure.

Reproduce the authored software benchmark with `python -m scripts.evaluate_risk_scenarios`. Eight synthetic scenarios and one three-asset ranking comparison are recorded in `docs/evaluation/risk-benchmark.json`. These narrow fixtures do not establish industrial effectiveness or expert approval.
