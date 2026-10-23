# Security and industrial safety boundaries

The validated deployment is a local simulated facility. Expert review is deferred by explicit user instruction dated 2026-10-21. This deferral does not waive maintenance-manager decisions, evidence authorization, site isolation/permits or facility OT/security approval for any future live connection. Automated software checks do not certify industrial diagnostic safety.

## Enforced controls

| Boundary | Enforcement | Deterministic regression evidence |
|---|---|---|
| Authentication / session revocation | PBKDF2 passwords; hashed eight-hour sessions; enabled-user checks; access changes revoke all target sessions | `test_phase1_foundation.py`, `test_phase7_security.py` |
| Role separation | Engineer investigates/drafts; manager alone reviews maintenance candidates and schedules; administrator manages local access/imports | `test_phase6_scheduling.py`, `test_phase7_security.py` |
| Plant / asset / source isolation | Every protected operation checks current principal and scoped grants; malformed cross-plant grants do not authorize data | `test_phase1_foundation.py`, `test_phase4_copilot.py`, `test_phase7_security.py` |
| Document applicability | Equipment/asset scope, document grants, reviewed passages and approved revisions checked at retrieval, citation, export and maintenance approval | `test_phase4_copilot.py`, `test_phase6_scheduling.py` |
| Prompt injection / credentials / unsafe requests | Quarantine/refusal screening, bounded evidence-selection schema, independent source/role checks, no arbitrary tool execution | `test_phase4_copilot.py`, `test_phase5_llmops.py`, actual twelve-case evaluations |
| Industrial read-only access | OPC UA node reads and historian GET through externally approved clients; no application write/control tool | `test_phase1_foundation.py`, loopback `phase1-opcua.json`, tool authorization probes |
| Stale / invalid observations | Finite values, explicit units/bounds/time/quality/state, rejected ingestion recorded, stale evidence excluded from current diagnosis | `test_phase0_inference.py`, `test_phase1_foundation.py`, `test_phase2_ml.py`, `test_phase3_risk.py` |
| Model provenance | Trusted catalog hashes, immutable model/input identities, dataset bindings and operating-state checks | `test_phase0_inference.py`, `test_phase2_ml.py` |
| Scheduling approval | Current enabled manager role; reauthorize evidence; serialized independent feasibility checks against latest reservations | `test_phase6_scheduling.py`, `test_phase7_security.py` |
| Configuration / release regression | Content-addressed implementation, model, prompt and dataset identities; complete current measured evidence required | `test_phase5_llmops.py`, injected regression report, release CLI |

The native browser suite also verifies engineer-to-manager approval, scoped-account access and revocation of a live session. A database fixture or model-output injection is labeled as such in tests; it is not reported as actual industrial evidence or actual model accuracy.

## Administration

`/api/admin/identities` requires a current enabled administrator with whole-plant asset visibility. Grant changes validate all target assets, sources and documents, refuse foreign-plant users/resources, preserve at least the requesting administrator by forbidding self-access changes, and revoke existing target sessions. Passwords are hashed and never included in responses or audit messages. Backend validation errors omit submitted values.

Registering a source creates a read-only **file import source** only. It does not accept an arbitrary endpoint, credential or write-capable connection. Registering an asset requires explicit sensor contracts and matching reviewed risk limits, creates no observations and grants no model applicability automatically. File-only assets are excluded from automatic demo measurements. Existing measurement units/bounds and model bindings are not silently rewritten by metadata edits.

The audit interface is administrator-only and plant-scoped. Evidence reads remain separately protected by asset/source/document grants. An audit event records actor, action, resource, time and outcome; it is not a substitute for permission enforcement. SQLite files, model artifacts and logs require appropriate filesystem protection from the host administrator.

## Browser and runtime boundaries

HTTP-only SameSite-strict cookies are used; cookie mutations and telemetry WebSockets require the configured local application origins. Bearer sessions support trusted CLI/API use. The tested launcher binds frontend and backend to `127.0.0.1`, checks for occupied ports, and only terminates process groups it started. It preserves the database and uploaded evidence when stopped.

Legacy routes remain public **local simulation/reference-data** functionality. They must not be exposed as an industrial production interface. Local demo credentials are documented conveniences, never production secrets. `MAINTENANCEAI_MODE=production` prevents demo seeding and sets secure cookies, but is not by itself a production deployment or migration plan. Existing demo identities/data must not be reused in a live environment.

## Unvalidated operational work

No real SCADA/PLC/historian/CMMS connection, industrial credential, secrets-manager integration, hosted identity provider, public deployment, penetration test, facility review or production availability SLA is claimed. The optional adapters accept an externally controlled, approved gateway client; that gateway must own certificate validation, least-privilege credentials from a secrets manager, network segmentation, timeouts and OT-owner approval. A loopback OPC UA fixture proves protocol reads and normalized ingestion, not plant connectivity.

Before any live deployment, obtain facility OT/security approval and qualified maintenance/reliability review; provision non-demo identities, TLS/origins, secrets management, network controls, upload/traffic limits, backups/retention and operational monitoring; validate actual equipment/model/procedure applicability. These are future deployment prerequisites, not automatically completed by a passing synthetic evaluation.

Regex screening can reject benign quotations and cannot identify every attack. The stronger controls are restricted data/tool access, constrained evidence selection, independent claim validation, absence of industrial-write interfaces and human maintenance approval. Model definitions for the public/synthetic catalog are shared, but activation choices and history are plant-scoped. An actor must hold access to all affected bound assets in that plant before activation. Migration 007 preserves existing choices and attributes prior activation events to their actors’ plants.

A compromised host or authorized administrator modifying database/model files is outside the application's software-only trust boundary.
