# MaintenanceAI

MaintenanceAI V2 is a locally verified FastAPI and Next.js maintenance demonstration integrating predictive ML, industrial data ingestion, risk prioritization, grounded Engineer Copilot, LLMOps/MLOps and manager-approved scheduling. See the [documentation index](docs/README.md), [repository layout](docs/PROJECT_STRUCTURE.md) and [local setup guide](docs/deployment/LOCAL_DEMO.md).

The dashboard uses **simulated equipment conditions**. It is not connected to SCADA, and no real-plant deployment or reliability benefit is claimed. Direct inference is separate from the simulation.

[Reference benchmarks and reproduction instructions](docs/evaluation/README.md) describe the measured results and their limitations.

## Local setup

Use Python 3.13 for the verified V2 environment and a Node.js version supported by the pinned Next.js package (Node 20.9+). The local verification environment and exact package versions are recorded with evaluation evidence.

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/scheduling.txt
cd frontend
npm ci
cd ..
```

Models and datasets are not committed. The three historical `.pkl` files are evaluation results, not inference models. Obtain the verified model bundle from the project maintainer, then install and verify it:

```sh
python -m scripts.model_artifacts import --bundle /path/to/maintenanceai-phase0-models.tar.gz
python -m scripts.model_artifacts verify
```

The existing local bundle is generated with:

```sh
python -m scripts.model_artifacts export --bundle artifacts/maintenanceai-phase0-models.tar.gz
```

Checksums and model identities are in `models/manifest.json`. Only load trusted model bundles: Python model formats may contain executable serialization. There is no public model-download URL currently configured. A fresh clone without a verified bundle returns explicit model-unavailable responses; it must not invent predictions.

To retrain the audio baseline, place the actual MIMII DUE WAV files under `Data/{fan,pump,valve}/{train,source_test,target_test}` and run:

```sh
python -m scripts.train_audio
```

The shared audio notebook invokes this same pipeline. Training consumes normal recordings only, calibrates on a disjoint subset and scores untouched test recordings. Results include recording hashes. New artifacts require a newly reviewed manifest; never relabel changed weights with an old checksum.

Start the services in separate terminals:

```sh
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```sh
cd frontend
npm run build
npm start
```

Open http://localhost:3000 for the organized Operations workspace, and API documentation at http://localhost:8000/docs. The original reference dashboard remains at `/legacy`.

If Turbopack cannot create its build-time worker/port in a restricted environment, the supported `npm run build -- --webpack` produces the verified production build. All Python commands should use the activated project virtual environment, or explicitly `venv/bin/python`.

## Supported inference contracts

| Route | Input | Scope |
|---|---|---|
| `POST /predict/audio?machine_type=fan&section=00` | WAV, 1–60 seconds, up to 20 MiB; resampled mono 16 kHz | MIMII fan/pump/valve, sections 00–02; fitted statistical baseline |
| `POST /predict/vibration` | `signal`: multiples of 1024 raw samples, max 65536; `sample_rate`: 48000 | CWRU bearing drive-end waveforms reshaped into 32×32 images; ten classes |
| `POST /predict/rul` | `sequence`: nonempty rectangular array with 17 ordered features | NASA C-MAPSS FD001 turbofan baseline; 80-cycle context, scale then pad |

RUL feature order, preprocessing, architecture inventory and dataset limitations are in [the compatibility report](docs/architecture/PHASE0_INVENTORY.md). Turbofan RUL predictions do not apply to industrial pumps, motors or conveyors. Softmax values and audio percentile scores are not calibrated failure probabilities. No validated uncertainty bounds exist yet.

Legacy plant routes remain: `/plant/summary`, `/plant/areas`, `/plant/machines`, `/plant/machines/{id}`, `/plant/maintenance-queue`, machine updates, and reset. `/ws/updates` carries simulated updates and heartbeat messages. Missing assets return 404; malformed measurements return 4xx without corrupting state. These public legacy routes are an isolated local simulation. The V2 `/api` routes enforce backend authentication and grants.

## Verification and reproducibility

```sh
python -m pytest -q
python -m scripts.evaluate_legacy_models
cd frontend
npm run lint
npx tsc --noEmit --incremental false
npm run build
```

Inference integration tests require the verified local model bundle and public datasets; missing inputs are a failed acceptance gate, not a passing mock. Audio training reports and legacy evaluation JSON are preserved in `docs/evaluation/`.

Measured CWRU accuracy is 96.9565% on the legacy example split. FD001 LSTM/Transformer RMSE is 18.2478/16.8538 cycles. Legacy test selection limits independent validity. The newly trained audio baseline has weak performance; see per-section results before using it for any evaluation or demonstration. No industrial deployment claims are supported.

The application has been validated locally with backend, component and browser workflows. Internal test records and UI recordings are kept outside the public commit. Reproduce checks using the commands above and the deployment guide. Qualified expert review remains pending; this is a simulated application, not industrial deployment certification.

MIT license; see LICENSE.

## V2 operations foundation (Phase 1 validated)

Open `/operations` from the dashboard. In explicitly enabled demo mode, local accounts are `engineer`, `admin`, `manager`, and `reliability`, each with password `<username>-demo-password`. The backend enforces role, plant, asset and source grants. Sessions use HTTP-only cookies; all demo identities must be disabled for any live deployment. `MAINTENANCEAI_MODE=production` disables demo seeding, but production provisioning and deployment hardening are not yet complete.

Application data lives at `artifacts/maintenanceai.db` by default (`MAINTENANCEAI_DB` overrides it). Ordered migrations run at startup. SQLite is the actual transactional local store; a PostgreSQL deployment is not claimed.

CSV/Parquet imports require the administrator role and `demo-import` source. Required columns: `asset_id,tag_id,timestamp_utc,value,unit,quality,source,operating_state`. Supported demo tags are `temperature` (`degC`), `vibration_rms` (`mm/s`), and `current` (`A`). Timestamps must include timezone offsets. Invalid rows are rejected with recorded reasons; stale/poor-quality observations are flagged. Imports do not write to industrial equipment.

Optional protocol verification:

```sh
pip install -r requirements/industrial.txt
python -m scripts.verify_opcua
```

This starts an ephemeral **loopback test server**, not a plant connection. Live OT integration requires an independently controlled, approved gateway and secret provisioning; those services are not configured in this repository.


## V2 predictive services

Before running the conveyor incident, reproduce the synthetic-only model artifacts:

```sh
python -m scripts.train_synthetic_models
```

This trains real Isolation Forest and Random Forest models on explicitly generated conveyor signals. Training, validation, calibration and final test runs use disjoint seeds. The saved benchmark does **not** establish accuracy on real motors. No existing CWRU or C-MAPSS model is repurposed as a conveyor model.

In `/operations`, select the demo conveyor and choose **Analyze synthetic conveyor incident**. The page shows trained outputs, spectrum evidence, uncertainty limitations, model version and persisted history. `/operations/models` shows verified artifacts and monitoring; administrator/reliability roles can activate a registered version or roll back with an audited reason.

Standard V2 inference uses `POST /api/assets/{asset_id}/signals` (vibration or sequence) or `/signals/audio` (original WAV), then `POST /api/assets/{asset_id}/predictions` with `family` and `input_id`. `GET /api/assets/{asset_id}/models` lists approved bindings. A matching equipment type alone is insufficient; a registered dataset binding is required. Historical inputs are explicitly marked. Current signals expire after 60 seconds. Recording content hashes, start/end times, sampling rates, positions and preprocessing identities are preserved.

C-MAPSS sequence imports require `feature_names` in this order: `op1, op2, op3, sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21`. Inputs are unscaled; the registered checkpoint applies its own scaler. Unsupported assets return `UNSUPPORTED_ASSET` without a prediction.

The registry uses transactional SQLite tables instead of a separate MLflow service. Raw signal files are stored beside the configured database under `signals/`. Input distribution warnings use reference standardized distances; they are diagnostic warnings, not calibrated failure probabilities. CWRU reference statistics can be reproduced with `python -m scripts.build_model_reference` using the original dataset.

## Maintenance risk and priority board

After analyzing the demo incident, use **Evaluate current maintenance risk** in Operations. `/operations/risks` evaluates and ranks authorized assets. It exposes supporting evidence, missing channels, criticality and required human verification. An assessment expires with its evidence or when its configuration changes.

Manager, reliability and administrator accounts can configure criticality, operating limits and risk-policy weights with an audit reason. Invalid limits, weight totals, dependency cycles and unauthorized changes are rejected by the backend. All demo limits are synthetic site configuration. The engine never authorizes operation or executes maintenance.

See [the risk policy](docs/architecture/RISK_POLICY.md). Reproduce the explicitly synthetic scenario comparison using `python -m scripts.evaluate_risk_scenarios`.

## Engineer Copilot and documents

Install the copilot dependencies and download the actual revision-pinned local models before starting the backend:

```sh
venv/bin/python -m pip install -r requirements/copilot.txt
venv/bin/python -m scripts.download_language_models
venv/bin/python -m scripts.verify_language_models
venv/bin/python -m scripts.verify_copilot_flow
```

Model files are ignored under `artifacts/language_models/`; `models/language-models.json` records revisions and checksums. The download needs internet access and approximately 1.3 GB for model weights. Inference runs locally without an API key. Missing or invalid artifacts produce explicit unavailable responses.

In **Assets**, open **Demo conveyor motor**. Use its **Predictions** tab to analyze the synthetic incident, then **Maintenance risk** to evaluate priority. Open **Ask Copilot** in the top bar to investigate the evidence. The actual local Qwen model selects authorized evidence; the backend validates and renders measurements, model results and approved passages separately. Open citations, record feedback, add notes and export an investigation draft. Prior conversations can be reopened. Any maintenance still requires qualified human review.

The **Approved documentation** page supports administrator PDF/Markdown/text imports and reviewer approval/revocation. Only approved, reviewed, applicable passages are retrieved. The supplied guide is explicitly synthetic DEMO-2 material; it is not an OEM manual or physical-machine procedure. CMMS CSV history imports are also available on this page. See [copilot architecture and limitations](docs/architecture/COPILOT.md).

Phase 5 adds `/operations/observability` for authorized trace inspection, measured evaluation cases and human review. Run `venv/bin/python -m scripts.evaluate_copilot` to compare the three retrieval configurations locally. [LLMOps documentation](docs/architecture/LLMOPS.md) explains reproducibility, metric limitations and release gates. Expert review and remote CI execution must not be inferred from authored fixtures or local tests; see the [evaluation guide](docs/evaluation/README.md) for evidence scope and reproduction.


## Maintenance planning

Expert review was explicitly deferred by the user on 2026-10-21 so development can continue. It remains pending; this does not waive maintenance manager approval or permit industrial control.

Open **Plan maintenance** on an asset, **Maintenance → New work order**, or **Draft maintenance from this investigation** after exporting an investigation. Select applicable approved inspection passages, enter duration/skills/parts/dependencies and save the evidence-linked draft. A maintenance manager reviews the candidate, selects approved work, proposes a feasible schedule and confirms it. Only final confirmation reserves resources internally. Administrator authority alone cannot approve maintenance.

`/operations/scheduling` separates **Work orders**, **New work order**, **Schedules** and **Resources**. The Resources tab exposes versioned technician, spare-part, maintenance-window and operational-restriction configuration. Demo resources initially cover the next day, 08:00–16:00 UTC. Update those periods when they become historical; restarting does not silently change the calendar. Times shown in planning are UTC. The solver returns explicit conflicts or a validated feasible plan; failed search limits are not mislabeled as infeasibility.

```sh
venv/bin/python -m pytest tests/test_phase6_scheduling.py -q
venv/bin/python -m scripts.evaluate_scheduling
```

See [scheduling architecture](docs/architecture/SCHEDULING.md) for constraint, approval, stock-reservation and optimization limits. No external CMMS write or physical maintenance completion is implied.


## Integrated local demonstration

Use `venv/bin/python -m scripts.run_demo` after installing the verified model artifacts and building the frontend. The launcher checks prerequisites, binds both services to loopback and preserves data when stopped. The complete setup, incident walkthrough, backup boundaries and actual startup checks are in [local deployment](docs/deployment/LOCAL_DEMO.md).

Operations now connects alarms, risk, work orders and upcoming confirmed maintenance. Administrators can manage local identities/grants, revoke sessions, register file sources and explicit asset/sensor contracts, inspect mappings/audits and configure facility metadata. Model selections and activation histories are isolated per plant. None of these actions opens an industrial connection.

See the [system architecture](docs/architecture/SYSTEM.md), [security boundaries](docs/security/SECURITY.md) and [evaluation guide](docs/evaluation/README.md).

## Workspace navigation

The home URL opens the authenticated dashboard. Use the left navigation in task order:

1. **Dashboard:** actual authorized summaries, priorities, upcoming confirmed work and a guided starting point.
2. **Assets:** search/filter equipment, then use Overview, Sensor readings, Predictions and Maintenance risk tabs.
3. **Priorities:** compare current assessments; expand configuration only when needed.
4. **Maintenance:** review work orders, create drafts, inspect schedules and configure resources in separate views.
5. **Investigations:** a dedicated workspace for cited conversations and report export.

**Ask Copilot** is available on every Operations page, retains the selected equipment context and preserves the drawer conversation while navigating. Documents, Data sources, Model registry and AI evaluations are grouped separately. Administration appears for authorized administrators. The mobile menu and Copilot support keyboard focus and Escape. Nothing in the UI grants backend access or approves maintenance automatically.

See the [field workflow walkthrough](docs/deployment/FIELD_WORKFLOW.md) for the complete engineer-to-manager process.

### Machine simulation and sensor onboarding

Use **Run simulation** in the workspace header to start/pause the demo readings. **Machine condition** shows current sensor-limit evidence independently of saved risk assessments. **Assets → Add machine** guides administrators through registering actual equipment, multiple sensors, reviewed limits and a source-bound gateway key. **Machine & sensor map** provides a shaded technical illustration with clickable live-value markers, optional private machine-photo upload and editable sensor locations.

The reference plant integration uses a read-only OPC UA bridge and the authenticated telemetry API. See [machine onboarding and gateway deployment](docs/deployment/MACHINE_ONBOARDING.md) for the complete workflow and exact limits. Registration alone does not connect a plant, qualify a predictive model or establish industrial readiness.

### Field work and repair verification

The demo condition selector includes mixed, normal, warning and critical scenarios. Confirmed work orders now support engineer start/coordination notes, repair reports, and independent maintenance-manager verification or return for further work. Status and history appear in the dashboard, equipment activity, maintenance views and authorized Copilot evidence. Closing work does not change sensor measurements or clear alarms. See [Field workflow and demo walkthrough](docs/deployment/FIELD_WORKFLOW.md).

### Field issue intake and personal work queues

Use **Issues** to report an observed symptom without needing sensor data or a prediction. Maintenance managers assign investigations to named engineers; **My Work** shows those investigations and confirmed jobs bound to the signed-in account. Reports support protected photo evidence and versioned notes. Closing an investigation does not mark a machine repaired.

The sidebar **Workspace** selector separates demo equipment from other registered machines within the authorized site. It clears prior machine/Copilot context when switching. Administration and planning resources remain site-wide; production tenant provisioning is still a separate requirement. See the [field workflow guide](docs/deployment/FIELD_WORKFLOW.md) for assignment, photo limits and remaining capabilities.

Machine QR labels and **Scan machine** provide authenticated equipment access. **Preventive plans** supports recurring calendar tasks that generate draft work for the existing human approval workflow. See [field workflow](docs/deployment/FIELD_WORKFLOW.md) for calendar rules and current limitations. Meter-triggered plans and inspection rounds remain future work.
