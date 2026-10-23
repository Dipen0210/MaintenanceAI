# Reproducible local demonstration

The tested deployment is native macOS ARM64, Python 3.13.2, the project `venv`, Node.js and a production Next.js build. SQLite and original evidence files remain local. Docker is not installed in the validation environment; no container build, cloud deployment or remote CI success is claimed. The specification's container/database products are suggested technologies; this implementation uses the verified modular native deployment and equivalent local registry/tracing/vector storage.

## Install and obtain legitimate artifacts

From the repository root:

```sh
python3.13 -m venv venv
venv/bin/python -m pip install -r requirements/scheduling.txt
cd frontend
npm ci
npm run build -- --webpack
cd ..
```

`requirements/lock.txt` records the actual local Python environment; `frontend/package-lock.json` records the frontend packages. Other operating systems/architectures need their compatible wheels and independent validation. Always use `venv/bin/python`, or activate this same environment before Python commands.

The repository intentionally excludes large trained weights, raw datasets and local databases. A fresh clone needs the trusted Phase 0 bundle from the project maintainer; there is no public bundle URL configured. Its contents are checked against the committed manifest before installation:

```sh
venv/bin/python -m scripts.model_artifacts import --bundle /path/to/maintenanceai-phase0-models.tar.gz
venv/bin/python -m scripts.model_artifacts verify
venv/bin/python -m scripts.train_synthetic_models
venv/bin/python -m scripts.download_language_models
```

The synthetic training script actually fits the conveyor models on explicitly generated, disjoint-seed datasets. It is not a substitute for physical-machine training data. Language downloads use pinned public model revisions and checksums; approximately 1.3 GB of model weights are required. Downloads require network access; subsequent local inference does not use a cloud API key. Existing weights can be preserved/exported through `scripts.model_artifacts`; never relabel changed weights with an old checksum.

Full legacy evaluation and training tests additionally require the legitimate local MIMII, CWRU and NASA C-MAPSS dataset copies described in `docs/architecture/PHASE0_INVENTORY.md`. Runtime demonstration does not require those raw datasets once verified weights/reference statistics are installed. Missing artifacts are explicit failures, never placeholder predictions.

## Start and stop

```sh
venv/bin/python -m scripts.run_demo
```

Open `http://localhost:3000/operations`. The launcher verifies all predictive and language artifact checksums and the production frontend build, refuses occupied ports, starts both services bound to loopback, and checks actual startup health. Ctrl-C stops only its own services and preserves data. Logs are under `artifacts/deployment/`.

Local accounts: `engineer`, `manager`, `reliability`, `admin`, with `<username>-demo-password`. These accounts belong only to the simulated facility. An administrator can create scoped local identities in **Access and asset administration** and revoke sessions by changing their access. Do not reuse demo credentials/data for a real facility.

The default database is `artifacts/maintenanceai.db`. Use `--database /absolute/path/demo.db` for a separate demonstration. Migrations run at startup; an existing database is not reset. Original signals/documents are stored in sibling `signals/` and `documents/` directories. Use a separate directory when separating evidence stores between environments.

The default simulator updates only assets explicitly mapped to `demo-simulator`. `MAINTENANCEAI_SIMULATION_ENABLED=0` disables those background V2 ticks for repeatable browser scenarios; the explicit **Analyze synthetic conveyor incident** action still generates labeled observations and executes the learned model. Legacy public simulation remains separate.

## Run the complete incident

1. Sign in as engineer. From the Dashboard, open **Assets**, then **Demo conveyor motor**.
2. Open **Predictions** and analyze the synthetic conveyor incident. Inspect actual model version, candidate class, spectrum and uncertainty limits. The source also records a distinct, explicitly simulated alarm event.
3. Open **Maintenance risk** and evaluate current maintenance risk and inspect components, criticality and required human verification.
4. Open **Ask Copilot** in the top bar and ask why the motor is flagged. Inspect observations, predictions, unconfirmed hypotheses, alternatives, approved checks and citation sources. Save/export the investigation draft.
5. If the observations have expired during investigation, close Copilot and refresh the synthetic incident in **Predictions**; a historical investigation never becomes fresh sensor evidence automatically.
6. Reopen Copilot and follow **Draft maintenance from this investigation**, select approved passages, skills, duration and parts, then save the draft with its current risk snapshot.
7. Sign out; sign in as manager. Open **Maintenance → Work orders** and review the candidate's risk/procedure evidence and record an approval reason. Select the approved candidate and propose a feasible schedule.
8. Open **Schedules**, inspect assignments and conflicts, then confirm. The schedule is stored internally, appears in the Dashboard and asset Overview and remains after reload. It does not execute maintenance or control equipment.
9. Inspect trace/feedback/evaluation evidence in **AI evaluations**. Expert review remains deferred; no review should be fabricated to make a release indicator green.

Initial simulated planning resources cover the next day, 08:00–16:00 UTC, including a conveyor restriction from 11:00–12:00. After those dates pass, a manager must update working periods and equipment windows in **Maintenance → Resources**. Restarting never silently shifts approved resource calendars. Confirmed work reserves parts permanently until a separately reviewed inventory reconciliation mechanism exists.

## Verify the actual deployment

```sh
venv/bin/python -m scripts.run_demo --database /tmp/maintenanceai-smoke/demo.db --smoke --output docs/evaluation/local-deployment-smoke.json
venv/bin/python -m pytest -q
venv/bin/python -m scripts.evaluate_scheduling
venv/bin/python -m scripts.evaluate_copilot
venv/bin/python -m scripts.check_copilot_release docs/evaluation/copilot-evaluation-tools_rag.json --technical-only
cd frontend
npm run lint
npm test
npm run build -- --webpack
npm run test:e2e
```

For browser verification, install Google Chrome and the recording binary matching the locked Playwright version: run `cd frontend` then `npx playwright install ffmpeg`. This downloads the actual recording runtime; it does not install a system-wide FFmpeg.

Run startup smoke and browser tests sequentially: both own ports 3000 and 8000. Browser tests use a separate temporary database, a production frontend, actual model artifacts and installed Chrome. Their complete suite records screenshots and an actual workflow video. The no-retrieval/document-only comparisons deliberately abstain without current equipment evidence; consult the benchmark report before interpreting scores.

The recorded deployment smoke checks 14 predictive artifacts and both language model revisions; the backend loads all 12 preserved legacy models, serves the frontend, accepts authorized planning reads, rejects anonymous asset access and shuts down its own processes. That evidence proves local process startup/health, not a public deployment or industrial uptime guarantee.

## Backup and extension limits

Stop the local services before a simple filesystem backup; preserve the database, associated signal/document directories, model files and their matching manifests together. For online backups, use SQLite's backup API and coordinate original-file retention. Restore into a separate local environment and verify artifact hashes before use.

Do not edit code, manifests or fixtures during a measured evaluation. The runner pins expected identities across all modes and rejects source/dataset changes before finalizing reports. Update the pending reviewer packet with `venv/bin/python -m scripts.prepare_review_packet`; this writes pending fields only and never creates an expert approval.

Optional protocol verification uses `requirements/industrial.txt` and `scripts.verify_opcua` against a loopback test server. Live integrations require the separately approved gateway and controls described in `docs/security/SECURITY.md`. They are not necessary for the standalone demo.
