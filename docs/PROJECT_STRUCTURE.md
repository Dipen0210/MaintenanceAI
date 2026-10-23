# Repository layout

| Folder | Contents |
| --- | --- |
| `backend/routes/` | Authorized FastAPI endpoints |
| `backend/core/` | Database, security and demo initialization |
| `backend/ingestion/` | Sensor adapters, ingestion and condition checks |
| `backend/ml/`, `backend/risk/` | Predictive registry and maintenance prioritization |
| `backend/copilot/` | Retrieval, generation, tools and evaluations |
| `backend/scheduling/` | Solver, approvals and field execution |
| `backend/services/` | Preserved legacy plant intelligence |
| `frontend/src/app/operations/` | Main workspace pages |
| `frontend/src/components/` | Shared workspace and equipment components |
| `src/`, `notebooks/` | Predictive models, inference and training notebooks |
| `models/` | Artifact manifests; no trained weights |
| `fixtures/` | Synthetic procedures and evaluation inputs |
| `infrastructure/migrations/` | Ordered database migrations |
| `tests/` | Backend regression tests |
| `frontend/tests/`, `frontend/e2e/` | Component and browser regressions |
| [scripts](../scripts/README.md) | Training, evaluation, verification and launch commands |
| [requirements](../requirements/README.md) | Python dependency groups and environment lock |
| `.github/workflows/` | Automated regression and manual release checks |
| [docs](README.md) | Public guides, architecture and reference benchmarks |

The API entry point is `backend/main.py`; base Python dependencies remain in `requirements.txt`. Existing Python imports and CLI entry points are stable.

Local-only data belongs under ignored `artifacts/`, `Data/` or `private/`. The virtual environment, model weights, databases, credentials and build caches are not published. Historical test results and screenshots are organized locally under `docs/design/evidence/`, `docs/design/screenshots/` and `docs/evaluation/`.
