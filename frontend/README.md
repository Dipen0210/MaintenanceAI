# MaintenanceAI frontend

Next.js workspace for machine monitoring, maintenance coordination and Engineer Copilot. Follow the [local setup guide](../docs/deployment/LOCAL_DEMO.md) to prepare the API, models and database.

## Commands

Run from `frontend/`:

```sh
npm ci
npm run dev
npm run lint
npm test
npm run build -- --webpack
```

Use `npm start` for the production build. The workspace opens at `http://localhost:3000/operations`; the API runs on loopback port 8000. From the repository root, `venv/bin/python -m scripts.run_demo` launches both services after prerequisites pass.

`npm run test:e2e` owns ports 3000 and 8000 and uses a temporary database. Stop existing services before running it. It needs a production build, verified local models, Chrome and the matching Playwright recording runtime. Generated screenshots, recordings and test reports are ignored by Git.

## Layout

| Location | Purpose |
| --- | --- |
| `src/app/operations/` | Dashboard, assets, priorities, maintenance, investigations and administration |
| `src/components/workspace/` | Shared shell, navigation and contextual Copilot |
| `src/components/operations/` | Equipment, sensor maps and maintenance UI |
| `src/lib/` | Typed API/planning clients |
| `src/hooks/` | Stream lifecycle |
| `src/app/legacy/` | Preserved reference simulation |
| `tests/` | Vitest regressions |
| `e2e/` | Playwright workflows and responsive checks |

The UI displays backend-authorized evidence. Work status, sensor-limit conditions and verified physical repair remain distinct. See the [field workflow guide](../docs/deployment/FIELD_WORKFLOW.md).
