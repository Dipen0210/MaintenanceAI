import { defineConfig } from '@playwright/test';
import { randomUUID } from 'node:crypto';
export default defineConfig({
  testDir: './e2e',
  workers: 1,
  use: { baseURL: 'http://localhost:3000', browserName: 'chromium', channel: 'chrome', trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  webServer: [
    { command: 'cd .. && venv/bin/python -m scripts.serve_demo_e2e', url: 'http://127.0.0.1:8000', reuseExistingServer: false, timeout: 120000, env: { MAINTENANCEAI_DB: `/tmp/maintenanceai-e2e-${randomUUID()}.db`, MAINTENANCEAI_SIMULATION_ENABLED: '0' } },
    { command: 'npm start', url: 'http://localhost:3000', reuseExistingServer: false, timeout: 120000 },
  ],
});
