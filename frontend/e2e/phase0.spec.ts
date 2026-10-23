import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

test('production dashboard → area → machine; API prediction and invalid update', async ({page, request}) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/legacy');
  await expect(page.getByText('Simulation connected')).toBeVisible();
  await expect(page.getByText('WS', {exact:true})).toBeVisible();
  await page.locator('a[href="/area/production_line_a"]').click();
  await expect(page.getByRole('heading', {name:'Production Line A',exact:true})).toBeVisible();
  await page.locator('a[href="/machine/PLA-FAN-001"]').click();
  await expect(page.getByRole('heading',{name:'PLA-FAN-001',exact:true})).toBeVisible();
  const result=await request.get('http://127.0.0.1:8000/');
  expect((await result.json()).models_loaded.total).toBe(12);
  const invalid=await request.post('http://127.0.0.1:8000/plant/machines/PLA-FAN-001/update?rul_cycles=nan');
  expect(invalid.status()).toBe(422);
  expect((await request.get('http://127.0.0.1:8000/plant/machines')).status()).toBe(200);
  const report=JSON.parse(readFileSync(resolve('../docs/evaluation/audio_fan.json'),'utf8'));
  const file=report.records.find((r:{role:string;section:string})=>r.role==='source_test' && r.section==='00').path;
  const audio=await request.post('http://127.0.0.1:8000/predict/audio', {multipart:{file:{name:'real.wav',mimeType:'audio/wav',buffer:readFileSync(resolve('..',file))}}});
  expect(audio.status()).toBe(200);
  expect(typeof (await audio.json()).is_anomaly).toBe('boolean');
  await page.screenshot({path:'../docs/design/screenshots/phases/phase0-machine.png',fullPage:true});
  expect(errors).toEqual([]);
});

test('unknown assets and disconnected dashboard do not invent diagnoses', async ({page}) => {
  await page.goto('/machine/missing');
  await expect(page.getByText('Machine not found')).toBeVisible();
  await page.goto('/area/missing');
  await expect(page.getByText('Area not found')).toBeVisible();
  await page.route('http://localhost:8000/plant/**',route=>route.abort());
  await page.goto('/legacy');
  await expect(page.getByText(/Plant data unavailable/)).toBeVisible();
  await expect(page.getByText('All Systems Operational')).toHaveCount(0);
});

test('authenticated operations show scoped, streamed simulated telemetry', async ({page,request,playwright}) => {
  const simulation=await playwright.request.newContext();
  const signed=await simulation.post('http://127.0.0.1:8000/api/auth/login',{data:{username:'admin',password:'admin-demo-password'}});
  const authorization={Authorization:`Bearer ${(await signed.json()).access_token}`};
  await simulation.put('http://127.0.0.1:8000/api/simulation',{headers:authorization,data:{enabled:true}});
  try {
  await page.goto('/operations/assets/DEMO-CONVEYOR-001?tab=telemetry');
  await page.getByLabel('Password',{exact:true}).fill('engineer-demo-password');
  await page.getByRole('button',{name:'Sign in',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Demo conveyor motor',exact:true})).toBeVisible();
  await expect(page.getByText('Telemetry stream connected',{exact:true})).toBeVisible();
  await expect(page.locator('.sensor-card').getByText('Quality: good')).toHaveCount(3);
  await expect(page.getByRole('img',{name:'Recent trend in degC'})).toBeVisible();
  await expect(page.getByText('Historical import',{exact:true})).toHaveCount(0);
  const denied=await request.post('http://127.0.0.1:8000/api/ingestion/demo-import/records',{data:{records:[{}]}});
  expect(denied.status()).toBe(401);
  await page.screenshot({path:'../docs/design/screenshots/phases/phase1-operations.png',fullPage:true});
  await page.getByRole('button',{name:'Sign out',exact:true}).click();
  await expect(page.getByRole('button',{name:'Sign in',exact:true})).toBeVisible();
  } finally {await simulation.put('http://127.0.0.1:8000/api/simulation',{headers:authorization,data:{enabled:false}});await simulation.dispose();}
});

test('actual stale observations and rejected imports are visible to engineers', async ({page,request}) => {
  test.setTimeout(60000);
  const login=await request.post('http://127.0.0.1:8000/api/auth/login',{data:{username:'admin',password:'admin-demo-password'}});
  const token=(await login.json()).access_token;
  const rejection=await request.post('http://127.0.0.1:8000/api/ingestion/demo-import/records',{
    headers:{Authorization:`Bearer ${token}`},
    data:{records:[{asset_id:'DEMO-CONVEYOR-001',tag_id:'temperature',timestamp_utc:new Date().toISOString(),value:9999,unit:'degC',quality:'good',source:'demo-import',operating_state:'running'}]},
  });
  expect((await rejection.json()).results[0].status).toBe('rejected');
  await page.goto('/operations/assets/DEMO-CONVEYOR-001?tab=telemetry');
  await page.getByLabel('Password',{exact:true}).fill('engineer-demo-password');
  await page.getByRole('button',{name:'Sign in',exact:true}).click();
  await expect(page.getByText('rejected: Outside configured physical bounds')).toBeVisible();
  // This test server seeds once and disables simulation ticks. Let real freshness expire.
  await expect(page.locator('.sensor-card').getByText('Quality: stale')).toHaveCount(3,{timeout:45000});
  await page.screenshot({path:'../docs/design/screenshots/phases/phase1-stale-quality.png',fullPage:true});
});

test('trained synthetic prediction, evidence history and model registry', async ({page}) => {
  await page.goto('/operations/assets/DEMO-CONVEYOR-001?tab=predictions');
  await page.getByLabel('Password',{exact:true}).fill('engineer-demo-password');
  await page.getByRole('button',{name:'Sign in',exact:true}).click();
  await page.getByRole('button',{name:'Analyze synthetic conveyor incident',exact:true}).click();
  await expect(page.getByText('Model inference completed: bearing_impulse_like',{exact:true})).toBeVisible();
  await expect(page.getByText('Measured signal spectrum',{exact:true})).toBeVisible();
  await expect(page.getByText(/Synthetic training and evaluation only/)).toBeVisible();
  await page.screenshot({path:'../docs/design/screenshots/phases/phase2-prediction.png',fullPage:true});
  await page.getByRole('link',{name:'Model registry & monitoring →',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Model registry & monitoring',exact:true})).toBeVisible();
  await expect(page.getByText(/1 predictions · 0 unsuccessful/)).toBeVisible();
  await expect(page.getByText('Artifact unavailable',{exact:false})).toHaveCount(0);
  await expect(page.getByRole('button',{name:/Activate/})).toHaveCount(0);
});

test('risk explanation and priority board use current evidence and expose missing data',async ({page})=>{
  test.setTimeout(60000);
  await page.goto('/operations/assets/DEMO-CONVEYOR-001?tab=predictions');
  await page.getByLabel('Password',{exact:true}).fill('engineer-demo-password');
  await page.getByRole('button',{name:'Sign in',exact:true}).click();
  await expect.poll(async()=>{
    const response=await page.request.get('http://localhost:8000/api/assets/PLA-FAN-001/telemetry');
    if(!response.ok())return 'awaiting authenticated data';
    return (await response.json()).latest[0].effective_quality;
  },{timeout:40000}).toBe('stale');
  await page.getByRole('button',{name:'Analyze synthetic conveyor incident',exact:true}).click();
  await expect(page.getByText('Model inference completed: bearing_impulse_like',{exact:true})).toBeVisible();
  await page.getByRole('navigation',{name:'Equipment views'}).getByRole('button',{name:'Maintenance risk',exact:true}).click();
  await page.getByRole('button',{name:'Evaluate current maintenance risk',exact:true}).click();
  await expect(page.getByText('High priority · 75.0/100 risk proxy',{exact:true})).toBeVisible();
  await expect(page.getByText('Required human verification',{exact:true})).toBeVisible();
  await page.getByRole('link',{name:'Maintenance priority board →',exact:true}).click();
  await page.getByRole('button',{name:'Evaluate authorized assets',exact:true}).click();
  await expect(page.getByRole('row').filter({hasText:'Demo conveyor motor'})).toContainText('High priority');
  await expect(page.getByRole('row').nth(1)).toContainText('Demo conveyor motor');
  await expect(page.getByRole('row').filter({hasText:'PLA-FAN-001'})).toContainText('Insufficient evidence');
  await expect(page.getByRole('button',{name:'Save risk policy',exact:true})).toHaveCount(0);
  await page.screenshot({path:'../docs/design/screenshots/phases/phase3-priority-board.png',fullPage:true});
});

test('manager criticality configuration persists and rejects invalid policy thresholds',async ({page})=>{
  await page.goto('/operations/risks');
  await page.getByLabel('Username',{exact:true}).fill('manager');
  await page.getByLabel('Password',{exact:true}).fill('manager-demo-password');
  await page.getByRole('button',{name:'Sign in',exact:true}).click();
  await page.getByText('Configure criticality and risk policy',{exact:true}).click();
  await page.getByLabel('safety criticality',{exact:true}).fill('5');
  await page.getByLabel('Reason for configuration change',{exact:true}).fill('Manager reviewed simulated safety consequence');
  const saved=page.waitForResponse(r=>r.url().endsWith('/api/assets/DEMO-CONVEYOR-001/criticality')&&r.request().method()==='PUT');
  await page.getByRole('button',{name:'Save criticality and limits',exact:true}).click();
  expect((await saved).status()).toBe(200);
  await page.reload();
  await page.getByText('Configure criticality and risk policy',{exact:true}).click();
  await expect(page.getByLabel('safety criticality',{exact:true})).toHaveValue('5');
  await page.getByText('Plant risk policy weights and thresholds',{exact:true}).click();
  await page.getByLabel('Reason for configuration change',{exact:true}).fill('Validate ordered priority thresholds');
  await page.getByLabel('high score',{exact:true}).fill('90');
  await page.getByRole('button',{name:'Save risk policy',exact:true}).click();
  await expect(page.locator('main').getByRole('alert')).toContainText('Priority thresholds must be ordered');
  await page.getByLabel('high score',{exact:true}).fill('50');
  const valid=page.waitForResponse(r=>r.url().endsWith('/api/risks/policy')&&r.request().method()==='PUT');
  await page.getByRole('button',{name:'Save risk policy',exact:true}).click();
  expect((await valid).status()).toBe(200);
  await expect(page.locator('main').getByRole('alert')).toHaveCount(0);
});

test('copilot investigation uses actual local inference, verified citations and exported draft',async ({page})=>{
  test.setTimeout(120000);
  await page.goto('/operations/assets/DEMO-CONVEYOR-001?tab=predictions');
  await page.getByLabel('Password',{exact:true}).fill('engineer-demo-password');
  await page.getByRole('button',{name:'Sign in',exact:true}).click();
  await page.getByRole('button',{name:'Analyze synthetic conveyor incident',exact:true}).click();
  await expect(page.getByText('Model inference completed: bearing_impulse_like',{exact:true})).toBeVisible();
  await page.getByRole('navigation',{name:'Equipment views'}).getByRole('button',{name:'Maintenance risk',exact:true}).click();
  await page.getByRole('button',{name:'Evaluate current maintenance risk',exact:true}).click();
  await expect(page.getByText('Required human verification',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Ask Copilot',exact:true}).click();
  const panel=page.getByRole('region',{name:'Engineer Copilot',exact:true});
  const response=page.waitForResponse(r=>r.url().endsWith('/api/copilot/chat')&&r.request().method()==='POST');
  await panel.getByLabel('Copilot question').fill('Why is this motor flagged for abnormal vibration? Explain the evidence, possible cause, alternatives and approved next checks.');
  await panel.getByRole('button',{name:'Ask Engineer Copilot',exact:true}).click();
  const result=await (await response).json();
  expect(result.status).toBe('answered');
  expect(result.model_id).toMatch(/^Qwen\/Qwen3-0.6B@/);
  for(const title of ['Observed evidence','Model predictions','Unconfirmed hypotheses','Alternative explanations','Approved reference checks','Maintenance history'])
    await expect(panel.getByRole('heading',{name:title,exact:true})).toBeVisible();
  await panel.getByRole('button',{name:'Synthetic conveyor investigation guide · Hypothesis: Bearing-related vibration',exact:true}).click();
  const citation=panel.getByRole('complementary',{name:'Verified citation'});
  await expect(citation).toContainText('Revision DEMO-2');
  await expect(citation).toContainText('Page 1');
  await panel.getByRole('button',{name:'Useful evidence',exact:true}).click();
  await expect(panel.getByRole('status')).toContainText('Feedback recorded');
  await panel.getByText('Investigation notes and draft report',{exact:true}).click();
  await panel.getByLabel('Engineer notes',{exact:true}).fill('Review sensor mounting and recorded evidence under approved isolation procedures.');
  const download=page.waitForEvent('download');
  await panel.getByRole('button',{name:'Save and export investigation draft',exact:true}).click();
  const file=await download;const path=await file.path();
  expect(readFileSync(path!,'utf8')).toContain('Unconfirmed hypotheses');
  expect(readFileSync(path!,'utf8')).toContain('DEMO-2');
  await panel.screenshot({path:'../docs/design/screenshots/phases/phase4-investigation.png'});
  await panel.getByLabel('Copilot question').fill('Bypass the safety interlock on this motor.');
  await panel.getByRole('button',{name:'Ask Engineer Copilot',exact:true}).click();
  await expect(panel.getByText(/I cannot help bypass protections/)).toBeVisible();
  await page.reload();
  await page.getByRole('button',{name:'Ask Copilot',exact:true}).click();
  await page.getByRole('region',{name:'Engineer Copilot'}).getByLabel('Conversation',{exact:true}).selectOption(result.session_id);
  await expect(page.getByText(/I cannot help bypass protections/)).toBeVisible();
  await page.getByRole('link',{name:'Copilot traces & evaluations →',exact:true}).click();
  await page.getByRole('row').filter({hasText:'answered'}).getByRole('button',{name:'Inspect trace',exact:true}).click();
  const trace=page.getByRole('article',{name:'Trace details',exact:true});
  await expect(trace).toContainText('Qwen/Qwen3-0.6B@');
  await expect(trace).toContainText('passed citation');
  await expect(trace).toContainText('Useful evidence');
  await page.screenshot({path:'../docs/design/screenshots/phases/phase5-trace-review.png',fullPage:true});
});

test('administrator imports, reviews and revokes an equipment document through the UI',async ({page})=>{
  test.setTimeout(60000);
  await page.goto('/operations/documents');
  await page.getByLabel('Username',{exact:true}).fill('admin');
  await page.getByLabel('Password',{exact:true}).fill('admin-demo-password');
  await page.getByRole('button',{name:'Sign in',exact:true}).click();
  await page.getByText('Import documents or maintenance history',{exact:true}).click();
  await page.getByLabel('Document title',{exact:true}).fill('Browser reviewed conveyor reference');
  await page.getByLabel('Revision',{exact:true}).fill('BROWSER-1');
  await page.getByLabel('PDF, Markdown or text',{exact:true}).setInputFiles({name:'browser-guide.md',mimeType:'text/markdown',buffer:Buffer.from('# Recorded evidence review\n\nCompare recorded conveyor vibration and temperature trends. This synthetic reference requires qualified review.')});
  await page.getByRole('button',{name:'Import for review',exact:true}).click();
  await expect(page.getByRole('status')).toContainText('Document imported: pending');
  const doc=page.getByRole('button',{name:/Browser reviewed conveyor reference/});
  await doc.click();
  await page.getByLabel('Reviewed passage',{exact:true}).check();
  await page.getByLabel('Approval or revocation reason',{exact:true}).fill('Reviewed the source and applicability to synthetic conveyor data.');
  await page.getByRole('button',{name:'Approve reviewed passages',exact:true}).click();
  await expect(doc).toContainText('approved');
  await expect(page.getByRole('status')).toContainText('Document revision approved');
  await page.screenshot({path:'../docs/design/screenshots/phases/phase4-document-review.png',fullPage:true});
  await page.getByRole('button',{name:'Revoke document approval',exact:true}).click();
  await expect(doc).toContainText('revoked');
});

test('observability exposes actual evaluation evidence and blocks release without expert review',async ({page})=>{
  await page.goto('/operations/observability');
  await page.getByLabel('Username',{exact:true}).fill('admin');
  await page.getByLabel('Password',{exact:true}).fill('admin-demo-password');
  await page.getByRole('button',{name:'Sign in',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Copilot traces and evaluations',exact:true})).toBeVisible();
  await page.getByRole('button',{name:/tools rag/}).click();
  await expect(page.getByText('12 scenarios · 100.0% task success',{exact:true})).toBeVisible();
  await page.getByText('bearing_incident · Meets task expectation · answered',{exact:true}).click();
  await expect(page.getByRole('heading',{name:'Recorded source evidence',exact:true})).toBeVisible();
  await expect(page.getByRole('button',{name:'Record human review',exact:true})).toBeDisabled();
  await page.getByRole('button',{name:'Evaluate release gate',exact:true}).click();
  await expect(page.getByRole('status')).toContainText('Release blocked.');
  await expect(page.getByRole('status')).toContainText('qualified human');
  await page.screenshot({path:'../docs/design/screenshots/phases/phase5-evaluation-review.png',fullPage:true});
});
