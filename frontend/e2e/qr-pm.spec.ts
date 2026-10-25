import {test,expect,Page} from '@playwright/test';
async function signIn(page:Page,name='engineer'){
 await page.getByLabel('Username',{exact:true}).fill(name);await page.getByLabel('Password',{exact:true}).fill(name+'-demo-password');await page.getByRole('button',{name:'Sign in',exact:true}).click();await expect(page.getByRole('button',{name:'Sign out',exact:true})).toBeVisible();
}
test('QR link survives login; labels download and camera denial permits manual lookup',async({page})=>{
 await page.goto('/operations/scan?asset=DEMO-CONVEYOR-001');await signIn(page);
 await expect(page).toHaveURL(/\/assets\/DEMO-CONVEYOR-001$/);await expect(page.getByRole('heading',{name:'Demo conveyor motor',exact:true})).toBeVisible();
 await page.getByText('Machine QR label',{exact:true}).click();await expect(page.getByRole('img',{name:'QR access label for Demo conveyor motor'})).toBeVisible();
 const download=page.waitForEvent('download');await page.getByRole('link',{name:'Download QR label'}).click();expect((await download).suggestedFilename()).toBe('machine-DEMO-CONVEYOR-001.png');
 await page.getByText('Machine QR label',{exact:true}).click();await page.getByText('Machine QR label',{exact:true}).click();await expect(page.getByRole('img',{name:'QR access label for Demo conveyor motor'})).toBeVisible();
 await page.addInitScript(()=>{Object.defineProperty(window,'BarcodeDetector',{value:class{async detect(){return [];}}});Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async()=>{throw new DOMException('Denied','NotAllowedError');}});});
 await page.goto('/operations/scan');await page.getByRole('button',{name:'Start camera'}).click();await expect(page.locator('main').getByRole('alert')).toContainText('Camera could not be opened');
 await page.getByLabel('Machine ID or QR link').fill('DEMO-CONVEYOR-001');await page.getByRole('button',{name:'Open machine',exact:true}).click();await expect(page).toHaveURL(/\/assets\/DEMO-CONVEYOR-001$/);
 await page.getByLabel('Workspace',{exact:true}).selectOption('plant');await expect(page.getByText('Plant data · read only',{exact:true})).toBeVisible();await page.goto('/operations/scan?asset=DEMO-CONVEYOR-001');await expect(page).toHaveURL(/\/assets\/DEMO-CONVEYOR-001$/);await expect(page.getByLabel('Workspace',{exact:true})).toHaveValue('demo');
});

test('manager creates recurring work, avoids duplicate generation, pauses and revises the plan',async({page})=>{
 test.setTimeout(120000);await page.goto('/operations/preventive');await signIn(page,'manager');await page.getByRole('button',{name:'New preventive plan',exact:true}).click();
 await page.getByLabel('Plan machine').selectOption('DEMO-CONVEYOR-001');await page.getByLabel('Plan title').fill('Browser weekly mounting inspection');await page.getByLabel('Preventive task scope').fill('Inspect sensor mounting using the approved procedure after site clearance.');await page.getByLabel('First due (local time)').fill(new Date(Date.now()-120000).toISOString().slice(0,16));await page.getByLabel('Calendar timezone').fill('UTC');await page.getByLabel('Grace period (hours)').fill('0');await page.locator('.procedure-options input[type=checkbox]').first().check();await page.getByLabel('Plan change reason').fill('Configure a recurring synthetic field inspection.');
 await page.setViewportSize({width:390,height:844});await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:'../docs/design/qr-pm/plan-form-mobile.png',fullPage:true});await page.setViewportSize({width:1440,height:1000});
 await page.getByRole('button',{name:'Save preventive plan',exact:true}).click();await page.getByRole('button').filter({hasText:'Browser weekly mounting inspection'}).click();const detail=page.getByRole('region',{name:'Preventive plan details'});
 // Verify the real lifespan worker, not just the manual generation endpoint.
 await expect(detail.locator('.pm-occurrences a')).toHaveCount(1,{timeout:75000});await detail.getByRole('button',{name:'Generate due work',exact:true}).click();await expect(page.getByRole('status')).toContainText('0 draft work orders');await expect(detail).toContainText('Overdue');
 await detail.getByLabel('Pause or resume reason').fill('Pause the plan during scheduled shutdown.');await detail.getByRole('button',{name:'Pause plan',exact:true}).click();await expect(detail.getByRole('button',{name:'Generate due work',exact:true})).toBeDisabled();await detail.getByRole('button',{name:'Revise plan',exact:true}).click();await page.getByLabel('Plan title').fill('Revised browser mounting inspection');await page.getByLabel('Plan change reason').fill('Clarify the inspection title for the next revision.');await page.getByRole('button',{name:'Save preventive plan',exact:true}).click();await expect(detail).toContainText('version 3');await expect(detail.locator('.pm-occurrences a')).toHaveCount(1);
 await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:'../docs/design/qr-pm/plan-details-desktop.png',fullPage:true});await detail.locator('.pm-occurrences a').click();await page.getByText('Review risk and procedure evidence',{exact:true}).click();await expect(page.getByText('Calendar-driven preventive work',{exact:false})).toBeVisible();
 await page.getByRole('button',{name:'Sign out',exact:true}).click();await signIn(page);await page.goto('/operations/preventive');await expect(page.getByRole('button',{name:'New preventive plan',exact:true})).toHaveCount(0);await page.getByRole('button').filter({hasText:'Revised browser mounting inspection'}).click();await expect(detail.locator('.pm-occurrences a')).toHaveCount(1);await expect(detail.getByRole('button',{name:'Revise plan',exact:true})).toHaveCount(0);
});

test('workspace panels fit phone, tablet and desktop without document overflow',async({page})=>{
 test.setTimeout(120000);await page.goto('/operations');await signIn(page,'admin');await expect(page.getByRole('heading',{name:'Operations dashboard',exact:true})).toBeVisible();
 for(const width of [390,768,1024,1440]){await page.setViewportSize({width,height:1000});for(const path of ['', '/assets','/assets/DEMO-CONVEYOR-001','/scheduling?view=resources','/scheduling?view=new','/issues','/my-work','/preventive','/scan','/admin','/data','/observability','/assets/new']){
  await page.goto('/operations'+path);await expect(page.locator('#workspace-content main')).toBeVisible();await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth),{message:`${path} at ${width}px`}).toBeLessThanOrEqual(width);
 }}
});
