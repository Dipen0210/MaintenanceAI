import {test,expect,Page} from '@playwright/test';

async function login(page:Page,username='engineer'){
 await page.goto('/operations');await page.getByLabel('Username',{exact:true}).fill(username);
 await page.getByLabel('Password',{exact:true}).fill(`${username}-demo-password`);
 await page.getByRole('button',{name:'Sign in',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Operations dashboard',exact:true})).toBeVisible();
}

test('simulation switch produces visible machine readings and pauses persistently',async({page})=>{
 await page.setViewportSize({width:1440,height:1000});await login(page);
 const control=page.getByRole('switch',{name:'Run simulation',exact:true});await expect(control).toHaveAttribute('aria-checked','false');
 await expect.poll(()=>control.evaluate(el=>{const box=el.getBoundingClientRect(),header=el.closest('header')!.getBoundingClientRect();return box.top>=header.top&&box.bottom<=header.bottom;})).toBeTruthy();
 await control.click();await expect(control).toHaveAttribute('aria-checked','true');
 const overview=page.getByRole('region',{name:'Machine condition overview'});
 await expect(overview.getByText('Warning readings',{exact:true})).toBeVisible();
 await expect(overview.getByText('Critical readings',{exact:true})).toBeVisible();
 await expect(overview.getByText('Within limits',{exact:true}).first()).toBeVisible();
 await page.screenshot({path:'../docs/design/screenshots/field-work/field-mixed-dashboard.png',fullPage:true});
 const scenario=page.getByLabel('Demo condition scenario',{exact:true});
 await scenario.selectOption('critical');await expect(overview.getByRole('link').filter({hasText:'Demo conveyor motor'})).toContainText('Critical readings');
 await scenario.selectOption('normal');await expect(overview.getByText('Critical readings',{exact:true})).toHaveCount(0);await expect(overview.getByText('Warning readings',{exact:true})).toHaveCount(0);
 await scenario.selectOption('mixed');await expect(overview.getByText('Critical readings',{exact:true})).toBeVisible();
 await page.setViewportSize({width:390,height:844});
 for(const element of [control,scenario])await expect.poll(()=>element.evaluate(el=>{const box=el.getBoundingClientRect(),header=el.closest('header')!.getBoundingClientRect();return box.top>=header.top&&box.bottom<=header.bottom&&box.right<=window.innerWidth;})).toBeTruthy();
 await page.screenshot({path:'../docs/design/screenshots/field-work/field-simulation-mobile.png',animations:'disabled'});
 await page.setViewportSize({width:1440,height:1000});
 await page.goto('/operations/assets/DEMO-CONVEYOR-001');
 const map=page.getByRole('region',{name:'Machine sensor map'});await expect(map.getByRole('img',{name:/Illustrative conveyor/})).toBeVisible();
 await expect(map.getByRole('button',{name:/Sensor temperature:/})).toBeVisible();
 await map.getByRole('button',{name:/Sensor temperature:/}).click();await expect(map.locator('.map-reading')).toContainText('degC');
 await expect(map.locator('.sensor-map-detail').getByText('Quality: good',{exact:true})).toBeVisible();
 await page.screenshot({path:'../docs/design/screenshots/machines/machine-conveyor-illustration.png',fullPage:true});
 await control.click();await expect(control).toHaveAttribute('aria-checked','false');await page.reload();await expect(control).toHaveAttribute('aria-checked','false');
 await page.goto('/operations/assets/new');await expect(page.getByRole('heading',{name:'Administrator access required',exact:true})).toBeVisible();
});

test('register a machine, receive scoped gateway data, and configure its visual sensor map',async({page})=>{
 test.setTimeout(90000);await page.setViewportSize({width:1440,height:1100});await login(page,'admin');
 await page.getByRole('link',{name:'Add machine',exact:true}).click();
 await page.getByLabel('Machine ID',{exact:true}).fill('UI-FACTORY-PUMP');await page.getByLabel('Machine name',{exact:true}).fill('Cooling water pump');
 await page.getByLabel('Plant area',{exact:true}).fill('Utility building');await page.getByRole('button',{name:'Continue to sensors',exact:false}).click();
 for(const [label,value] of [['Sensor tag','temperature'],['Unit','degC'],['Physical minimum','-20'],['Physical maximum','150'],['Warning threshold','70'],['Critical threshold','90']])await page.getByLabel(`Sensor 1 ${label}`,{exact:true}).fill(value);
 await page.getByRole('button',{name:'Add another sensor',exact:true}).click();
 for(const [label,value] of [['Sensor tag','vibration_rms'],['Unit','mm/s'],['Physical minimum','0'],['Physical maximum','50'],['Warning threshold','4.5'],['Critical threshold','9']])await page.getByLabel(`Sensor 2 ${label}`,{exact:true}).fill(value);
 await page.screenshot({path:'../docs/design/screenshots/machines/machine-onboarding-sensors.png',fullPage:true});
 await page.getByRole('button',{name:'Review setup',exact:false}).click();
 await page.getByLabel('Commissioning notes and limit-review reason',{exact:true}).fill('Browser test only: synthetic sensor contracts reviewed for this local gateway demonstration.');
 await page.getByRole('checkbox').check();await page.getByRole('button',{name:'Register machine & create gateway key',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Machine registered',exact:true})).toBeVisible();
 await expect(page.getByRole('heading',{name:'Waiting for sensor readings',exact:true})).toBeVisible();
 const endpoint=await page.getByLabel('Telemetry endpoint',{exact:true}).inputValue();const token=await page.getByLabel('Gateway key',{exact:true}).inputValue();const source=endpoint.split('/').at(-2)!;
 const before=await page.request.get('http://localhost:8000/api/assets/UI-FACTORY-PUMP/telemetry');expect((await before.json()).history).toEqual([]);
 const records=[{tag_id:'temperature',unit:'degC',value:46.25},{tag_id:'vibration_rms',unit:'mm/s',value:1.2}].map(r=>({...r,asset_id:'UI-FACTORY-PUMP',source,timestamp_utc:new Date().toISOString(),quality:'good',operating_state:'running'}));
 const sent=await page.request.post(endpoint,{headers:{Authorization:`Bearer ${token}`},data:{records}});expect(sent.ok()).toBeTruthy();expect((await sent.json()).results.every((r:{status:string})=>r.status==='accepted')).toBeTruthy();
 await expect(page.getByRole('heading',{name:'Sensor readings received',exact:true})).toBeVisible({timeout:10000});
 await page.getByRole('button',{name:'Open machine monitoring',exact:true}).click();
 const map=page.getByRole('region',{name:'Machine sensor map'});
 await expect(map.getByRole('img',{name:'Illustrative pump assembly',exact:true})).toBeVisible();
 await map.getByRole('button',{name:/Sensor temperature:/}).click();await expect(map.locator('.map-reading')).toContainText('46.25');
 await expect(page.getByText('Within limits',{exact:true})).toBeVisible();
 await page.screenshot({path:'../docs/design/screenshots/machines/machine-pump-illustration.png',fullPage:true,animations:'disabled'});
 await map.getByRole('button',{name:'Edit sensor locations',exact:true}).click();
 await map.getByLabel('Sensor location label',{exact:true}).fill('Drive-end bearing housing');await map.getByLabel('Horizontal sensor position',{exact:true}).fill('42');await map.getByLabel('Vertical sensor position',{exact:true}).fill('38');
 await map.getByRole('button',{name:'Save sensor locations',exact:true}).click();await expect(map.getByRole('status')).toContainText('Sensor locations saved');
 await page.reload();await expect(map.getByRole('button',{name:'Sensor temperature: Drive-end bearing housing',exact:true})).toBeVisible();
 await page.setViewportSize({width:390,height:844});await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();await expect(page.getByRole('button',{name:'Open navigation',exact:true})).toBeVisible();await map.screenshot({path:'../docs/design/screenshots/machines/machine-pump-mobile.png',animations:'disabled'});
 // A tiny generated test PNG verifies authenticated photo storage; it is not plant evidence.
 await map.getByLabel('Upload machine photo',{exact:true}).setInputFiles({name:'test-machine.png',mimeType:'image/png',buffer:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=','base64')});
 await expect(map.getByRole('img',{name:'Photo of Cooling water pump',exact:true})).toBeVisible();
 await expect(map.locator('image')).toHaveAttribute('href',/^blob:/);
 await page.reload();await expect(map.getByRole('img',{name:'Photo of Cooling water pump',exact:true})).toBeVisible();
 await page.goto('/operations/data');const gateway=page.locator('.source-card').filter({hasText:'Cooling water pump gateway'});await gateway.getByText('Gateway setup & access',{exact:true}).click();
 await gateway.getByText('Manage gateway access',{exact:true}).click();await gateway.getByRole('button',{name:'Revoke gateway access',exact:true}).click();
 await expect(gateway.getByRole('status')).toContainText('Gateway access revoked');
 const rejected=await page.request.post(endpoint,{headers:{Authorization:`Bearer ${token}`},data:{records}});expect(rejected.status()).toBe(401);
 // Registration is private by default. Explicitly share this fixture with the
 // plant manager before later tests configure resources for the whole plant.
 const identities=await page.request.get('http://localhost:8000/api/admin/identities');expect(identities.ok()).toBeTruthy();
 const manager=(await identities.json()).find((identity:{username:string})=>identity.username==='manager');expect(manager).toBeTruthy();
 expect(manager.asset_ids).not.toContain('UI-FACTORY-PUMP');
 const shared=await page.request.put(`http://localhost:8000/api/admin/identities/${manager.id}/access`,{
  headers:{Origin:'http://localhost:3000'},data:{role:manager.role,enabled:manager.enabled,
   asset_ids:[...manager.asset_ids,'UI-FACTORY-PUMP'],source_ids:[...manager.source_ids,source],document_ids:manager.document_ids,
   reason:'Explicitly share the browser fixture machine with the plant maintenance manager.'},
 });expect(shared.ok()).toBeTruthy();expect((await shared.json()).asset_ids).toContain('UI-FACTORY-PUMP');
});
