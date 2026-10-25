import {test,expect,Page} from '@playwright/test';
async function login(page:Page){await page.goto('/');await page.getByLabel('Password',{exact:true}).fill('engineer-demo-password');await page.getByRole('button',{name:'Sign in',exact:true}).click();await expect(page.getByRole('heading',{name:'Operations dashboard',exact:true})).toBeVisible();}

test('dashboard navigation is organized and Copilot follows the selected equipment across pages',async({page})=>{
 test.setTimeout(90000);await page.setViewportSize({width:1440,height:1000});const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await login(page);await expect(page).toHaveURL(/\/operations$/);const nav=page.getByRole('navigation',{name:'Primary navigation'});
 await expect(nav.getByRole('link')).toHaveText(['Dashboard','My Work','Issues','Assets','Scan machine','Priorities','Maintenance','Preventive plans','Investigations','Documents','Data sources','Model registry','AI evaluations']);
 await expect(page.getByRole('button',{name:'Analyze synthetic conveyor incident',exact:true})).toHaveCount(0);
 await page.screenshot({animations:'disabled',path:'../docs/design/screenshots/workspace/workspace-dashboard.png',fullPage:true});
 await nav.getByRole('link',{name:'Assets',exact:true}).click();await page.getByLabel('Search assets').fill('PLA-PUMP-001');await expect(page.getByRole('row')).toHaveCount(2);
 await page.getByRole('link',{name:'Open PLA-PUMP-001',exact:true}).click();await expect(page.getByRole('heading',{name:'PLA-PUMP-001',exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Ask Copilot',exact:true}).click();const dialog=page.getByRole('dialog',{name:'Engineer Copilot assistant'});
 await expect(dialog.getByLabel('Copilot equipment')).toHaveValue('PLA-PUMP-001');await dialog.getByLabel('Copilot question').fill('Explain the source quality for this pump.');await page.keyboard.press('Escape');await expect(dialog).toBeHidden();await expect(page.getByRole('button',{name:'Ask Copilot',exact:true})).toBeFocused();
 for(const name of ['Priorities','Maintenance','Documents','Data sources','Model registry','AI evaluations']){
  await nav.getByRole('link',{name,exact:true}).click();await page.getByRole('button',{name:'Ask Copilot',exact:true}).click();await expect(dialog.getByLabel('Copilot equipment')).toHaveValue('PLA-PUMP-001');await expect(dialog.getByLabel('Copilot question')).toHaveValue('Explain the source quality for this pump.');await page.keyboard.press('Escape');
 }
 await nav.getByRole('link',{name:'Investigations',exact:true}).click();await expect(page.getByLabel('Investigation equipment')).toHaveValue('PLA-PUMP-001');
 await page.getByLabel('Investigation equipment').selectOption('DEMO-CONVEYOR-001');await page.getByRole('button',{name:'Ask Copilot',exact:true}).click();await expect(dialog.getByLabel('Copilot equipment')).toHaveValue('DEMO-CONVEYOR-001');await page.screenshot({animations:'disabled',path:'../docs/design/screenshots/workspace/workspace-copilot.png'});await page.keyboard.press('Escape');
 await nav.getByRole('link',{name:'Assets',exact:true}).click();await page.screenshot({animations:'disabled',path:'../docs/design/screenshots/workspace/workspace-assets.png',fullPage:true});
 await page.getByRole('link',{name:'Open Demo conveyor motor',exact:true}).click();await expect(page.getByRole('heading',{name:'Upcoming approved maintenance',exact:true})).toBeVisible();await page.screenshot({animations:'disabled',path:'../docs/design/screenshots/workspace/workspace-equipment.png',fullPage:true});
 await nav.getByRole('link',{name:'Maintenance',exact:true}).click();await expect(page.getByRole('heading',{name:'Draft maintenance candidate',exact:true})).toBeHidden();await page.getByRole('button',{name:'New work order',exact:true}).click();await expect(page.getByRole('heading',{name:'Draft maintenance candidate',exact:true})).toBeVisible();await expect(page.getByRole('heading',{name:'Maintenance candidates',exact:true})).toBeHidden();await page.screenshot({animations:'disabled',path:'../docs/design/screenshots/workspace/workspace-planning.png',fullPage:true});
 expect(errors).toEqual([]);
});

test('mobile navigation, asset views and Copilot keyboard controls fit a small screen',async({page})=>{
 await page.setViewportSize({width:390,height:844});await login(page);
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();await page.screenshot({animations:'disabled',path:'../docs/design/screenshots/workspace/workspace-mobile.png',fullPage:true});
 await page.getByRole('button',{name:'Open navigation',exact:true}).click();await page.getByRole('navigation',{name:'Primary navigation'}).getByRole('link',{name:'Assets',exact:true}).click();
 await page.getByRole('link',{name:'Open Demo conveyor motor',exact:true}).click();await expect(page.getByRole('heading',{name:'Demo conveyor motor',exact:true})).toBeVisible();await expect(page.getByRole('heading',{name:'Upcoming approved maintenance',exact:true})).toBeVisible();await page.screenshot({animations:'disabled',path:'../docs/design/screenshots/workspace/workspace-mobile-equipment.png',fullPage:true});await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
 await page.getByRole('button',{name:'Ask Copilot',exact:true}).click();const dialog=page.getByRole('dialog',{name:'Engineer Copilot assistant'});await expect(dialog).toBeVisible();
 await expect(dialog.getByRole('button',{name:'Close Copilot',exact:true})).toBeFocused();await page.keyboard.press('Shift+Tab');expect(await dialog.evaluate(el=>el.contains(document.activeElement))).toBeTruthy();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();await page.screenshot({animations:'disabled',path:'../docs/design/screenshots/workspace/workspace-mobile-copilot.png'});await page.keyboard.press('Escape');await expect(dialog).toBeHidden();
});

test('dashboard failure shows unavailable summaries instead of false zero counts',async({page})=>{
 await page.route('**/api/risks',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'Risk service unavailable'})}));
 await login(page);await expect(page.locator('main').getByRole('alert')).toContainText('Dashboard summaries are unavailable');await expect(page.getByText('Priorities unavailable',{exact:true})).toBeVisible();await expect(page.locator('.metric-card').filter({hasText:'Needs attention'}).locator('strong')).toHaveText('—');
});
