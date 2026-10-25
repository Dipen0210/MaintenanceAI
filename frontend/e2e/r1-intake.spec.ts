import {test,expect,Page} from '@playwright/test';
async function login(page:Page,name='engineer'){
 await page.goto('/operations');
 if(await page.getByRole('button',{name:'Sign out',exact:true}).isVisible())await page.getByRole('button',{name:'Sign out',exact:true}).click();
 await page.getByLabel('Username',{exact:true}).fill(name);await page.getByLabel('Password',{exact:true}).fill(name+'-demo-password');await page.getByRole('button',{name:'Sign in',exact:true}).click();await expect(page.getByRole('heading',{name:'Operations dashboard',exact:true})).toBeVisible();
}

test('field report becomes an assigned investigation and closes without claiming a repair',async({page})=>{
 test.setTimeout(90000);await page.setViewportSize({width:1440,height:1000});await login(page);
 await page.getByRole('navigation',{name:'Primary navigation'}).getByRole('link',{name:'Issues',exact:true}).click();
 await page.getByLabel('Issue machine').selectOption('DEMO-CONVEYOR-001');await page.getByLabel('Issue title',{exact:true}).fill('R1 field observation: rattling housing');await page.getByLabel('Observed symptom').fill('Rattling observed near the drive end during the shift walk. Human observation only.');await page.getByLabel('Reported urgency').selectOption('high');
 await page.getByRole('button',{name:'Submit issue',exact:true}).click();const detail=page.getByRole('region',{name:'Issue details'});await expect(detail.getByRole('heading',{name:'R1 field observation: rattling housing',exact:true})).toBeVisible();
 await expect(detail.getByRole('button',{name:'Close investigation',exact:true})).toHaveCount(0);
 await page.screenshot({animations:'disabled',path:'../docs/design/screenshots/r1-intake/issue-reported.png',fullPage:true});
 await login(page,'manager');await page.goto('/operations/issues');await page.getByRole('button').filter({hasText:'R1 field observation: rattling housing'}).click();
 await detail.getByLabel('Assign engineer').selectOption('engineer');await detail.getByLabel('Investigation update').fill('Assign the field inspection to the engineer for investigation.');await detail.getByRole('button',{name:'Assign investigation',exact:true}).click();await expect(detail).toContainText('version 2');
 await login(page);await page.getByRole('navigation',{name:'Primary navigation'}).getByRole('link',{name:'My Work',exact:true}).click();await page.getByRole('link').filter({hasText:'R1 field observation: rattling housing'}).click();
 await detail.getByLabel('Investigation update').fill('Starting an observation-only inspection under the local site process.');await detail.getByRole('button',{name:'Start investigation',exact:true}).click();await expect(detail).toContainText('investigating');
 await detail.getByRole('button',{name:'Ask Copilot about machine',exact:true}).click();const dialog=page.getByRole('dialog',{name:'Engineer Copilot assistant'});await expect(dialog.getByLabel('Copilot equipment')).toHaveValue('DEMO-CONVEYOR-001');await page.keyboard.press('Escape');
 // Real browser multipart upload and authorized binary download, using a tiny test PNG.
 await detail.getByLabel('Issue photo',{exact:true}).setInputFiles({name:'field.png',mimeType:'image/png',buffer:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=','base64')});await detail.getByRole('button',{name:'Attach photo',exact:true}).click();await expect(detail.getByRole('button',{name:'Download photo 1',exact:true})).toBeVisible();const download=page.waitForEvent('download');await detail.getByRole('button',{name:'Download photo 1',exact:true}).click();expect((await download).suggestedFilename()).toMatch(/\.jpg$/);
 await page.setViewportSize({width:390,height:844});await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();await expect(page.locator('.workspace-sidebar')).toHaveAttribute('aria-hidden','true');await page.evaluate(()=>scrollTo(0,0));await page.screenshot({animations:'disabled',path:'../docs/design/screenshots/r1-intake/issue-mobile.png'});await page.setViewportSize({width:1440,height:1000});
 await login(page,'manager');await page.goto('/operations/issues');await page.getByRole('button').filter({hasText:'R1 field observation: rattling housing'}).click();await detail.getByLabel('Investigation update').fill('Investigation complete; physical repair remains a separately reviewed work order.');await detail.getByRole('button',{name:'Close investigation',exact:true}).click();await expect(detail).toContainText('closed');
 await login(page);await page.goto('/operations/my-work');await expect(page.getByRole('link').filter({hasText:'R1 field observation: rattling housing'})).toHaveCount(0);
});

test('switching workspace resets machine context, persists the view, and leaves demo control separate',async({page})=>{
 test.setTimeout(60000);await login(page);await page.getByRole('button',{name:'Ask Copilot',exact:true}).click();const dialog=page.getByRole('dialog',{name:'Engineer Copilot assistant'});await dialog.getByLabel('Copilot question').fill('Old demo context');await page.keyboard.press('Escape');
 await page.getByLabel('Workspace',{exact:true}).selectOption('plant');await expect(page.getByText('Plant data · read only',{exact:true})).toBeVisible();await expect(page.locator('.machine-condition-card')).toHaveCount(0);await expect(page.getByRole('switch')).toHaveCount(0);
 await page.getByRole('button',{name:'Ask Copilot',exact:true}).click();await expect(dialog).toContainText('No accessible equipment');await expect(dialog.getByLabel('Copilot question')).toHaveCount(0);await page.keyboard.press('Escape');
 await page.reload();await expect(page.getByLabel('Workspace',{exact:true})).toHaveValue('plant');await page.goto('/operations/assets/DEMO-CONVEYOR-001');await expect(page.getByRole('heading',{name:'Demo conveyor motor',exact:true})).toHaveCount(0);
 await page.getByLabel('Workspace',{exact:true}).selectOption('demo');await expect(page.getByRole('heading',{name:'Operations dashboard',exact:true})).toBeVisible();await expect(page.locator('.machine-condition-card')).not.toHaveCount(0);
 await page.getByRole('button',{name:'Ask Copilot',exact:true}).click();await expect(dialog.getByLabel('Copilot question')).toHaveValue('Summarize the current equipment evidence, any concerns, missing information and approved next checks.');
});
