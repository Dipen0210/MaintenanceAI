import {test,expect} from '@playwright/test';

test('administrator creates a scoped identity and revokes its live session',async({page,browser})=>{
 test.setTimeout(90000);
 await page.goto('/operations');
 await page.getByLabel('Username',{exact:true}).fill('admin');
 await page.getByLabel('Password',{exact:true}).fill('admin-demo-password');
 await page.getByRole('button',{name:'Sign in',exact:true}).click();
 await page.getByRole('navigation',{name:'Primary navigation'}).getByRole('link',{name:'Administration',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Authorized users and roles',exact:true})).toBeVisible();
 await page.getByLabel('New username',{exact:true}).fill('browser-scoped');
 await page.getByLabel('Initial password',{exact:true}).fill('browser-test-password-only');
 await page.getByLabel('Grant asset DEMO-CONVEYOR-001',{exact:true}).check();
 await page.getByLabel('Grant source demo-simulator',{exact:true}).check();
 await page.getByLabel('Grant source demo-import',{exact:true}).check();
 await page.getByLabel('Access change reason',{exact:true}).fill('Browser regression test: grant one equipment context to a local engineer.');
 await page.getByRole('button',{name:'Create account',exact:true}).click();
 await expect(page.getByRole('status')).toContainText('saved and audited');
 const identity=page.getByLabel('Managed account',{exact:true});
 const option=identity.locator('option').filter({hasText:'browser-scoped'});
 await expect(option).toHaveCount(1);const id=await option.getAttribute('value');
 const context=await browser.newContext();const engineer=await context.newPage();
 try{
  await engineer.goto('http://localhost:3000/operations');
  await engineer.getByLabel('Username',{exact:true}).fill('browser-scoped');
  await engineer.getByLabel('Password',{exact:true}).fill('browser-test-password-only');
  await engineer.getByRole('button',{name:'Sign in',exact:true}).click();
  await engineer.getByRole('navigation',{name:'Primary navigation'}).getByRole('link',{name:'Assets',exact:true}).click();
  await expect(engineer.getByText('Authorized assets (1)',{exact:true})).toBeVisible();
  await expect(engineer.getByRole('link',{name:'Administration',exact:true})).toHaveCount(0);
  await identity.selectOption(id!);
  await page.getByLabel('Account enabled',{exact:true}).uncheck();
  await page.getByLabel('Access change reason',{exact:true}).fill('Disable the temporary browser-test account and revoke its current session.');
  await page.getByRole('button',{name:'Save access and revoke sessions',exact:true}).click();
  await expect(page.getByRole('status')).toContainText('saved and audited');
  await engineer.reload();await expect(engineer.getByRole('button',{name:'Sign in',exact:true})).toBeVisible();
 }finally{await context.close();}
 await page.getByRole('button',{name:'Asset registry',exact:true}).click();
 await page.getByLabel('New file source name',{exact:true}).fill('Browser historical archive');
 await page.getByLabel('Source registration reason',{exact:true}).fill('Register a read-only local file archive for the browser verification.');
 await page.getByRole('button',{name:'Register read-only file source',exact:true}).click();
 await expect(page.getByText('Browser historical archive · file · awaiting import · read only',{exact:true})).toBeVisible();
 await page.screenshot({path:'../docs/design/screenshots/phases/phase7-administration.png',fullPage:true});
 await expect(page.locator('main').getByRole('alert')).toHaveCount(0);
});
