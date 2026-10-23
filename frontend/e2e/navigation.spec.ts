import {test,expect,Page} from '@playwright/test';

async function login(page:Page,username='engineer'){
 await page.goto('/operations');
 await page.getByLabel('Username',{exact:true}).fill(username);
 await page.getByLabel('Password',{exact:true}).fill(`${username}-demo-password`);
 await page.getByRole('button',{name:'Sign in',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Operations dashboard',exact:true})).toBeVisible();
}

test('maintenance views survive refresh and history without losing draft input or link context',async({page})=>{
 await login(page);
 // This context marker tests navigation only; no investigation or work is created.
 await page.goto('/operations/scheduling?asset=PLA-PUMP-001&investigation=navigation-context-only');
 const views=page.getByRole('navigation',{name:'Maintenance views'});
 const draft=page.getByRole('heading',{name:'Draft maintenance candidate',exact:true});
 await expect(draft).toBeVisible();
 await page.getByLabel('Work-order title',{exact:true}).fill('Keep this unsaved inspection draft');
 await views.getByRole('button',{name:'Resources',exact:true}).click();
 await expect(page).toHaveURL(/view=resources/);
 await expect(draft).toBeHidden();
 await page.goBack();await expect(draft).toBeVisible();
 await expect(page.getByLabel('Work-order title',{exact:true})).toHaveValue('Keep this unsaved inspection draft');
 await page.goForward();await expect(views.getByRole('button',{name:'Resources',exact:true})).toHaveAttribute('aria-current','page');
 await page.reload();await expect(views.getByRole('button',{name:'Resources',exact:true})).toHaveAttribute('aria-current','page');
 await expect(page.getByLabel('Planning equipment',{exact:true})).toHaveValue('PLA-PUMP-001');
 await views.getByRole('button',{name:'Work orders',exact:true}).click();
 await expect(page).toHaveURL(/view=work/);
 await expect(views.getByRole('button',{name:'Work orders',exact:true})).toHaveAttribute('aria-current','page');
 await page.reload();await expect(draft).toBeHidden();
 await expect(page.getByRole('heading',{name:'Maintenance candidates',exact:true})).toBeVisible();
 const query=new URL(page.url()).searchParams;
 expect(query.get('asset')).toBe('PLA-PUMP-001');expect(query.get('investigation')).toBe('navigation-context-only');
 await views.getByRole('button',{name:'Schedules',exact:true}).click();
 await expect(page).toHaveURL(/view=schedule/);
 await expect(views.getByRole('button',{name:'Schedules',exact:true})).toHaveAttribute('aria-current','page');
 await page.reload();await expect(page.getByRole('heading',{name:'Proposals and confirmed schedules',exact:true})).toBeVisible();
 await page.goto('/operations/scheduling?view=invalid');
 await expect(views.getByRole('button',{name:'Work orders',exact:true})).toHaveAttribute('aria-current','page');
});

test('administration views support direct links, refresh and browser history',async({page})=>{
 await login(page,'admin');
 await page.goto('/operations/admin?view=facility');
 const views=page.getByRole('navigation',{name:'Administration views'});
 await expect(page.getByRole('heading',{name:'Facility identity and location',exact:true})).toBeVisible();
 await views.getByRole('button',{name:'Audit log',exact:true}).click();
 await expect(page).toHaveURL(/view=audit/);
 await page.reload();await expect(page.getByRole('heading',{name:'Latest administration and access audit events',exact:true})).toBeVisible();
 await page.goBack();await expect(views.getByRole('button',{name:'Facility',exact:true})).toHaveAttribute('aria-current','page');
 await page.goForward();await expect(views.getByRole('button',{name:'Audit log',exact:true})).toHaveAttribute('aria-current','page');
 await views.getByRole('button',{name:'Asset registry',exact:true}).click();
 await expect(page).toHaveURL(/view=registry/);
 await expect(views.getByRole('button',{name:'Asset registry',exact:true})).toHaveAttribute('aria-current','page');
 await page.reload();await expect(page.getByRole('heading',{name:'Registered assets and signal contracts',exact:true})).toBeVisible();
 await page.goto('/operations/admin?view=invalid');
 await expect(views.getByRole('button',{name:'People & access',exact:true})).toHaveAttribute('aria-current','page');
});
