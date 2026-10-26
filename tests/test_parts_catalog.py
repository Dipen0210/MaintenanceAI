import pytest
from tests.test_phase4_copilot import copilot,ASSET
from tests.test_phase6_scheduling import planning,login
from tests.test_inventory import setup,stock,move,PART,REASON
from tests.test_work_lifecycle import work
from backend.core.database import transaction

DETAILS=dict(expected_version=0,manufacturer='Demonstration Manufacturer',manufacturer_part_number='KIT-100',unit='units',location='Stores A / shelf 1',minimum_stock=3,target_stock=10,lead_time_days=4,compatible_assets=[ASSET],substitute_ids=[],reason=REASON)

def save(c,**kw):return c.put('/api/inventory/catalog/'+PART,json={**DETAILS,**kw})

def test_catalog_replenishment_uses_reserved_and_approved_demand_without_changing_stock(planning):
 c=planning;setup(c,5);assert save(c).status_code==200
 work(c,parts={PART:3});work(c,'approved',parts={PART:4})
 p=c.get('/api/inventory/catalog').json()['parts'][0]
 assert p['on_hand']==5 and p['available']==2 and p['replenishment']['shortage']==2 and p['replenishment']['suggested_quantity']==8
 assert save(c).status_code==409
 assert save(c,expected_version=1,unit='boxes').status_code==409
 assert stock(c)['on_hand']==5

def test_new_part_enters_planning_without_fabricating_stock(planning):
 c=planning;setup(c);before=c.get('/api/planning/config').json()['id']
 r=c.post('/api/inventory/catalog',json={**DETAILS,'part_id':'NEW-BEARING','name':'Replacement bearing','compatible_assets':[]});assert r.status_code==200,r.text
 cfg=c.get('/api/planning/config').json();assert cfg['id']!=before
 assert next(p for p in cfg['config']['parts'] if p['id']=='NEW-BEARING')['quantity']==0
 assert next(p for p in c.get('/api/inventory/catalog').json()['parts'] if p['part_id']=='NEW-BEARING')['on_hand']==0
 assert c.post('/api/inventory/catalog',json={**DETAILS,'part_id':'NEW-BEARING','name':'Replacement bearing'}).status_code==409

def test_catalog_roles_scope_and_metadata_validation(planning):
 c=planning;setup(c);login(c,'engineer');assert save(c).status_code==403
 login(c,'manager');assert save(c,target_stock=2).status_code==422
 assert save(c,minimum_stock=-1).status_code==422
 assert save(c,compatible_assets=['not-authorized']).status_code==404
 assert save(c,substitute_ids=[PART]).status_code==422
 c.headers['X-Maintenance-Workspace']='plant';assert save(c).status_code==409
 c.headers['X-Maintenance-Workspace']='demo'
 with transaction() as conn:conn.execute("DELETE FROM asset_grants WHERE user_id='manager' AND asset_id=?",(ASSET,))
 assert c.get('/api/inventory/catalog').status_code==403
