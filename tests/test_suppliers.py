from uuid import uuid4
from datetime import timedelta
import pytest
from tests.test_phase4_copilot import copilot
from tests.test_phase6_scheduling import planning,login
from tests.test_inventory import setup,PART,REASON
from backend.core.database import transaction,utcnow,iso

def supplier(c):
 r=c.post('/api/inventory/suppliers',json=dict(name='Synthetic supplier',contact='Stores desk extension 123',reference='SIM-SUP-1',notes='Synthetic test record',reason=REASON,request_key=str(uuid4())));assert r.status_code==200,r.text;return r.json()['id']
def approve(c,sid,version=1,action='approve'):return c.post('/api/inventory/suppliers/'+sid+'/review',json=dict(expected_version=version,action=action,verified=True,reason=REASON))
def quotation(c,sid,**kwargs):return c.post('/api/inventory/quotes',json=dict(supplier_id=sid,part_id=PART,reference='SIM-QUOTE-1',currency='USD',unit_price_minor=1250,shipping_minor=500,tax_minor=100,minimum_quantity=2,maximum_quantity=20,lead_time_days=5,valid_until=iso(utcnow()+timedelta(days=5)),reason=REASON,request_key=str(uuid4()),**kwargs))
def verify(c,qid,version=1,action='verify'):return c.post('/api/inventory/quotes/'+qid+'/review',json=dict(expected_version=version,action=action,verified=True,reason=REASON))

def test_supplier_quotes_must_be_reviewed_current_and_same_currency(planning):
 c=planning;setup(c);sid=supplier(c);assert quotation(c,sid).status_code==409;assert approve(c,sid).status_code==200
 q=quotation(c,sid);assert q.status_code==200;qid=q.json()['id']
 assert c.get('/api/inventory/quotes/compare',params=dict(part_id=PART,quantity=4)).json()['options']==[]
 assert verify(c,qid).status_code==200
 options=c.get('/api/inventory/quotes/compare',params=dict(part_id=PART,quantity=4)).json()['options'];assert options[0]['total_minor']==5600 and options[0]['currency']=='USD'
 assert c.get('/api/inventory/quotes/compare',params=dict(part_id=PART,quantity=1)).json()['options']==[]
 assert approve(c,sid,2,'suspend').status_code==200
 assert c.get('/api/inventory/quotes/compare',params=dict(part_id=PART,quantity=4)).json()['options']==[]
 assert approve(c,sid,3).status_code==200
 assert c.get('/api/inventory/quotes/compare',params=dict(part_id=PART,quantity=4)).json()['options']==[]

def test_supplier_roles_workspace_and_stale_review(planning):
 c=planning;setup(c);sid=supplier(c)
 with transaction() as conn:conn.execute("UPDATE users SET role='stores_coordinator' WHERE id='reliability'")
 login(c,'reliability');assert approve(c,sid).status_code==403
 login(c,'manager');assert approve(c,sid).status_code==200;assert approve(c,sid).status_code==409
 c.headers['X-Maintenance-Workspace']='plant';assert approve(c,sid,2).status_code==409
 login(c,'engineer');c.headers['X-Maintenance-Workspace']='demo';assert c.get('/api/inventory/suppliers').status_code==403

def test_duplicate_supplier_command_and_expired_quote_do_not_get_fabricated(planning):
 c=planning;setup(c);body=dict(name='Test supplier',contact='Store desk 100',reference='REF-100',notes='',reason=REASON,request_key=str(uuid4()))
 a=c.post('/api/inventory/suppliers',json=body);b=c.post('/api/inventory/suppliers',json=body);assert a.json()==b.json()
 assert c.post('/api/inventory/suppliers',json={**body,'name':'Different'}).status_code==409
 sid=a.json()['id'];approve(c,sid);qid=quotation(c,sid).json()['id'];verify(c,qid)
 import json
 with transaction() as conn:
  q=json.loads(conn.execute('SELECT content FROM procurement_quotes WHERE id=?',(qid,)).fetchone()[0]);q['valid_until']=iso(utcnow()-timedelta(seconds=1));conn.execute('UPDATE procurement_quotes SET content=? WHERE id=?',(json.dumps(q),qid))
 assert c.get('/api/inventory/quotes/compare',params=dict(part_id=PART,quantity=4)).json()['options']==[]
