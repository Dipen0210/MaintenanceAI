"""Real stock accounting, custody reconciliation and transactional scheduling protections."""
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import pytest
from fastapi import HTTPException
from backend.core.database import transaction
from backend.inventory.service import apply
from backend.routes.inventory import Movement
from tests.test_phase4_copilot import copilot,ASSET
from tests.test_phase6_scheduling import planning,login,proposed,APPROVAL
from tests.test_work_lifecycle import named_config,work,lifecycle
from tests.test_field_work import change

PART='demo-inspection-kit'
REASON='Recorded a reviewed synthetic stock movement for this test.'


def setup(c,quantity=10):
    c.headers['X-Maintenance-Workspace']='demo';named_config(c)
    r=c.post('/api/inventory/activate',json=dict(counting_confirmed=True,reason=REASON));assert r.status_code==200,r.text
    assert stock(c)['on_hand']==0
    if quantity:assert move(c,'receipt',quantity).status_code==200


def stock(c):
    r=c.get('/api/inventory');assert r.status_code==200,r.text;return next(p for p in r.json()['parts'] if p['part_id']==PART)


def get_work(c,wid):return next(w for w in c.get('/api/work-orders').json() if w['id']==wid)


def move(c,action,quantity,wid=None,version=None,key=None):
    if version is None:version=get_work(c,wid)['execution']['version'] if wid else stock(c)['version']
    return c.post((f'/api/work-orders/{wid}/parts/' if wid else '/api/inventory/parts/')+PART+'/movements',json=dict(action=action,quantity=quantity,expected_version=version,request_key=key or str(uuid4()),reason=REASON))


def test_activation_never_converts_estimates_and_blocks_existing_work(planning):
    c=planning;c.headers['X-Maintenance-Workspace']='demo';wid,_=work(c)
    r=c.post('/api/inventory/activate',json=dict(counting_confirmed=True,reason=REASON));assert r.status_code==409
    assert lifecycle(c,wid,'replan').status_code==200
    assert c.post('/api/inventory/activate',json=dict(counting_confirmed=True,reason=REASON)).status_code==200
    assert stock(c)['on_hand']==0
    assert proposed(c,[wid]).json()['result']['feasibility']=='infeasible'
    assert move(c,'receipt',2).status_code==200
    assert c.post('/api/inventory/activate',json=dict(counting_confirmed=True,reason=REASON)).status_code==200
    assert stock(c)['on_hand']==2


def test_reserve_issue_consume_return_and_verification_release(planning):
    c=planning;setup(c);wid,_=work(c,parts={PART:4})
    assert (stock(c)['on_hand'],stock(c)['reserved'],stock(c)['available'])==(10,4,6)
    assert move(c,'issue',3,wid).status_code==200
    login(c,'engineer');w=get_work(c,wid);assert change(c,wid,'start',w['execution']['version'],site_clearance_confirmed=True).status_code==200
    assert move(c,'consume',2,wid).status_code==200;w=get_work(c,wid);assert change(c,wid,'report_fixed',w['execution']['version']).status_code==200
    login(c,'manager');w=get_work(c,wid);assert change(c,wid,'verify',w['execution']['version']).status_code==409
    assert move(c,'return',1,wid).status_code==200
    assert change(c,wid,'verify',w['execution']['version']).status_code==409
    w=get_work(c,wid);assert change(c,wid,'verify',w['execution']['version']).status_code==200
    row=get_work(c,wid)['inventory']['parts'][0];assert (row['reserved'],row['issued'],row['consumed'],row['returned'])==(0,0,2,1)
    assert (stock(c)['on_hand'],stock(c)['reserved'],stock(c)['available'])==(8,0,8)
    assert move(c,'issue',1,wid).status_code==409
    with transaction() as conn:assert conn.execute("SELECT COUNT(*) FROM inventory_movements WHERE work_order_id=? AND action='release'",(wid,)).fetchone()[0]==1


def test_return_before_cancel_and_replanning_release_exactly_once(planning):
    c=planning;setup(c);wid,_=work(c,parts={PART:3});assert move(c,'issue',1,wid).status_code==200
    w=get_work(c,wid);assert lifecycle(c,wid,version=w['execution']['version']).status_code==409
    assert move(c,'return',1,wid).status_code==200;w=get_work(c,wid);assert lifecycle(c,wid,'replan',w['execution']['version']).status_code==200
    assert stock(c)['reserved']==0
    p=proposed(c,[wid]).json();assert c.post('/api/schedules/'+p['id']+'/approve',json=APPROVAL).status_code==200;assert stock(c)['reserved']==3
    w=get_work(c,wid);assert lifecycle(c,wid,version=w['execution']['version']).status_code==200;assert stock(c)['reserved']==0
    assert lifecycle(c,wid,version=w['execution']['version']+1).status_code==409


def test_fresh_stock_is_rechecked_at_proposal_confirmation(planning):
    c=planning;setup(c,2);wid,_=work(c,'approved',parts={PART:2});p=proposed(c,[wid]).json();assert p['result']['feasibility']=='feasible'
    assert move(c,'adjust',-1).status_code==200
    assert c.post('/api/schedules/'+p['id']+'/approve',json=APPROVAL).status_code==409
    assert stock(c)['reserved']==0
    assert move(c,'receipt',1).status_code==200
    assert c.post('/api/schedules/'+p['id']+'/approve',json=APPROVAL).status_code==200
    assert move(c,'adjust',-1).status_code==409


def test_issued_stock_is_not_subtracted_again_by_scheduler(planning):
    c=planning;setup(c,4);wid,_=work(c,parts={PART:2});assert move(c,'issue',2,wid).status_code==200
    other,_=work(c,'approved',parts={PART:2});p=proposed(c,[other]).json();assert p['result']['feasibility']=='feasible',p
    assert c.post('/api/schedules/'+p['id']+'/approve',json=APPROVAL).status_code==200
    assert stock(c)['on_hand']==2 and stock(c)['reserved']==2


def test_planning_estimate_edits_do_not_create_stock(planning):
    c=planning;setup(c,1);config=c.get('/api/planning/config').json()['config'];config['parts'][0]['quantity']=99999;config['reason']=REASON
    assert c.put('/api/planning/config',json=config).status_code==200;assert stock(c)['on_hand']==1
    wid,_=work(c,'approved',parts={PART:2});assert proposed(c,[wid]).json()['result']['feasibility']=='infeasible'


def test_concurrent_idempotent_receipts_and_payload_collision(planning):
    c=planning;setup(c,0);user=dict(id='manager',role='maintenance_manager',plant_id='demo-plant',workspace='demo');body=Movement(action='receipt',quantity=3,expected_version=0,request_key=str(uuid4()),reason=REASON)
    with ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(lambda _:apply(PART,body,user),range(2)))
    assert len({r['movement_id'] for r in result})==1 and stock(c)['on_hand']==3
    other=body.model_copy(update={'quantity':4})
    with pytest.raises(HTTPException) as e:apply(PART,other,user)
    assert e.value.status_code==409


def test_competing_adjustments_cannot_make_stock_negative(planning):
    c=planning;setup(c,1);v=stock(c)['version'];user=dict(id='manager',role='maintenance_manager',plant_id='demo-plant',workspace='demo')
    def withdraw(_):
        try:apply(PART,Movement(action='adjust',quantity=-1,expected_version=v,request_key=str(uuid4()),reason=REASON),user);return 200
        except HTTPException as e:return e.status_code
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(withdraw,range(2)))
    assert sorted(results)==[200,409] and stock(c)['on_hand']==0


@pytest.mark.parametrize('quantity',[0,-1,True,1.5])
def test_invalid_receipts_rejected(planning,quantity):
    c=planning;setup(c,0);assert move(c,'receipt',quantity,version=0).status_code==422;assert stock(c)['on_hand']==0


def test_roles_named_engineer_and_current_evidence_authorization(planning):
    c=planning;setup(c);wid,_=work(c,parts={PART:2});assert move(c,'issue',1,wid).status_code==200
    login(c,'engineer');assert c.get('/api/inventory').status_code==403;assert move(c,'receipt',1,version=0).status_code==403
    w=get_work(c,wid);assert change(c,wid,'start',w['execution']['version'],site_clearance_confirmed=True).status_code==200
    login(c,'reliability');assert move(c,'consume',1,wid).status_code==403
    login(c,'engineer');version=get_work(c,wid)['execution']['version'];key=str(uuid4());assert move(c,'consume',1,wid,version,key).status_code==200
    with transaction() as conn:conn.execute("DELETE FROM document_grants WHERE user_id='engineer'")
    assert move(c,'consume',1,wid,version,key).status_code==404


def test_stores_role_can_record_stock_but_cannot_approve_work(planning):
    c=planning;setup(c);wid,_=work(c,'draft')
    with transaction() as conn:conn.execute("UPDATE users SET role='stores_coordinator' WHERE id='reliability'")
    login(c,'reliability');assert c.get('/api/inventory').status_code==200;assert move(c,'receipt',2).status_code==200
    assert c.post('/api/work-orders/'+wid+'/review',json=APPROVAL).status_code==403
    with transaction() as conn:conn.execute("DELETE FROM asset_grants WHERE user_id='reliability' AND asset_id=?",(ASSET,))
    assert c.get('/api/inventory').status_code==403;assert move(c,'receipt',1,version=1).status_code==403


def test_demo_and_plant_ledgers_and_unselected_scope_are_isolated(planning):
    c=planning;setup(c);c.headers['X-Maintenance-Workspace']='plant';assert c.get('/api/inventory/status').json()['enabled'] is False;assert c.get('/api/inventory').json()['parts']==[]
    assert move(c,'receipt',1,version=0).status_code==409
    assert c.post('/api/inventory/activate',json=dict(counting_confirmed=True,reason=REASON)).status_code==200;assert stock(c)['on_hand']==0
    assert move(c,'receipt',2).status_code==200;c.headers['X-Maintenance-Workspace']='demo';assert stock(c)['on_hand']==10
    del c.headers['X-Maintenance-Workspace'];assert c.get('/api/inventory').status_code==422


def test_old_stock_snapshots_and_work_versions_do_not_overwrite_new_updates(planning):
    c=planning;setup(c);wid,_=work(c);version=get_work(c,wid)['execution']['version'];key=str(uuid4())
    assert move(c,'issue',1,wid,version,key).status_code==200
    assert move(c,'issue',1,wid,version,key).json()['replayed'] is True
    assert move(c,'issue',1,wid,version).status_code==409
    login(c,'engineer');assert change(c,wid,'start',version,site_clearance_confirmed=True).status_code==409


def test_concurrent_confirmations_cannot_overreserve_counted_stock(planning):
    from backend.scheduling.service import review_proposal
    from backend.scheduling.contracts import Review
    c=planning;setup(c,1);a,_=work(c,'approved');b,_=work(c,'approved')
    proposals=[proposed(c,[wid]).json()['id'] for wid in [a,b]]
    user=dict(id='manager',role='maintenance_manager',plant_id='demo-plant',workspace='demo')
    def confirm(pid):
        try:review_proposal(pid,Review(**APPROVAL),user);return 200
        except HTTPException as e:return e.status_code
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(confirm,proposals))
    assert sorted(results)==[200,409]
    assert stock(c)['on_hand']==1 and stock(c)['reserved']==1
    with transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM schedule_assignments').fetchone()[0]==1
        assert conn.execute("SELECT SUM(quantity) FROM inventory_movements WHERE action='reserve'").fetchone()[0]==1


def test_history_reauthorizes_work_and_new_catalog_entries_start_at_zero(planning):
    c=planning;setup(c);wid,_=work(c)
    assert any(m['work_order_id']==wid for m in c.get('/api/inventory').json()['history'])
    with transaction() as conn:conn.execute("DELETE FROM document_grants WHERE user_id='manager'")
    history=c.get('/api/inventory').json()['history'];assert history and all(m['work_order_id'] is None for m in history)
    config=c.get('/api/planning/config').json()['config'];config['parts'].append({**config['parts'][0],'id':'new-counted-part','name':'New counted part','quantity':999});config['reason']=REASON
    assert c.put('/api/planning/config',json=config).status_code==200
    new=next(p for p in c.get('/api/inventory').json()['parts'] if p['part_id']=='new-counted-part');assert new['on_hand']==new['reserved']==0


def test_reopening_verified_repair_reserves_remaining_allowance_atomically(planning):
    from tests.test_operations_coordination import configure,decision,handover
    c=planning;setup(c,2);assert configure(c).status_code==200
    login(c,'admin');admin=next(u for u in c.get('/api/admin/identities').json() if u['id']=='admin')
    body={k:v for k,v in {**admin,'username':'operations','password':'operations-demo-password','role':'operations_coordinator','reason':REASON}.items() if k!='id'}
    assert c.post('/api/admin/identities',json=body).status_code==200
    wid,_=work(c,'approved',parts={PART:2});p=proposed(c,[wid]).json()
    login(c,'operations');assert decision(c,p['operations_requests'][0]['id']).status_code==200
    login(c,'manager');assert c.post('/api/schedules/'+p['id']+'/approve',json=APPROVAL).status_code==200
    assert move(c,'issue',1,wid).status_code==200
    login(c,'operations');assert handover(c,wid,'release').status_code==200
    login(c,'engineer');assert change(c,wid,'start',1,site_clearance_confirmed=True).status_code==200
    assert move(c,'consume',1,wid).status_code==200;assert change(c,wid,'report_fixed',3).status_code==200
    login(c,'manager');assert change(c,wid,'verify',4).json()['status']=='awaiting_return_to_service'
    assert stock(c)['reserved']==0;assert move(c,'adjust',-1).status_code==200
    assert change(c,wid,'return_to_work',5).status_code==409
    w=get_work(c,wid);assert w['status']=='awaiting_return_to_service' and w['operations']['status']=='released' and w['execution']['version']==5
    assert move(c,'receipt',1).status_code==200
    assert change(c,wid,'return_to_work',5).status_code==200
    w=get_work(c,wid);assert w['status']=='scheduled' and w['operations']['status']=='held'
    assert stock(c)['reserved']==1 and w['inventory']['parts'][0]['consumed']==1
