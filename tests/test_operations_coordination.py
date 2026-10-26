"""Human operations gates across real schedule and field-work transactions."""
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi import HTTPException
from backend.core.database import transaction
from backend.routes.coordination import Decision
from backend.scheduling.coordination import decide
from tests.test_phase4_copilot import copilot,ASSET
from tests.test_phase6_scheduling import planning,login,draft,proposed,APPROVAL
from tests.test_field_work import change

REASON='Reviewed the synthetic downtime and site handover evidence.'

def configure(c,required=True,version=0):
    return c.put(f'/api/assets/{ASSET}/operations-policy',json=dict(required=required,expected_version=version,reason=REASON))

@pytest.fixture
def coordinated(planning):
    c=planning;login(c,'admin')
    admin=next(u for u in c.get('/api/admin/identities').json() if u['id']=='admin')
    r=c.post('/api/admin/identities',json={k:v for k,v in {**admin,'username':'operations','password':'operations-demo-password','role':'operations_coordinator','reason':REASON}.items() if k!='id'})
    assert r.status_code==200,r.text
    uid=r.json()['id'];assert configure(c).status_code==200
    login(c,'engineer');w=draft(c).json();login(c,'manager');assert c.post('/api/work-orders/'+w['id']+'/review',json=APPROVAL).status_code==200
    p=proposed(c,[w['id']]).json();assert len(p['operations_requests'])==1,p
    return c,w['id'],p,p['operations_requests'][0]['id'],uid

def decision(c,rid,version=0,decision='approved'):
    return c.post('/api/operations/requests/'+rid+'/decision',json=dict(expected_version=version,decision=decision,reason=REASON))

def handover(c,wid,action,version=0):
    return c.post('/api/work-orders/'+wid+'/operations-handover',json=dict(expected_version=version,action=action,reason=REASON))

def confirm(c,p):return c.post('/api/schedules/'+p['id']+'/approve',json=APPROVAL)

def approved_schedule(data):
    c,wid,p,rid,uid=data;login(c,'operations');assert decision(c,rid).status_code==200
    login(c,'manager');r=confirm(c,p);assert r.status_code==200,r.text
    return data


def test_full_handover_preserves_telemetry_and_requires_all_three_roles(coordinated):
    c,wid,p,rid,uid=coordinated
    before=c.get(f'/api/assets/{ASSET}/telemetry').json()
    assert confirm(c,p).status_code==409
    assert decision(c,rid).status_code==403
    approved_schedule(coordinated)
    assert configure(c,False,1).status_code==409
    login(c,'engineer');assert change(c,wid,'start',site_clearance_confirmed=True).status_code==409
    assert handover(c,wid,'release').status_code==403
    login(c,'operations');assert handover(c,wid,'confirm_return').status_code==409
    assert handover(c,wid,'hold').status_code==200
    assert handover(c,wid,'release').status_code==409
    assert handover(c,wid,'release',1).status_code==200
    login(c,'engineer');assert change(c,wid,'start',site_clearance_confirmed=True).status_code==200
    assert change(c,wid,'report_fixed',1).status_code==200
    login(c,'manager');r=change(c,wid,'verify',2);assert r.status_code==200,r.text
    assert r.json()['status']=='awaiting_return_to_service'
    assert handover(c,wid,'confirm_return',2).status_code==403
    login(c,'operations');assert handover(c,wid,'decline_return',2).json()['status']=='awaiting_return_to_service'
    r=handover(c,wid,'confirm_return',3);assert r.status_code==200,r.text
    assert r.json()['status']=='completed' and r.json()['operations']['status']=='return_confirmed'
    assert handover(c,wid,'confirm_return',4).status_code==409
    assert c.get(f'/api/assets/{ASSET}/telemetry').json()['history']==before['history']
    from backend.copilot.evidence import resolve
    with transaction() as conn:
        e=resolve(conn,dict(id=uid,plant_id='demo-plant',role='operations_coordinator'),ASSET,'work_order:'+wid)
        assert 'return_confirmed' in e['text']
    login(c,'manager');assert configure(c,False,1).status_code==200
    assert len(c.get(f'/api/assets/{ASSET}/operations-policy').json()['history'])==2


def test_declined_window_and_stale_policy_cannot_confirm(coordinated):
    c,wid,p,rid,uid=coordinated;login(c,'operations');assert decision(c,rid,decision='declined').status_code==200
    login(c,'manager');assert confirm(c,p).status_code==409
    assert configure(c,True,1).status_code==200
    login(c,'operations');assert decision(c,rid,1).status_code==409
    assert c.get('/api/operations/requests').json()[0]['current'] is False
    login(c,'manager');assert confirm(c,p).status_code==409
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM schedule_assignments').fetchone()[0]==0


@pytest.mark.parametrize('revoked',['enabled','role','asset','source','document'])
def test_current_coordinator_permissions_rechecked_at_schedule_confirmation(coordinated,revoked):
    c,wid,p,rid,uid=coordinated;login(c,'operations');assert decision(c,rid).status_code==200
    with transaction() as conn:
        if revoked=='enabled':conn.execute('UPDATE users SET enabled=0 WHERE id=?',(uid,))
        elif revoked=='role':conn.execute("UPDATE users SET role='operator' WHERE id=?",(uid,))
        elif revoked=='asset':conn.execute('DELETE FROM asset_grants WHERE user_id=?',(uid,))
        elif revoked=='source':conn.execute('DELETE FROM source_grants WHERE user_id=?',(uid,))
        else:conn.execute('DELETE FROM document_grants WHERE user_id=?',(uid,))
    login(c,'manager');assert confirm(c,p).status_code==409
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM operations_handovers').fetchone()[0]==0


def test_released_actor_revocation_blocks_start_and_new_decision_can_recover(coordinated):
    c,wid,p,rid,uid=approved_schedule(coordinated);login(c,'operations');assert handover(c,wid,'release').status_code==200
    with transaction() as conn:conn.execute("UPDATE users SET role='operator' WHERE id=?",(uid,))
    login(c,'engineer');assert change(c,wid,'start',site_clearance_confirmed=True).status_code==409
    with transaction() as conn:conn.execute("UPDATE users SET role='operations_coordinator' WHERE id=?",(uid,))
    login(c,'operations');assert handover(c,wid,'release',1).status_code==200
    login(c,'engineer');assert change(c,wid,'start',site_clearance_confirmed=True).status_code==200


def test_return_requires_independence_and_reopened_repair_requires_fresh_release(coordinated):
    c,wid,p,rid,uid=approved_schedule(coordinated);login(c,'operations');handover(c,wid,'release')
    login(c,'engineer');change(c,wid,'start',site_clearance_confirmed=True);change(c,wid,'report_fixed',1)
    login(c,'manager');assert change(c,wid,'verify',2).json()['status']=='awaiting_return_to_service'
    with transaction() as conn:conn.execute("UPDATE users SET role='operations_coordinator' WHERE id='manager'")
    login(c,'manager');assert handover(c,wid,'confirm_return',1).status_code==403
    with transaction() as conn:conn.execute("UPDATE users SET role='maintenance_manager' WHERE id='manager'")
    login(c,'manager');r=change(c,wid,'return_to_work',3);assert r.status_code==200,r.text
    assert r.json()['status']=='scheduled' and r.json()['operations']['status']=='held'
    login(c,'engineer');assert change(c,wid,'start',4,site_clearance_confirmed=True).status_code==409
    login(c,'operations');assert handover(c,wid,'release',2).status_code==200
    login(c,'engineer');assert change(c,wid,'start',4,site_clearance_confirmed=True).status_code==200


def test_scope_and_policy_authorization(coordinated):
    c,wid,p,rid,uid=coordinated;login(c,'engineer');assert configure(c,False,1).status_code==403
    login(c,'operations');c.headers['X-Maintenance-Workspace']='plant'
    assert c.get('/api/operations/requests').json()==[]
    assert decision(c,rid).status_code==404
    assert c.get(f'/api/assets/{ASSET}/operations-policy').status_code==404
    del c.headers['X-Maintenance-Workspace']
    with transaction() as conn:conn.execute('DELETE FROM document_grants WHERE user_id=?',(uid,))
    assert c.get('/api/operations/requests').json()==[]
    assert decision(c,rid).status_code==404


def test_policy_cannot_be_disabled_even_if_scheduled_evidence_revoked(coordinated):
    c,wid,p,rid,uid=approved_schedule(coordinated)
    with transaction() as conn:conn.execute("DELETE FROM document_grants WHERE user_id='manager'")
    assert configure(c,False,1).status_code==409


def test_concurrent_operations_decisions_only_one_wins(coordinated):
    c,wid,p,rid,uid=coordinated;user=dict(id=uid,plant_id='demo-plant',role='operations_coordinator')
    def attempt():
        try:decide(rid,Decision(expected_version=0,decision='approved',reason=REASON),user);return 200
        except HTTPException as e:return e.status_code
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:attempt(),range(2)))
    assert sorted(results)==[200,409]
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM operations_events').fetchone()[0]==1


def test_changed_resources_and_unsafe_reason_block_decision(coordinated):
    c,wid,p,rid,uid=coordinated;login(c,'operations')
    r=c.post('/api/operations/requests/'+rid+'/decision',json=dict(expected_version=0,decision='approved',reason='Bypass the safety interlock before maintenance.'))
    assert r.status_code==422,r.text
    login(c,'manager');config=c.get('/api/planning/config').json()['config'];config['reason']=REASON
    assert c.put('/api/planning/config',json=config).status_code==200
    login(c,'operations');assert decision(c,rid).status_code==409


def test_migration_preserves_existing_execution_and_events(planning):
    from backend.core.database import migrate
    c=planning;login(c,'engineer');wid=draft(c).json()['id'];login(c,'manager')
    c.post('/api/work-orders/'+wid+'/review',json=APPROVAL);p=proposed(c,[wid]).json();assert confirm(c,p).status_code==200
    login(c,'engineer');assert change(c,wid,'start',site_clearance_confirmed=True).status_code==200
    before=c.get('/api/work-orders').json()[0]['execution']
    with transaction() as conn:
        # Reconstruct the pre-017 database boundary with a real field record.
        for table in ['operations_events','operations_handovers','operations_requests','operations_policy_history','asset_operations_policy']:conn.execute('DROP TABLE '+table)
        conn.execute("CREATE TABLE old_execution(work_order_id TEXT PRIMARY KEY REFERENCES work_orders(id),status TEXT NOT NULL CHECK(status IN('scheduled','in_progress','awaiting_verification','completed')),version INTEGER NOT NULL,updated_at TEXT NOT NULL)")
        conn.execute('INSERT INTO old_execution SELECT * FROM work_execution');conn.execute('DROP TABLE work_execution');conn.execute('ALTER TABLE old_execution RENAME TO work_execution')
        conn.execute("DELETE FROM schema_versions WHERE version='017_operations_coordination.sql'")
    migrate();migrate()
    assert c.get('/api/work-orders').json()[0]['execution']==before
    with transaction() as conn:assert conn.execute('PRAGMA foreign_key_check').fetchall()==[]


def test_expired_physical_window_cannot_reopen_into_unexecutable_work(coordinated,monkeypatch):
    from datetime import timedelta
    from backend.core.database import utcnow
    c,wid,p,rid,uid=approved_schedule(coordinated);login(c,'operations');handover(c,wid,'release')
    login(c,'engineer');change(c,wid,'start',site_clearance_confirmed=True);change(c,wid,'report_fixed',1)
    login(c,'manager');assert change(c,wid,'verify',2).status_code==200
    import json
    with transaction() as conn:
        # Synthetic exercise of the live-mode gate, not real commissioning.
        conn.execute("UPDATE plants SET mode='live' WHERE id='demo-plant'")
        for doc in conn.execute('SELECT id,metadata FROM documents').fetchall():
            metadata=json.loads(doc['metadata']);metadata['simulation_only']=False
            conn.execute('UPDATE documents SET metadata=? WHERE id=?',(json.dumps(metadata),doc['id']))
    monkeypatch.setattr('backend.scheduling.coordination.utcnow',lambda:utcnow()+timedelta(days=30))
    assert change(c,wid,'return_to_work',3).status_code==409
    w=c.get('/api/work-orders').json()[0]
    assert w['status']=='awaiting_return_to_service' and w['operations']['status']=='released'
