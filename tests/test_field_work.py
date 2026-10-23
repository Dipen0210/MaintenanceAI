"""Field completion records are human decisions, never a substitute for sensor evidence."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import pytest
from fastapi import HTTPException
from backend.core.database import transaction,utcnow,iso,migrate
from backend.scheduling.contracts import FieldUpdate
from backend.scheduling.execution import update
from tests.test_phase4_copilot import copilot,ASSET
from tests.test_phase6_scheduling import planning,login,draft,proposed,APPROVAL


@pytest.fixture
def scheduled(planning):
    c=planning;login(c,'engineer');r=draft(c);assert r.status_code==200,r.text
    wid=r.json()['id'];login(c,'manager');assert c.post(f'/api/work-orders/{wid}/review',json=APPROVAL).status_code==200
    p=proposed(c,[wid]).json();r=c.post(f'/api/schedules/{p["id"]}/approve',json=APPROVAL);assert r.status_code==200,r.text
    login(c,'engineer');return c,wid


def change(c,wid,action,version=0,**extra):
    return c.post(f'/api/work-orders/{wid}/field-update',json=dict(action=action,expected_version=version,
        notes='Synthetic field check: sensor mounting inspected and operations team informed.',**extra))


def test_repair_handover_verification_persist_and_do_not_clear_sensor_evidence(scheduled):
    c,wid=scheduled
    before=c.get(f'/api/assets/{ASSET}/telemetry').json()['history']
    assert change(c,wid,'report_fixed').status_code==409
    assert change(c,wid,'start').status_code==422
    r=change(c,wid,'start',site_clearance_confirmed=True);assert r.status_code==200,r.text
    assert r.json()['status']=='in_progress'
    assert change(c,wid,'note',0).status_code==409
    assert change(c,wid,'note',1).json()['execution']['version']==2
    r=change(c,wid,'report_fixed',2);assert r.json()['status']=='awaiting_verification',r.text
    assert change(c,wid,'verify',3).status_code==403
    login(c,'admin');assert change(c,wid,'verify',3).status_code==403
    login(c,'manager');assert change(c,wid,'return_to_work',3).json()['status']=='in_progress'
    login(c,'engineer');assert change(c,wid,'report_fixed',4).status_code==200
    login(c,'manager');r=change(c,wid,'verify',5);assert r.status_code==200,r.text
    assert r.json()['status']=='completed'
    assert [e['version'] for e in r.json()['execution']['events']]==list(range(1,7))
    assert change(c,wid,'verify',6).status_code==409
    assert change(c,wid,'note',6).status_code==409
    migrate();assert next(w for w in c.get('/api/work-orders').json() if w['id']==wid)['status']=='completed'
    assert c.get('/api/schedules').json()[0]['result']['assignments'][0]['work_status']=='completed'
    assert c.get(f'/api/assets/{ASSET}/telemetry').json()['history']==before
    from backend.copilot.tools import invoke
    from backend.copilot.evidence import resolve
    user=dict(id='engineer',plant_id='demo-plant',role='engineer')
    assert next(w for w in invoke('get_existing_work_orders',{'asset_id':ASSET},user) if w['id']==wid)['status']=='completed'
    assert not any(w.get('external_id')==wid for w in invoke('get_maintenance_schedule',{'asset_id':ASSET},user))
    with transaction() as conn:
        evidence=resolve(conn,user,ASSET,'work_order:'+wid)
        assert 'Status: completed' in evidence['text'] and 'Human record by manager' in evidence['text']
        assert 'do not establish current equipment health' in evidence['text']
        assert conn.execute("SELECT COUNT(*) FROM audit_events WHERE action LIKE 'field_work_%'").fetchone()[0]==6


def test_current_permissions_and_independent_reviewer(scheduled):
    c,wid=scheduled;assert change(c,wid,'start',site_clearance_confirmed=True).status_code==200
    assert change(c,wid,'report_fixed',1).status_code==200
    with transaction() as conn:conn.execute("UPDATE users SET role='maintenance_manager' WHERE id='engineer'")
    login(c,'engineer');assert change(c,wid,'verify',2).status_code==403
    login(c,'manager')
    with transaction() as conn:conn.execute("DELETE FROM source_grants WHERE user_id='manager' AND source_id='demo-simulator'")
    assert change(c,wid,'verify',2).status_code in [403,404]
    with transaction() as conn:assert conn.execute('SELECT status FROM work_execution WHERE work_order_id=?',(wid,)).fetchone()[0]=='awaiting_verification'


def test_field_updates_need_confirmed_work_and_scoped_assets(planning):
    c=planning;login(c,'engineer');wid=draft(c).json()['id']
    assert change(c,wid,'note').status_code==409
    with transaction() as conn:conn.execute('DELETE FROM asset_grants WHERE user_id=? AND asset_id=?',('engineer',ASSET))
    assert change(c,wid,'start',site_clearance_confirmed=True).status_code==404
    assert change(c,'missing','note').status_code==404


def test_real_work_start_requires_approved_time_window_and_dependency(scheduled):
    c,wid=scheduled
    with transaction() as conn:
        # Explicit synthetic test of the live-mode gate, not a real plant conversion.
        conn.execute("UPDATE plants SET mode='live' WHERE id='demo-plant'")
        for doc in conn.execute('SELECT id,metadata FROM documents').fetchall():
            metadata=json.loads(doc['metadata']);metadata['simulation_only']=False
            conn.execute('UPDATE documents SET metadata=? WHERE id=?',(json.dumps(metadata),doc['id']))
    assert change(c,wid,'start',site_clearance_confirmed=True).status_code==409
    with transaction() as conn:
        conn.execute('UPDATE schedule_assignments SET start_utc=?,end_utc=? WHERE work_order_id=?',(iso(utcnow()-timedelta(minutes=1)),iso(utcnow()+timedelta(minutes=59)),wid))
        row=dict(conn.execute('SELECT * FROM work_orders WHERE id=?',(wid,)).fetchone());original=row['content']
        row.update(id='unfinished-dependency',status='approved');conn.execute('INSERT INTO work_orders VALUES(?,?,?,?,?,?,?,?,?)',tuple(row.values()))
        content=json.loads(original);content['dependencies']=['unfinished-dependency'];conn.execute('UPDATE work_orders SET content=? WHERE id=?',(json.dumps(content),wid))
    assert change(c,wid,'start',site_clearance_confirmed=True).status_code==409
    with transaction() as conn:conn.execute('UPDATE work_orders SET content=? WHERE id=?',(original,wid))
    assert change(c,wid,'start',site_clearance_confirmed=True).status_code==200


def test_concurrent_field_notes_do_not_overwrite_each_other(scheduled):
    c,wid=scheduled;user=dict(id='engineer',plant_id='demo-plant',role='engineer')
    body=FieldUpdate(action='note',expected_version=0,notes='Synthetic concurrency test handover to operations.')
    def write():
        try:return update(wid,body,user)['execution']['version']
        except HTTPException as exc:return exc.status_code
    with ThreadPoolExecutor(max_workers=2) as executor:assert sorted(executor.map(lambda _:write(),range(2)))==[1,409]


def test_demo_scenarios_generate_visible_critical_and_preserve_real_mappings(copilot):
    c=copilot
    for scenario,conveyor,pump in [('mixed','warning','critical'),('critical','critical','normal'),('normal','normal','normal'),('warning','warning','normal')]:
        r=c.put('/api/simulation',json={'enabled':True,'scenario':scenario});assert r.status_code==200,r.text
        assert c.get(f'/api/assets/{ASSET}').json()['condition']==conveyor
        assert c.get('/api/assets/PLA-PUMP-002').json()['condition']==pump
    c.put('/api/simulation',json={'enabled':False});migrate()
    state=c.get('/api/simulation').json();assert state['scenario']=='warning' and not state['enabled']
    assert c.put('/api/simulation',json={'enabled':True,'scenario':'invented'}).status_code==422
    with transaction() as conn:conn.execute("DELETE FROM source_grants WHERE user_id='admin' AND source_id='demo-simulator'")
    assert c.put('/api/simulation',json={'enabled':True,'scenario':'critical'}).status_code==403
