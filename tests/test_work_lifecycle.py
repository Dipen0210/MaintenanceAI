"""Unstarted lifecycle changes preserve evidence and cannot bypass approval or field gates."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta
import json
import pytest
from fastapi import HTTPException
from backend.core.database import transaction,iso,utcnow
from backend.scheduling.contracts import LifecycleChange
from backend.scheduling.lifecycle import change as lifecycle_change
from tests.test_phase4_copilot import copilot,ASSET
from tests.test_phase6_scheduling import planning,login,draft,proposed,APPROVAL
from tests.test_field_work import change
from tests.test_operations_coordination import coordinated,approved_schedule,decision,handover

REASON='Confirmed with the field team that this synthetic work has not started.'

def lifecycle(c,wid,action='cancel',version=0,**extra):
    return c.post('/api/work-orders/'+wid+'/lifecycle',json=dict(action=action,expected_version=version,reason=REASON,not_started_confirmed=True,**extra))

def named_config(c):
    login(c,'manager');config=c.get('/api/planning/config').json()['config']
    for t in config['technicians']:t['user_id']='engineer' if t['id']=='demo-mechanical' else 'reliability'
    config['reason']=REASON;assert c.put('/api/planning/config',json=config).status_code==200
    return config

def work(c,status='scheduled',**extra):
    login(c,'engineer');r=draft(c,**extra);assert r.status_code==200,r.text;w=r.json();login(c,'manager');p=None
    if status!='draft':assert c.post('/api/work-orders/'+w['id']+'/review',json=APPROVAL).status_code==200
    if status=='scheduled':
        p=proposed(c,[w['id']]).json();r=c.post('/api/schedules/'+p['id']+'/approve',json=APPROVAL);assert r.status_code==200,r.text
    return w['id'],p

@pytest.mark.parametrize('stage',['draft','approved','scheduled'])
def test_cancel_releases_only_unstarted_reservations_and_preserves_evidence(planning,stage):
    c=planning;named_config(c);wid,p=work(c,stage)
    with transaction() as conn:old=[dict(r) for r in conn.execute('SELECT * FROM schedule_assignments WHERE work_order_id=?',(wid,))]
    before=c.get(f'/api/assets/{ASSET}/telemetry').json()['history']
    r=lifecycle(c,wid);assert r.status_code==200,r.text
    w=r.json();assert w['status']=='cancelled' and w['execution']['window'] is None
    assert w['lifecycle_history'][0]['snapshot']['status']==stage
    assert lifecycle(c,wid,version=1).status_code==409
    assert proposed(c,[wid]).status_code==409
    if p:
        proposal=next(p2 for p2 in c.get('/api/schedules').json() if p2['id']==p['id'])
        assert proposal['status']=='approved' and not proposal['result']['assignments'][0]['active']
        assert proposal['result']['assignments'][0]['reservation_status']=='cancelled'
        from backend.copilot.evidence import resolve
        with transaction() as conn:
            e=resolve(conn,dict(id='manager',plant_id='demo-plant',role='maintenance_manager'),ASSET,'schedule:'+old[0]['id'])
            assert 'Historical cancelled reservation' in e['text'] and 'no longer reserved' in e['text']
    with transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM schedule_assignments WHERE work_order_id=?',(wid,)).fetchone()[0]==0
        assert conn.execute('SELECT COUNT(*) FROM schedule_assignment_archive WHERE work_order_id=?',(wid,)).fetchone()[0]==len(old)
        assert conn.execute('SELECT COUNT(*) FROM work_user_assignments WHERE work_order_id=?',(wid,)).fetchone()[0]==0
    login(c,'engineer');assert change(c,wid,'start',1,site_clearance_confirmed=True).status_code==409
    assert not any(w['id']==wid for w in c.get('/api/my-work').json()['work_orders'])
    assert c.get(f'/api/assets/{ASSET}/telemetry').json()['history']==before


def test_replan_reassigns_named_engineer_and_new_window_without_reusing_old_proposal(planning):
    c=planning;config=named_config(c);wid,_=work(c,'approved')
    window=next(w for w in config['equipment_windows'] if w['asset_id']==ASSET)
    body=dict(work_order_ids=[wid],start_utc=window['start_utc'],end_utc=window['end_utc'],technician_ids=['demo-mechanical'])
    old=c.post('/api/schedules/propose',json=body).json();alternative=c.post('/api/schedules/propose',json=body).json()
    assert c.post('/api/schedules/'+old['id']+'/approve',json=APPROVAL).status_code==200
    login(c,'engineer');assert any(w['id']==wid for w in c.get('/api/my-work').json()['work_orders'])
    login(c,'manager');r=lifecycle(c,wid,'replan');assert r.status_code==200,r.text
    assert r.json()['status']=='approved' and r.json()['execution']['version']==1
    assert c.post('/api/schedules/'+alternative['id']+'/approve',json=APPROVAL).status_code==409
    body.update(start_utc=iso(datetime.fromisoformat(window['start_utc'])+timedelta(hours=1)),technician_ids=['demo-electrical'])
    replacement=c.post('/api/schedules/propose',json=body);assert replacement.status_code==200,replacement.text
    new=replacement.json();assert new['result']['assignments'][0]['technician_ids']==['demo-electrical']
    assert c.post('/api/schedules/'+new['id']+'/approve',json=APPROVAL).status_code==200
    w=c.get('/api/work-orders').json()[0];assert w['execution']['version']==2 and w['status']=='scheduled'
    assert w['lifecycle_history'][0]['snapshot']['user_assignments'][0]['user_id']=='engineer'
    login(c,'engineer');assert c.get('/api/my-work').json()['work_orders']==[]
    assert change(c,wid,'start',2,site_clearance_confirmed=True).status_code==403
    login(c,'reliability');assert c.get('/api/my-work').json()['work_orders'][0]['id']==wid
    assert change(c,wid,'start',1,site_clearance_confirmed=True).status_code==409
    assert change(c,wid,'start',2,site_clearance_confirmed=True).status_code==200


def test_coordination_hold_fresh_window_approval_and_release_required_again(coordinated):
    c,wid,p,rid,uid=approved_schedule(coordinated);login(c,'operations');assert handover(c,wid,'release').status_code==200
    login(c,'manager');assert lifecycle(c,wid,'replan').status_code==409
    login(c,'operations');assert handover(c,wid,'hold',1).status_code==200
    login(c,'manager');assert lifecycle(c,wid,'replan').status_code==200
    new=proposed(c,[wid]).json();fresh=new['operations_requests'][0]['id'];assert fresh!=rid
    assert c.post('/api/schedules/'+new['id']+'/approve',json=APPROVAL).status_code==409
    login(c,'operations');assert decision(c,rid,1).status_code==409;assert decision(c,fresh).status_code==200
    login(c,'manager');assert c.post('/api/schedules/'+new['id']+'/approve',json=APPROVAL).status_code==200
    w=c.get('/api/work-orders').json()[0];assert w['operations']['status']=='awaiting_release' and w['operations']['version']>2
    login(c,'engineer');assert change(c,wid,'start',w['execution']['version'],site_clearance_confirmed=True).status_code==409
    login(c,'operations');assert handover(c,wid,'release',0).status_code==409
    assert handover(c,wid,'release',w['operations']['version']).status_code==200


@pytest.mark.parametrize('action',['cancel','replan'])
def test_started_work_cannot_be_abandoned(planning,action):
    c=planning;wid,p=work(c);login(c,'engineer');assert change(c,wid,'start',site_clearance_confirmed=True).status_code==200
    login(c,'manager');assert lifecycle(c,wid,action,1).status_code==409
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM schedule_assignments').fetchone()[0]==1


def test_reopened_previously_started_work_is_not_treated_as_unstarted(coordinated):
    c,wid,p,rid,uid=approved_schedule(coordinated);login(c,'operations');handover(c,wid,'release')
    login(c,'engineer');change(c,wid,'start',site_clearance_confirmed=True);change(c,wid,'report_fixed',1)
    login(c,'manager');change(c,wid,'verify',2);assert change(c,wid,'return_to_work',3).status_code==200
    assert lifecycle(c,wid,'cancel',4).status_code==409
    assert lifecycle(c,wid,'replan',4).status_code==409


def test_dependency_protection_and_cancelled_dependency_rejected(planning):
    c=planning;wid,p=work(c);dependent,_=work(c,'draft',dependencies=[wid])
    assert lifecycle(c,wid).status_code==409
    assert lifecycle(c,dependent).status_code==200
    assert lifecycle(c,wid).status_code==200
    login(c,'engineer');assert draft(c,dependencies=[wid]).status_code==409


def test_concurrent_lifecycle_changes_archive_once(planning):
    c=planning;wid,p=work(c);user=dict(id='manager',plant_id='demo-plant',role='maintenance_manager')
    def attempt(action):
        try:lifecycle_change(wid,LifecycleChange(action=action,expected_version=0,reason=REASON,not_started_confirmed=True),user);return 200
        except HTTPException as e:return e.status_code
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(attempt,['cancel','replan']))
    assert sorted(results)==[200,409]
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM schedule_assignment_archive').fetchone()[0]==1


def test_permissions_acknowledgement_and_revoked_evidence(planning):
    c=planning;wid,p=work(c)
    r=c.post('/api/work-orders/'+wid+'/lifecycle',json=dict(action='cancel',expected_version=0,reason=REASON));assert r.status_code==422
    login(c,'admin');assert lifecycle(c,wid).status_code==403
    login(c,'engineer');assert lifecycle(c,wid).status_code==403
    login(c,'manager');c.headers['X-Maintenance-Workspace']='plant';assert lifecycle(c,wid).status_code==404;del c.headers['X-Maintenance-Workspace']
    with transaction() as conn:conn.execute("DELETE FROM document_grants WHERE user_id='manager'")
    assert lifecycle(c,wid).status_code==404


def test_late_solver_result_cannot_resurrect_cancelled_work(planning,monkeypatch):
    from backend.scheduling import service
    c=planning;wid,p=work(c,'approved');original=service.solve
    def concurrent_cancel(snapshot):
        lifecycle_change(wid,LifecycleChange(action='cancel',expected_version=0,reason=REASON,not_started_confirmed=True),dict(id='manager',plant_id='demo-plant',role='maintenance_manager'))
        return original(snapshot)
    monkeypatch.setattr(service,'solve',concurrent_cancel)
    assert proposed(c,[wid]).status_code==409
    assert c.get('/api/schedules').json()==[]


@pytest.mark.parametrize('selection',[['missing'],['demo-mechanical','demo-mechanical'],[]])
def test_invalid_team_selection_rejected(planning,selection):
    c=planning;config=named_config(c);wid,p=work(c,'approved');window=next(w for w in config['equipment_windows'] if w['asset_id']==ASSET)
    assert c.post('/api/schedules/propose',json=dict(work_order_ids=[wid],start_utc=window['start_utc'],end_utc=window['end_utc'],technician_ids=selection)).status_code==422


def test_revoked_selected_engineer_blocks_confirmation(planning):
    c=planning;config=named_config(c);wid,p=work(c,'approved');window=next(w for w in config['equipment_windows'] if w['asset_id']==ASSET)
    p=c.post('/api/schedules/propose',json=dict(work_order_ids=[wid],start_utc=window['start_utc'],end_utc=window['end_utc'],technician_ids=['demo-mechanical'])).json()
    with transaction() as conn:conn.execute("UPDATE users SET enabled=0 WHERE id='engineer'")
    assert c.post('/api/schedules/'+p['id']+'/approve',json=APPROVAL).status_code==409


def test_cancelled_calendar_occurrence_remains_deduplicated(planning):
    from tests.test_preventive import template
    c=planning;login(c,'manager');body=template(c);body['start_local']=(utcnow()-timedelta(hours=1)).replace(tzinfo=None).isoformat();body['timezone']='UTC'
    p=c.post('/api/preventive-plans',json=body).json();r=c.post('/api/preventive-plans/'+p['id']+'/generate');assert r.status_code==200,r.text
    wid=r.json()['created_work_order_ids'][0];assert lifecycle(c,wid).status_code==200
    assert c.post('/api/preventive-plans/'+p['id']+'/generate').json()['created_work_order_ids']==[]
    assert c.get('/api/preventive-plans').json()[0]['occurrences'][0]['overdue'] is False


def test_replacement_still_respects_other_confirmed_machine_reservations(planning):
    c=planning;config=named_config(c);first,p=work(c);second,q=work(c,title='Other approved machine inspection')
    occupied=q['result']['assignments'][0]
    assert lifecycle(c,first,'replan').status_code==200
    r=c.post('/api/schedules/propose',json=dict(work_order_ids=[first],start_utc=occupied['start_utc'],end_utc=occupied['end_utc'],technician_ids=['demo-electrical']))
    assert r.status_code==200,r.text
    assert r.json()['result']['feasibility']=='infeasible'
    assert c.post('/api/schedules/'+r.json()['id']+'/approve',json=APPROVAL).status_code==409
    with transaction() as conn:
        assert [r[0] for r in conn.execute('SELECT work_order_id FROM schedule_assignments')]==[second]


def test_cancel_racing_engineer_start_has_only_one_accepted_outcome(planning):
    from backend.scheduling.execution import update
    from backend.scheduling.contracts import FieldUpdate
    c=planning;wid,p=work(c)
    def act(action):
        try:
            if action=='start':update(wid,FieldUpdate(action='start',expected_version=0,notes=REASON,site_clearance_confirmed=True),dict(id='engineer',plant_id='demo-plant',role='engineer'))
            else:lifecycle_change(wid,LifecycleChange(action='cancel',expected_version=0,reason=REASON,not_started_confirmed=True),dict(id='manager',plant_id='demo-plant',role='maintenance_manager'))
            return 200
        except HTTPException as e:return e.status_code
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(act,['start','cancel']))
    assert sorted(results)==[200,409]


def test_replan_with_explicit_policy_change_does_not_reuse_old_handover(coordinated):
    from tests.test_operations_coordination import configure
    c,wid,p,rid,uid=approved_schedule(coordinated)
    assert lifecycle(c,wid,'replan').status_code==200
    assert configure(c,False,1).status_code==200
    new=proposed(c,[wid]).json();assert new['operations_requests']==[]
    assert c.post('/api/schedules/'+new['id']+'/approve',json=APPROVAL).status_code==200
    w=c.get('/api/work-orders').json()[0];assert w['operations'] is None
    assert w['lifecycle_history'][0]['snapshot']['operations']['request_id']==rid
    login(c,'engineer');assert change(c,wid,'start',2,site_clearance_confirmed=True).status_code==200


def test_cancelled_meter_occurrence_remains_deduplicated(planning):
    from tests.test_meter_maintenance import setup,sample,check
    c=planning;p,_=setup(c);sample(100);wid=check(c,p)['created_work_order_ids'][0]
    assert lifecycle(c,wid).status_code==200
    assert check(c,p)['created_work_order_ids']==[]
    occurrence=c.get('/api/meter-plans').json()[0]['occurrences'][0]
    assert occurrence['status']=='cancelled' and occurrence['overdue'] is False


def test_lifecycle_migration_preserves_existing_field_records(planning):
    from backend.core.database import migrate
    c=planning;wid,p=work(c);login(c,'engineer');assert change(c,wid,'note').status_code==200
    before=c.get('/api/work-orders').json()[0]
    with transaction() as conn:
        for table in ['work_lifecycle_events','schedule_assignment_archive']:conn.execute('DROP TABLE '+table)
        conn.execute("CREATE TABLE old_execution(work_order_id TEXT PRIMARY KEY REFERENCES work_orders(id),status TEXT NOT NULL CHECK(status IN('scheduled','in_progress','awaiting_verification','awaiting_return_to_service','completed')),version INTEGER NOT NULL,updated_at TEXT NOT NULL)")
        conn.execute('INSERT INTO old_execution SELECT * FROM work_execution');conn.execute('DROP TABLE work_execution');conn.execute('ALTER TABLE old_execution RENAME TO work_execution')
        conn.execute("DELETE FROM schema_versions WHERE version='018_work_lifecycle.sql'")
    migrate();migrate()
    assert c.get('/api/work-orders').json()[0]==before
    with transaction() as conn:assert conn.execute('PRAGMA foreign_key_check').fetchall()==[]
