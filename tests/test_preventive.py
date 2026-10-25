"""Calendar, authorization and real work-order lifecycle regressions."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import json
import pytest
from backend.core.database import transaction,iso,utcnow,migrate
from backend.preventive.calendar import occurrence
from backend.preventive.service import generate,sweep
from tests.test_phase4_copilot import copilot,ASSET
from tests.test_phase6_scheduling import planning,login,APPROVAL,proposed
from tests.test_field_work import change


def template(c,**changes):
    refs=c.get('/api/assets/'+ASSET+'/procedures').json();assert refs
    return dict(asset_id=ASSET,title='Weekly drive inspection',task='Inspect the sensor mounting according to the approved reference after site clearance.',
        procedure_ids=[refs[0]['id']],duration_minutes=30,required_skills=['mechanical'],team_size=1,parts={},
        start_local=(utcnow()-timedelta(minutes=2)).replace(tzinfo=None).isoformat(timespec='minutes'),timezone='UTC',interval_unit='weeks',interval_count=1,grace_hours=0,
        reason='Set up a recurring synthetic inspection for this test.',**changes)


def create(c,**changes):
    body=template(c);body.update(changes);r=c.post('/api/preventive-plans',json=body);assert r.status_code==200,r.text
    return r.json(),body


@pytest.mark.parametrize('anchor,zone,unit,index,expected',[
 ('2026-01-31T09:00:00','UTC','months',1,'2026-02-28T09:00:00+00:00'),
 ('2026-01-31T09:00:00','UTC','months',2,'2026-03-31T09:00:00+00:00'),
 ('2028-01-31T09:00:00','UTC','months',1,'2028-02-29T09:00:00+00:00'),
 ('2026-03-07T02:30:00','America/Chicago','days',1,'2026-03-08T08:00:00+00:00'),
 ('2026-03-07T02:30:00','America/Chicago','days',2,'2026-03-09T07:30:00+00:00'),
 ('2026-10-31T01:30:00','America/Chicago','days',1,'2026-11-01T06:30:00+00:00'),
 ('2026-10-31T01:30:00','America/Chicago','days',2,'2026-11-02T07:30:00+00:00'),
])
def test_calendar_anchor_dst_and_month_end(anchor,zone,unit,index,expected):
    assert occurrence(dict(start_local=anchor,timezone=zone,interval_unit=unit,interval_count=1),index).isoformat()==expected


def test_generation_idempotent_concurrent_and_does_not_assess_condition(planning):
    c=planning;login(c,'manager');p,_=create(c)
    user=dict(id='manager',plant_id='demo-plant',role='maintenance_manager',workspace='demo')
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:generate(p['id'],user),range(2)))
    assert sorted(len(r['created_work_order_ids']) for r in results)==[0,1]
    works=c.get('/api/work-orders').json();assert len(works)==1 and works[0]['status']=='draft'
    work=works[0];assert work['content']['risk_snapshot'] is None and work['content']['priority']=='Planned maintenance'
    with transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM risk_assessments').fetchone()[0]==0
        assert conn.execute('SELECT COUNT(*) FROM schedule_assignments').fetchone()[0]==0
    migrate();assert c.post('/api/preventive-plans/'+p['id']+'/generate').json()['created_work_order_ids']==[]
    result=c.get('/api/preventive-plans').json()[0];assert len(result['occurrences'])==1 and result['occurrences'][0]['overdue']


def test_generated_draft_can_be_reviewed_scheduled_executed_and_cited(planning):
    c=planning;login(c,'manager');p,_=create(c);wid=c.post('/api/preventive-plans/'+p['id']+'/generate').json()['created_work_order_ids'][0]
    login(c,'engineer');assert c.post('/api/work-orders/'+wid+'/review',json=APPROVAL).status_code==403
    login(c,'manager');assert c.post('/api/work-orders/'+wid+'/review',json=APPROVAL).status_code==200
    proposed_plan=proposed(c,[wid]).json();r=c.post('/api/schedules/'+proposed_plan['id']+'/approve',json=APPROVAL);assert r.status_code==200,r.text
    login(c,'engineer');assert change(c,wid,'start',site_clearance_confirmed=True).status_code==200
    assert change(c,wid,'report_fixed',1).status_code==200
    login(c,'manager');assert change(c,wid,'verify',2).status_code==200
    assert not c.get('/api/preventive-plans').json()[0]['occurrences'][0]['overdue']
    from backend.copilot.evidence import resolve
    with transaction() as conn:
        evidence=resolve(conn,dict(id='manager',plant_id='demo-plant',role='maintenance_manager'),ASSET,'work_order:'+wid)
        assert 'Calendar preventive task' in evidence['text'] and 'No condition-based risk assessment' in evidence['text']


def test_pause_resume_catchup_bounded_and_revisions_preserve_jobs(planning):
    c=planning;login(c,'manager');start=(utcnow()-timedelta(days=25)).replace(tzinfo=None).isoformat(timespec='minutes')
    p,body=create(c,start_local=start,interval_unit='days')
    state={'expected_version':p['version'],'enabled':False,'reason':'Pause the recurring plan during the test shutdown.'}
    r=c.post('/api/preventive-plans/'+p['id']+'/state',json=state);assert r.status_code==200
    sweep();assert c.get('/api/work-orders').json()==[]
    assert c.post('/api/preventive-plans/'+p['id']+'/generate').status_code==409
    assert c.post('/api/preventive-plans/'+p['id']+'/state',json={**state,'enabled':True}).status_code==409
    state.update(expected_version=2,enabled=True);assert c.post('/api/preventive-plans/'+p['id']+'/state',json=state).status_code==200
    r=c.post('/api/preventive-plans/'+p['id']+'/generate').json();assert len(r['created_work_order_ids'])==20 and r['catch_up_pending']
    sweep();assert len(c.get('/api/work-orders').json())==26
    old=c.get('/api/work-orders').json()[0]
    r=c.put('/api/preventive-plans/'+p['id'],json={**body,'title':'Revised drive inspection','expected_version':3});assert r.status_code==200,r.text
    assert c.post('/api/preventive-plans/'+p['id']+'/generate').json()['created_work_order_ids']==[]
    assert next(w for w in c.get('/api/work-orders').json() if w['id']==old['id'])['content']==old['content']
    assert [v['version'] for v in r.json()['history']]==[4,3,2,1]


@pytest.mark.parametrize('revocation',['disabled','role','asset','document'])
def test_background_generation_rechecks_owner_and_documents(planning,revocation):
    c=planning;login(c,'manager');p,_=create(c)
    with transaction() as conn:
        if revocation=='disabled':conn.execute("UPDATE users SET enabled=0 WHERE id='manager'")
        elif revocation=='role':conn.execute("UPDATE users SET role='operator' WHERE id='manager'")
        elif revocation=='asset':conn.execute("DELETE FROM asset_grants WHERE user_id='manager'")
        else:conn.execute("UPDATE documents SET status='revoked'")
    sweep()
    with transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM work_orders').fetchone()[0]==0
        assert conn.execute('SELECT last_error FROM preventive_plans WHERE id=?',(p['id'],)).fetchone()[0]


def test_scope_role_and_calendar_validation(planning):
    c=planning;body=template(c)
    assert c.post('/api/preventive-plans',json=body).status_code==403
    login(c,'manager');p,_=create(c)
    assert c.get('/api/preventive-plans',headers={'X-Maintenance-Workspace':'plant'}).json()==[]
    assert c.post('/api/preventive-plans/'+p['id']+'/generate',headers={'X-Maintenance-Workspace':'plant'}).status_code==404
    for extra in [{'timezone':'invented/zone'},{'start_local':'2026-10-24T09:00:00Z'},{'interval_count':0},{'procedure_ids':[]},{'task':'Bypass the safety interlock during inspection'}]:
        assert c.post('/api/preventive-plans',json={**body,**extra}).status_code==422
    login(c,'engineer');assert c.post('/api/preventive-plans/'+p['id']+'/generate').status_code==403
    assert c.get('/api/preventive-plans').status_code==200
