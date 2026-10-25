"""Assigned ordered inspection evidence, revisions and atomic issue linkage."""
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import pytest
from fastapi import HTTPException
from backend.core.database import transaction,iso,utcnow
from backend.inspections.service import update,Update
from tests.test_phase4_copilot import copilot,ASSET
from tests.test_phase6_scheduling import planning,login


def template(c):
    login(c,'manager');refs=c.get('/api/assets/'+ASSET+'/procedures').json()
    step=dict(asset_id=ASSET,title='Recorded temperature check',instructions='Read the existing recorded temperature according to the approved reference.',procedure_ids=[refs[0]['id']],kind='numeric',unit='degC',lower=10,upper=80)
    body=dict(title='Conveyor inspection round',steps=[step,{**step,'title':'Sensor quality check','kind':'check','unit':'','lower':None,'upper':None}],enabled=True,reason='Configure an explicitly synthetic inspection round.')
    r=c.post('/api/inspections/templates',json=body);assert r.status_code==200,r.text
    return r.json(),body


def assign(c,t):
    body=dict(expected_template_version=t['version'],assignee_id='engineer',due_at=iso(utcnow()+timedelta(days=1)),request_key=str(uuid4()))
    r=c.post('/api/inspections/templates/'+t['id']+'/assign',json=body);assert r.status_code==200,r.text
    return r.json(),body


def act(c,r,action,**extra):
    response=c.post('/api/inspections/runs/'+r['id']+'/updates',json=dict(expected_version=r['version'],action=action,notes='Recorded observation from the synthetic inspection.',**extra))
    return response


def started(c):
    t,_=template(c);r,_=assign(c,t);login(c,'engineer');response=act(c,r,'start',site_clearance_confirmed=True);assert response.status_code==200,response.text
    return response.json()


def test_ordered_measurements_issue_link_and_completion(planning):
    c=planning;r=started(c)
    assert act(c,r,'complete').status_code==409
    assert act(c,r,'record',step_index=1,outcome='pass').status_code==409
    response=act(c,r,'record',step_index=0,outcome='measured',value=95);assert response.status_code==200,response.text
    r=response.json();iid=r['results'][0]['issue_id'];assert iid and r['results'][0]['abnormal']
    assert c.get('/api/my-work').json()['inspection_rounds'][0]['id']==r['id']
    # A corrected measurement preserves the finding for triage and immutable prior evidence.
    response=act(c,r,'record',step_index=0,outcome='measured',value=70);assert response.status_code==200
    r=response.json();assert r['results'][0]['issue_id']==iid and not r['results'][0]['abnormal']
    assert len(c.get('/api/issues').json())==1
    assert r['events'][-1]['detail']['before']['value']==95
    r=act(c,r,'record',step_index=1,outcome='pass').json()
    r=act(c,r,'complete').json();assert r['status']=='completed'
    assert c.get('/api/my-work').json()['inspection_rounds']==[]
    assert act(c,r,'record',step_index=1,outcome='fail').status_code==409
    with transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM work_orders').fetchone()[0]==0
        assert conn.execute('SELECT COUNT(*) FROM risk_assessments').fetchone()[0]==0
        assert conn.execute('SELECT status FROM issues WHERE id=?',(iid,)).fetchone()[0]=='reported'


def test_skips_and_concurrent_duplicate_findings(planning):
    c=planning;r=started(c)
    body=Update(expected_version=r['version'],action='record',step_index=0,outcome='measured',value=95,notes='Synthetic test observation requires issue triage.')
    user=dict(id='engineer',plant_id='demo-plant',role='engineer')
    def call(_):
        try:return update(r['id'],body,user)
        except HTTPException as e:return e.status_code
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(call,range(2)))
    assert sum(v==409 for v in results)==1
    r=next(v for v in results if isinstance(v,dict));assert len(c.get('/api/issues').json())==1
    assert act(c,r,'record',step_index=1,outcome='skipped',value=2).status_code==422
    r=act(c,r,'record',step_index=1,outcome='skipped').json();r=act(c,r,'complete').json()
    assert r['status']=='completed_with_skips' and r['events'][-1]['detail']['skipped_steps']==1


def test_template_revision_preserves_assigned_snapshot_and_assignment_is_idempotent(planning):
    c=planning;t,body=template(c);r,assignment=assign(c,t)
    assert c.post('/api/inspections/templates/'+t['id']+'/assign',json=assignment).json()['id']==r['id']
    assert len(c.get('/api/inspections/runs').json())==1
    response=c.put('/api/inspections/templates/'+t['id'],json={**body,'title':'Revised inspection template','enabled':False,'expected_version':1});assert response.status_code==200,response.text
    assert c.get('/api/inspections/runs').json()[0]['content']['title']=='Conveyor inspection round'
    assert c.post('/api/inspections/templates/'+t['id']+'/assign',json={**assignment,'expected_template_version':2,'request_key':str(uuid4())}).status_code==409
    assert c.put('/api/inspections/templates/'+t['id'],json={**body,'expected_version':1}).status_code==409
    login(c,'engineer');assert act(c,r,'start',site_clearance_confirmed=True).status_code==200


@pytest.mark.parametrize('revocation',['asset','document','assignee','disabled'])
def test_current_access_and_role_gate_updates(planning,revocation):
    c=planning;r=started(c)
    with transaction() as conn:
        if revocation=='asset':conn.execute("DELETE FROM asset_grants WHERE user_id='engineer'")
        elif revocation=='document':conn.execute("UPDATE documents SET status='revoked'")
        elif revocation=='assignee':conn.execute("UPDATE inspection_runs SET assignee_id='manager'")
        else:conn.execute("UPDATE users SET enabled=0 WHERE id='engineer'")
    assert act(c,r,'record',step_index=0,outcome='measured',value=70).status_code in [401,403,404]
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM inspection_results').fetchone()[0]==0


def test_scope_and_record_contracts(planning):
    c=planning;t,body=template(c);r,_=assign(c,t)
    assert act(c,r,'start',site_clearance_confirmed=True).status_code==403
    assert c.get('/api/inspections/runs',headers={'X-Maintenance-Workspace':'plant'}).json()==[]
    login(c,'engineer');assert c.post('/api/inspections/templates',json=body).status_code==403
    assert act(c,r,'start').status_code==409
    r=act(c,r,'start',site_clearance_confirmed=True).json()
    assert act(c,r,'record',step_index=0,outcome='pass').status_code==422
    assert act(c,r,'record',step_index=0,outcome='measured').status_code==422
    assert act(c,r,'record',step_index=0,outcome='skipped',value=1).status_code==422
    assert act(c,r,'cancel').status_code==403
    login(c,'manager');assert act(c,r,'cancel').json()['status']=='cancelled'
    for step in [{**body['steps'][0],'lower':90,'upper':10},{**body['steps'][0],'unit':''},{**body['steps'][1],'instructions':'Bypass the safety interlock for inspection.'}]:
        assert c.post('/api/inspections/templates',json={**body,'steps':[step]}).status_code==422


def test_copilot_reads_current_round_findings_with_immutable_citations(planning):
    from backend.copilot.tools import invoke
    from backend.copilot.evidence import resolve
    c=planning;r=started(c);r=act(c,r,'record',step_index=0,outcome='measured',value=95).json()
    user=dict(id='engineer',plant_id='demo-plant',role='engineer')
    items=invoke('get_fault_history',dict(asset_id=ASSET),user);saved=next(v for v in items if v.get('evidence_kind')=='inspection')
    act(c,r,'record',step_index=0,outcome='measured',value=70)
    items=invoke('get_fault_history',dict(asset_id=ASSET),user);current=next(v for v in items if v.get('evidence_kind')=='inspection')
    assert saved['id']!=current['id']
    with transaction() as conn:
        assert 'Recorded 95.0 degC' in resolve(conn,user,ASSET,'inspection:'+saved['id'])['text']
        assert 'Recorded 70.0 degC' in resolve(conn,user,ASSET,'inspection:'+current['id'])['text']
        conn.execute("UPDATE documents SET status='revoked'")
    assert not any(v.get('evidence_kind')=='inspection' for v in invoke('get_fault_history',dict(asset_id=ASSET),user))
    with transaction() as conn:
        with pytest.raises(HTTPException):resolve(conn,user,ASSET,'inspection:'+saved['id'])
