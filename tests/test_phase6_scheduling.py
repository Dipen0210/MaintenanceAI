"""Actual CP-SAT and transactional planning tests with explicitly synthetic resources."""
from copy import deepcopy
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
import json
import pytest
from fastapi import HTTPException
from backend.core.database import transaction,iso,utcnow
from backend.scheduling.solver import solve,validate,minute,stamp
from backend.scheduling.contracts import Review
from backend.scheduling.service import review_proposal
from backend.scheduling.demo import seed_planning
from tests.test_phase4_copilot import copilot,incident,ASSET


def scenario():
    start=minute(utcnow()+timedelta(days=1),True);period={'start_utc':stamp(start),'end_utc':stamp(start+240)}
    work=lambda wid,priority:dict(id=wid,asset_id=ASSET,content=dict(duration_minutes=60,required_skills=['mechanical'],team_size=1,parts={'kit':1},dependencies=[],due_at=stamp(start+120),priority=priority))
    return dict(**period,config=dict(technicians=[dict(id='tech',name='Test technician',skills=['mechanical'],availability=[period])],
        parts=[dict(id='kit',name='Test kit',quantity=3,available_at=stamp(start))],
        equipment_windows=[dict(**period,id='window',asset_id=ASSET,disruption_cost_per_minute=1)],blackouts=[],simulation_only=True),
        orders=[work('low','Monitor'),work('high','High priority')],existing=[])


def test_actual_solver_prioritizes_and_repeats_without_overlap():
    snapshot=scenario();result=solve(snapshot);assert result['feasibility']=='feasible',result
    assert result['optimal'] and validate(result['assignments'],snapshot)==[]
    rows={a['work_order_id']:a for a in result['assignments']}
    assert rows['high']['end_utc']<=rows['low']['start_utc']
    again=solve(snapshot);assert again['assignments']==result['assignments'] and again['objective']==result['objective']


@pytest.mark.parametrize('constraint',['skills','working_period','parts','equipment','blackout','cycle','duration','team'])
def test_each_hard_constraint_can_make_plan_infeasible(constraint):
    s=scenario();c=s['config'];start=minute(s['start_utc'])
    if constraint=='skills':c['technicians'][0]['skills']=['electrical']
    elif constraint=='working_period':c['technicians'][0]['availability'][0]['end_utc']=stamp(start+30)
    elif constraint=='parts':c['parts'][0]['quantity']=1
    elif constraint=='equipment':c['equipment_windows'][0]['asset_id']='different-asset'
    elif constraint=='blackout':c['blackouts']=[dict(asset_id=ASSET,start_utc=s['start_utc'],end_utc=s['end_utc'],reason='Operational restriction')]
    elif constraint=='cycle':s['orders'][0]['content']['dependencies']=['high'];s['orders'][1]['content']['dependencies']=['low']
    elif constraint=='duration':s['orders'][0]['content']['duration_minutes']=300
    elif constraint=='team':s['orders'][0]['content']['team_size']=2
    r=solve(s);assert r['feasibility']=='infeasible' and r['conflicts'] and r['assignments']==[],r


def test_parts_delivery_dependencies_and_existing_reservations():
    s=scenario();start=minute(s['start_utc']);s['config']['parts'][0]['available_at']=stamp(start+60)
    s['orders'][1]['content']['dependencies']=['low']
    # Overlapping existing reservation and blackout must be unioned, not treated as conflicting fixed tasks.
    s['existing']=[dict(work_order_id='prior',asset_id=ASSET,start_utc=stamp(start),end_utc=stamp(start+30),technician_ids=['tech'],parts={'kit':1})]
    s['config']['blackouts']=[dict(asset_id=ASSET,start_utc=stamp(start+15),end_utc=stamp(start+60),reason='Production window')]
    r=solve(s);assert r['feasibility']=='feasible',r
    rows={a['work_order_id']:a for a in r['assignments']}
    assert minute(rows['low']['start_utc'])>=start+60 and rows['low']['end_utc']<=rows['high']['start_utc']
    assert validate(r['assignments'],s)==[]


def test_combined_team_and_disruption_preference():
    s=scenario();s['orders']=s['orders'][:1];s['orders'][0]['content'].update(team_size=2,required_skills=['mechanical','electrical'])
    t=deepcopy(s['config']['technicians'][0]);t.update(id='electrician',skills=['electrical']);s['config']['technicians'].append(t)
    expensive=deepcopy(s['config']['equipment_windows'][0]);expensive.update(id='expensive',disruption_cost_per_minute=500);s['config']['equipment_windows'].append(expensive)
    r=solve(s);assert r['feasibility']=='feasible',r
    assert set(r['assignments'][0]['technician_ids'])=={'electrician','tech'} and r['assignments'][0]['window_id']=='window'


def test_independent_validator_rejects_mutated_outputs():
    s=scenario();r=solve(s)
    for change in ['overlap','parts','team','asset','duplicate','dependency']:
        a=deepcopy(r['assignments']);snapshot=deepcopy(s)
        if change=='overlap':a[1]['start_utc']=a[0]['start_utc'];a[1]['end_utc']=a[0]['end_utc']
        elif change=='parts':a[0]['parts']={}
        elif change=='team':a[0]['technician_ids']=[]
        elif change=='asset':a[0]['asset_id']='other'
        elif change=='duplicate':a.append(a[0])
        elif change=='dependency':snapshot['orders'][0]['content']['dependencies']=['missing']
        assert validate(a,snapshot),change


@pytest.fixture
def planning(copilot):
    seed_planning();return copilot


def login(client,name):
    r=client.post('/api/auth/login',json={'username':name,'password':name+'-demo-password'});assert r.status_code==200
    client.headers['Authorization']='Bearer '+r.json()['access_token']


def draft(client,**extra):
    incident(client)
    risk=client.get(f'/api/assets/{ASSET}/risk').json();refs=client.get(f'/api/assets/{ASSET}/procedures').json()
    assert refs
    body=dict(asset_id=ASSET,title='Review vibration sensor mounting',task='Perform the referenced inspection after qualified site isolation and permit review.',
              duration_minutes=60,required_skills=['mechanical'],team_size=1,parts={'demo-inspection-kit':1},dependencies=[],procedure_ids=[refs[0]['id']],risk_id=risk['id'],due_at=iso(utcnow()+timedelta(days=3)))
    return client.post('/api/work-orders/draft',json={**body,**extra})


def proposed(client,wids):
    c=client.get('/api/planning/config').json()['config'];w=next(w for w in c['equipment_windows'] if w['asset_id']==ASSET)
    return client.post('/api/schedules/propose',json={'work_order_ids':wids,'start_utc':w['start_utc'],'end_utc':w['end_utc']})


APPROVAL={'decision':'approved','reason':'Reviewed evidence, procedures and simulated resource availability.'}


def test_workflow_requires_manager_and_copilot_reads_confirmed_work(planning):
    c=planning;login(c,'engineer');r=draft(c);assert r.status_code==200,r.text
    work=r.json();wid=work['id'];assert work['status']=='draft' and work['content']['risk_snapshot']['id']
    assert proposed(c,[wid]).status_code==409
    assert c.post(f'/api/work-orders/{wid}/review',json=APPROVAL).status_code==403
    login(c,'admin');assert c.post(f'/api/work-orders/{wid}/review',json=APPROVAL).status_code==403
    login(c,'manager');r=c.post(f'/api/work-orders/{wid}/review',json=APPROVAL);assert r.status_code==200,r.text
    login(c,'engineer');r=proposed(c,[wid]);assert r.status_code==200,r.text
    p=r.json();assert p['result']['feasibility']=='feasible',p
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM schedule_assignments').fetchone()[0]==0
    assert c.post(f'/api/schedules/{p["id"]}/approve',json=APPROVAL).status_code==403
    login(c,'manager');r=c.post(f'/api/schedules/{p["id"]}/approve',json=APPROVAL);assert r.status_code==200,r.text
    assert r.json()['status']=='approved'
    assert c.post(f'/api/schedules/{p["id"]}/approve',json=APPROVAL).status_code==409
    from backend.copilot.tools import invoke
    from backend.copilot.evidence import resolve
    user=dict(id='engineer',plant_id='demo-plant',role='engineer')
    for tool,kind in [('get_existing_work_orders','work_order'),('get_maintenance_schedule','schedule')]:
        rows=invoke(tool,{'asset_id':ASSET},user);ref=next(r for r in rows if r.get('evidence_kind')==kind)
        with transaction() as conn:
            evidence=resolve(conn,user,ASSET,kind+':'+ref['id']);assert evidence['role']=='history'
    with transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM schedule_assignments').fetchone()[0]==1
        actions={r[0] for r in conn.execute('SELECT action FROM audit_events')}
        assert {'draft_work_order','review_work_order','propose_schedule','review_schedule'}<=actions


def test_draft_rejects_unsafe_stale_wrong_procedure_and_injected_priority(planning):
    c=planning
    for extra,status in [({'task':'Bypass the safety interlock before inspection'},422),({'risk_id':'not-current'},409),({'procedure_ids':['missing']},404),({'priority':'Monitor'},422)]:
        r=draft(c,**extra);assert r.status_code==status,r.text
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM work_orders').fetchone()[0]==0


def approved_work(c):
    login(c,'engineer')
    r=draft(c);assert r.status_code==200,r.text
    wid=r.json()['id'];login(c,'manager');assert c.post(f'/api/work-orders/{wid}/review',json=APPROVAL).status_code==200
    return wid


def test_concurrent_approvals_cannot_double_book(planning):
    c=planning;wid=approved_work(c);proposals=[proposed(c,[wid]).json() for _ in range(2)]
    user=dict(id='manager',plant_id='demo-plant',role='maintenance_manager')
    def approve(p):
        try:return review_proposal(p['id'],Review(**APPROVAL),user)['status']
        except HTTPException as exc:return exc.status_code
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(approve,proposals))
    assert sorted(map(str,results))==['409','approved']
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM schedule_assignments').fetchone()[0]==1


def test_different_work_conflicting_proposals_are_revalidated(planning):
    c=planning;ids=[approved_work(c) for _ in range(2)];ps=[proposed(c,[wid]).json() for wid in ids]
    assert ps[0]['result']['assignments'][0]['start_utc']==ps[1]['result']['assignments'][0]['start_utc']
    assert c.post('/api/schedules/'+ps[0]['id']+'/approve',json=APPROVAL).status_code==200
    r=c.post('/api/schedules/'+ps[1]['id']+'/approve',json=APPROVAL);assert r.status_code==409 and 'overlap' in r.text,r.text
    again=proposed(c,[ids[1]]);assert again.status_code==200,again.text
    assert c.post('/api/schedules/'+again.json()['id']+'/approve',json=APPROVAL).status_code==200


@pytest.mark.parametrize('change',['config','document','source','asset'])
def test_approval_rechecks_access_and_configuration(planning,change):
    c=planning;wid=approved_work(c);p=proposed(c,[wid]).json()
    if change=='config':
        cfg=c.get('/api/planning/config').json()['config'];assert c.put('/api/planning/config',json=cfg).status_code==200
    else:
        with transaction() as conn:
            if change=='document':conn.execute("UPDATE documents SET status='revoked'")
            elif change=='source':conn.execute("DELETE FROM source_grants WHERE user_id='manager'")
            else:conn.execute('DELETE FROM asset_grants WHERE user_id=? AND asset_id=?',('manager',ASSET))
    r=c.post('/api/schedules/'+p['id']+'/approve',json=APPROVAL);assert r.status_code in [404,409],r.text
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM schedule_assignments').fetchone()[0]==0


def test_resource_config_scope_and_rejection(planning):
    c=planning;config=c.get('/api/planning/config').json()['config'];login(c,'engineer')
    assert c.put('/api/planning/config',json=config).status_code==403
    wid=draft(c).json()['id'];login(c,'manager')
    r=c.post(f'/api/work-orders/{wid}/review',json={'decision':'rejected','reason':'Required site conditions have not been verified.'});assert r.status_code==200
    assert proposed(c,[wid]).status_code==409
    with transaction() as conn:conn.execute("DELETE FROM asset_grants WHERE user_id='manager' AND asset_id='PLA-PUMP-001'")
    assert c.put('/api/planning/config',json=config).status_code==403
    filtered=c.get('/api/planning/config').json()['config']
    assert all(w['asset_id']!='PLA-PUMP-001' for w in filtered['equipment_windows'])
