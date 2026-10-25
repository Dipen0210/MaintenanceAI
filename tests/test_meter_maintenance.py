"""Recorded cumulative meters: threshold generation, authorization and reconciliation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import pytest
from backend.core.database import transaction,iso,utcnow,identifier
from backend.preventive.meters import generate,sweep,threshold
from tests.test_phase4_copilot import copilot,ASSET
from tests.test_phase6_scheduling import planning,login,APPROVAL,proposed
from tests.test_preventive import template
from tests.test_field_work import change


def setup(c,**changes):
    login(c,'admin')
    r=c.post(f'/api/assets/{ASSET}/meters',json=dict(tag_id='operating_hours',unit='h',source_id='demo-import',maximum=1e9,stale_after_seconds=300,reason='Map an explicitly synthetic test counter.'))
    assert r.status_code==200,r.text
    login(c,'manager');body=template(c)
    for key in ['start_local','timezone','interval_unit','interval_count']:body.pop(key)
    body.update(sensor_id=r.json()['id'],first_due=100,interval=100,**changes)
    r=c.post('/api/meter-plans',json=body);assert r.status_code==200,r.text
    return r.json(),body


def sample(value,seconds=0,quality='good',unit='h'):
    with transaction() as conn:
        sid=identifier()
        conn.execute('INSERT INTO telemetry VALUES(?,?,?,?,?,?,?,?,?,?)',(sid,ASSET,'operating_hours',iso(utcnow()+timedelta(seconds=seconds)),value,unit,quality,'demo-import','running',iso()))
    return sid


def check(c,p):
    r=c.post('/api/meter-plans/'+p['id']+'/generate');assert r.status_code==200,r.text
    return r.json()


def test_meter_exact_threshold_concurrent_and_real_work_lifecycle(planning):
    c=planning;p,_=setup(c);sample(100)
    user=dict(id='manager',plant_id='demo-plant',role='maintenance_manager')
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:generate(p['id'],user),range(2)))
    assert sorted(len(r['created_work_order_ids']) for r in results)==[0,1]
    work=c.get('/api/work-orders').json()[0];wid=work['id'];assert work['status']=='draft' and work['content']['risk_snapshot'] is None
    assert work['content']['preventive']['threshold']==100
    assert check(c,p)['created_work_order_ids']==[]
    assert c.post('/api/work-orders/'+wid+'/review',json=APPROVAL).status_code==200
    scheduled=proposed(c,[wid]).json();assert c.post('/api/schedules/'+scheduled['id']+'/approve',json=APPROVAL).status_code==200
    login(c,'engineer');assert change(c,wid,'start',site_clearance_confirmed=True).status_code==200
    assert change(c,wid,'report_fixed',1).status_code==200
    login(c,'manager');assert change(c,wid,'verify',2).status_code==200
    from backend.copilot.evidence import resolve
    with transaction() as conn:
        evidence=resolve(conn,user,ASSET,'work_order:'+wid)
        assert 'Meter preventive task at 100.0 h' in evidence['text']
        assert conn.execute('SELECT COUNT(*) FROM risk_assessments').fetchone()[0]==0


@pytest.mark.parametrize('case',['stale','future','bad','unit','absent'])
def test_invalid_latest_never_generates(planning,case):
    c=planning;p,_=setup(c)
    if case!='absent':sample(200,seconds=-400 if case=='stale' else 40 if case=='future' else 0,quality='bad' if case=='bad' else 'good',unit='cycles' if case=='unit' else 'h')
    assert check(c,p)['blocked']
    assert c.get('/api/work-orders').json()==[]


def test_observed_hidden_reset_blocks_until_attributed_reconciliation(planning):
    c=planning;p,_=setup(c);sample(90,-3);check(c,p)
    blocked=sample(10,-2);sample(150,-1)
    assert check(c,p)['blocked']
    p=c.get('/api/meter-plans').json()[0];assert p['blocked_sample']['id']==blocked and p['total']==90
    body=dict(expected_version=1,sample_id=blocked,action='reset',confirmed_total=100,reason='Verified synthetic counter replacement record at this sample.')
    assert c.post('/api/meter-plans/'+p['id']+'/reconcile',json={**body,'confirmed_total':80}).status_code==422
    login(c,'engineer');assert c.post('/api/meter-plans/'+p['id']+'/reconcile',json=body).status_code==403
    login(c,'manager');r=c.post('/api/meter-plans/'+p['id']+'/reconcile',json=body);assert r.status_code==200,r.text
    assert r.json()['total']==100 and r.json()['reconciliations'][0]['previous_total']==90
    assert c.post('/api/meter-plans/'+p['id']+'/reconcile',json=body).status_code==409
    assert len(check(c,p)['created_work_order_ids'])==2
    p=c.get('/api/meter-plans').json()[0];assert p['total']==240 and p['next_threshold']==300


def test_rollover_pause_revision_and_bounded_catchup(planning):
    c=planning;p,body=setup(c);sample(990,-2);assert len(check(c,p)['created_work_order_ids'])==9
    blocked=sample(5,-1);assert check(c,p)['blocked']
    r=c.post('/api/meter-plans/'+p['id']+'/reconcile',json=dict(expected_version=1,sample_id=blocked,action='rollover',modulus=1000,reason='Verified counter wraps at one thousand units in the test.'))
    assert r.status_code==200 and r.json()['total']==1005
    assert len(check(c,p)['created_work_order_ids'])==1
    state=dict(expected_version=2,enabled=False,reason='Pause during a synthetic shutdown interval.')
    assert c.post('/api/meter-plans/'+p['id']+'/state',json=state).status_code==200
    sample(3000);sweep();assert len(c.get('/api/work-orders').json())==10
    assert c.post('/api/meter-plans/'+p['id']+'/generate').status_code==409
    assert c.post('/api/meter-plans/'+p['id']+'/state',json={**state,'expected_version':3,'enabled':True}).status_code==200
    r=check(c,p);assert len(r['created_work_order_ids'])==20 and r['catch_up_pending']
    assert len(check(c,p)['created_work_order_ids'])==10
    assert c.put('/api/meter-plans/'+p['id'],json={**body,'expected_version':4,'interval':200}).status_code==422
    assert c.put('/api/meter-plans/'+p['id'],json={**body,'expected_version':4,'title':'Revised meter inspection'}).status_code==200
    assert check(c,p)['created_work_order_ids']==[]


@pytest.mark.parametrize('revocation',['disabled','role','asset','source','document'])
def test_sweep_rechecks_owner_and_provenance(planning,revocation):
    c=planning;p,_=setup(c);sample(200)
    with transaction() as conn:
        if revocation=='disabled':conn.execute("UPDATE users SET enabled=0 WHERE id='manager'")
        elif revocation=='role':conn.execute("UPDATE users SET role='operator' WHERE id='manager'")
        elif revocation=='asset':conn.execute("DELETE FROM asset_grants WHERE user_id='manager'")
        elif revocation=='source':conn.execute("DELETE FROM source_grants WHERE user_id='manager'")
        else:conn.execute("UPDATE documents SET status='revoked'")
    sweep()
    with transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM work_orders').fetchone()[0]==0
        assert conn.execute('SELECT last_error FROM meter_plans').fetchone()[0]


def test_source_revocation_hides_generated_work_and_plan(planning):
    c=planning;p,_=setup(c);sample(100);check(c,p)
    with transaction() as conn:conn.execute("DELETE FROM source_grants WHERE user_id='manager' AND source_id='demo-import'")
    assert c.get('/api/work-orders').json()==[] and c.get('/api/meter-plans').json()==[]
    assert c.post('/api/meter-plans/'+p['id']+'/generate').status_code==404


def test_scope_mapping_and_decimal_anchor(planning):
    c=planning;p,body=setup(c)
    assert c.get('/api/meter-plans',headers={'X-Maintenance-Workspace':'plant'}).json()==[]
    login(c,'engineer');assert c.post('/api/meter-plans',json=body).status_code==403
    assert c.post(f'/api/assets/{ASSET}/meters',json=dict(tag_id='cycles',unit='cycles',source_id='demo-import',maximum=1e6,stale_after_seconds=60,reason='Configure the test counter.')).status_code==403
    assert threshold(dict(first_due=0.1,interval=0.1),2)==0.3


def test_mapping_cannot_mix_demo_and_plant_origins(planning):
    c=planning;p,_=setup(c);login(c,'admin')
    with transaction() as conn:
        conn.execute("INSERT INTO sources(id,plant_id,name,kind,status,read_only) VALUES('physical-import','demo-plant','Plant import','file','configured',1)")
        conn.execute("INSERT INTO source_grants VALUES('admin','physical-import')")
    response=c.post(f'/api/assets/{ASSET}/meters',json=dict(tag_id='mixed_counter',unit='h',source_id='physical-import',maximum=1e6,stale_after_seconds=60,reason='This mixed workspace mapping must be rejected.'))
    assert response.status_code==422
    assert all(m['tag_id']!='mixed_counter' for m in c.get(f'/api/assets/{ASSET}/meters').json())


def test_mapping_change_cannot_reinterpret_an_existing_counter_cursor(planning):
    c=planning;p,body=setup(c);sample(90);check(c,p)
    with transaction() as conn:conn.execute("UPDATE sensors SET unit='cycles' WHERE id=?",(p['sensor_id'],))
    assert c.post('/api/meter-plans/'+p['id']+'/generate').status_code==409
    r=c.put('/api/meter-plans/'+p['id'],json={**body,'expected_version':1})
    assert r.status_code==422
    assert c.get('/api/work-orders').json()==[]
    with transaction() as conn:
        stored=conn.execute('SELECT total,version,template FROM meter_plans WHERE id=?',(p['id'],)).fetchone()
        assert stored['total']==90 and stored['version']==1
        import json
        assert json.loads(stored['template'])['unit']=='h'
