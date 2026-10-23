from copy import deepcopy
from datetime import timedelta
import json
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.core.database import transaction,migrate,iso,utcnow
from backend.core.demo import seed_demo
from backend.ml.registry import initialize_registry
from backend.risk.service import seed_risk,DEFAULT_PROFILE,DEFAULT_POLICY
from backend.risk.engine import assess

@pytest.fixture
def risk(tmp_path,monkeypatch):
    monkeypatch.setenv('MAINTENANCEAI_DB',str(tmp_path/'risk.db'));monkeypatch.setenv('MAINTENANCEAI_MODE','demo')
    migrate();seed_demo();initialize_registry();seed_risk()
    c=TestClient(app,raise_server_exceptions=False)
    token=c.post('/api/auth/login',json={'username':'admin','password':'admin-demo-password'}).json()['access_token']
    c.headers['Authorization']='Bearer '+token
    return c


def readings(temp=78,vibration=7.2,current=28,now=None):
    return [dict(sensor={'tag_id':tag},effective_quality='good',observation=dict(id=tag,value=value,unit=unit,timestamp_utc=iso(now),source='demo-simulator'))
            for tag,value,unit in [('temperature',temp,'degC'),('vibration_rms',vibration,'mm/s'),('current',current,'A')]]


def model(now=None,**changes):
    result=dict(id='prediction',model_id='synthetic-v1',family='synthetic',status='SUCCEEDED',historical=False,signal_timestamp=iso(now),timestamp_utc=iso(now),
                data_mode='simulation',input_id='signal',drift={'flagged':False},source_id='demo-simulator',
                output={'is_anomaly':True,'anomaly_score':1.,'fault_class':'bearing_impulse_like'})
    result.update(changes);return result


def test_reproducible_and_criticality_outweighs_anomaly():
    now=utcnow();low=dict(DEFAULT_PROFILE,safety=1,production=1,maintenance=1,damage=1);high=dict(DEFAULT_PROFILE,safety=4,production=5,maintenance=4,damage=4)
    a=assess(low,DEFAULT_POLICY,readings(now=now),[model(now)],now)
    b=assess(high,DEFAULT_POLICY,readings(now=now),[model(now,output={'is_anomaly':True,'anomaly_score':.96,'fault_class':'imbalance_like'})],now)
    assert b['score']>a['score'] and b['rank']<a['rank']
    assert b==assess(high,DEFAULT_POLICY,readings(now=now),[model(now,output={'is_anomaly':True,'anomaly_score':.96,'fault_class':'imbalance_like'})],now)
    assert b['human_review_required'] and b['confidence']!='calibrated'


def test_missing_critical_sensor_does_not_report_low_risk():
    now=utcnow();rows=readings(temp=40,vibration=1,current=10,now=now);rows[0]['effective_quality']='stale'
    r=assess(DEFAULT_PROFILE,DEFAULT_POLICY,rows,[],now)
    assert r['priority']=='Insufficient evidence' and r['score'] is None
    assert r['missing_evidence']==['temperature: stale'] and r['human_review_required']


def test_safety_escalation_even_with_other_missing_channel():
    now=utcnow();rows=readings(temp=110,now=now);rows[1]['observation']=None
    r=assess(dict(DEFAULT_PROFILE,safety=5),DEFAULT_POLICY,rows,[],now)
    assert r['priority']=='Critical review required' and r['escalation']
    assert r['score'] is None and r['missing_evidence']


def test_invalid_latest_model_does_not_resurrect_old_output():
    now=utcnow();bad=model(now,drift={'flagged':True})
    r=assess(DEFAULT_PROFILE,DEFAULT_POLICY,readings(temp=45,vibration=1,current=12,now=now),[bad,model(now-timedelta(seconds=1))],now)
    assert r['model_output'] is None and r['priority']=='Monitor'
    assert any('distribution warning' in s for s in r['explanation'])
    for p in [model(now,historical=True),model(now,status='MODEL_ERROR'),model(now-timedelta(minutes=5))]:
        assert assess(DEFAULT_PROFILE,DEFAULT_POLICY,readings(now=now),[p],now)['model_output'] is None


def test_api_immutable_evidence_policy_versions_and_ranking(risk):
    assert risk.post('/api/assets/DEMO-CONVEYOR-001/demo-prediction',json={}).status_code==200
    r=risk.post('/api/risks/evaluate',json={'asset_ids':['PLA-FAN-001','DEMO-CONVEYOR-001']})
    assert r.status_code==200,r.text
    result=r.json();assert result[0]['asset_id']=='DEMO-CONVEYOR-001' and result[0]['priority']=='High priority'
    saved=risk.get('/api/assets/DEMO-CONVEYOR-001/risk').json()
    assert saved['id']==result[0]['id'] and len(saved['evidence'])==4
    policy=risk.get('/api/risks/policy').json();config=policy['config'];config['high_score']=80;config['reason']='Review alternative planning threshold'
    changed=risk.put('/api/risks/policy',json=config)
    assert changed.status_code==200 and changed.json()['id']!=policy['id']
    assert risk.get('/api/assets/DEMO-CONVEYOR-001/risk').json()['stale']
    with transaction() as conn:
        prior=json.loads(conn.execute('SELECT result FROM risk_assessments WHERE id=?',(saved['id'],)).fetchone()[0])
    assert prior['policy_id']==policy['id'] and prior['priority']=='High priority'


def test_role_source_and_asset_isolation(risk):
    risk.post('/api/assets/DEMO-CONVEYOR-001/demo-prediction',json={})
    assert risk.post('/api/risks/evaluate',json={'asset_ids':['DEMO-CONVEYOR-001']}).status_code==200
    with transaction() as conn:conn.execute("DELETE FROM source_grants WHERE user_id='admin' AND source_id='demo-simulator'")
    assert risk.get('/api/assets/DEMO-CONVEYOR-001/risk').json() is None
    result=risk.post('/api/risks/evaluate',json={'asset_ids':['DEMO-CONVEYOR-001']}).json()[0]
    assert result['priority']=='Insufficient evidence' and result['evidence']==[]
    with transaction() as conn:conn.execute("DELETE FROM asset_grants WHERE user_id='admin' AND asset_id='DEMO-CONVEYOR-001'")
    assert risk.get('/api/assets/DEMO-CONVEYOR-001/risk').status_code==404
    assert risk.post('/api/risks/evaluate',json={'asset_ids':['DEMO-CONVEYOR-001']}).status_code==404
    token=risk.post('/api/auth/login',json={'username':'engineer','password':'engineer-demo-password'}).json()['access_token'];risk.headers['Authorization']='Bearer '+token
    assert risk.put('/api/risks/policy',json=DEFAULT_POLICY).status_code==403
    assert risk.put('/api/assets/DEMO-CONVEYOR-001/criticality',json=DEFAULT_PROFILE).status_code==403


def test_config_validation_limits_and_dependency_cycles(risk):
    invalid=dict(DEFAULT_POLICY,safety_weight=.9)
    assert risk.put('/api/risks/policy',json=invalid).status_code==422
    profile=deepcopy(DEFAULT_PROFILE);profile['limits']['temperature']['unit']='K'
    assert risk.put('/api/assets/DEMO-CONVEYOR-001/criticality',json=profile).status_code==422
    profile=deepcopy(DEFAULT_PROFILE);profile['dependencies']=['DEMO-CONVEYOR-001']
    assert risk.put('/api/assets/PLA-PUMP-001/criticality',json=profile).status_code==422
    profile['dependencies']=['unapproved-asset']
    assert risk.put('/api/assets/PLA-PUMP-001/criticality',json=profile).status_code==404
    profile=deepcopy(DEFAULT_PROFILE);profile['safety']=5;profile['reason']='Human review of critical motor consequences'
    result=risk.put('/api/assets/DEMO-CONVEYOR-001/criticality',json=profile)
    assert result.status_code==200 and result.json()['config']['safety']==5
    with transaction() as conn:
        assert conn.execute("SELECT count(*) FROM asset_profiles WHERE asset_id='DEMO-CONVEYOR-001'").fetchone()[0]==2
        assert conn.execute("SELECT count(*) FROM audit_events WHERE action='update_criticality'").fetchone()[0]==1


def test_cross_asset_evaluation_is_atomic(risk):
    response=risk.post('/api/risks/evaluate',json={'asset_ids':['DEMO-CONVEYOR-001','forbidden']})
    assert response.status_code==404
    with transaction() as conn:assert conn.execute('SELECT count(*) FROM risk_assessments').fetchone()[0]==0


def test_operating_conditions_and_snapshot_expiration(risk,monkeypatch):
    now=utcnow();rows=readings(now=now)
    for r in rows:r['observation']['operating_state']='stopped'
    assert assess(DEFAULT_PROFILE,DEFAULT_POLICY,rows,[],now)['priority']=='Insufficient evidence'
    response=risk.post('/api/risks/evaluate',json={'asset_ids':['DEMO-CONVEYOR-001']})
    assert response.status_code==200,response.text
    monkeypatch.setattr('backend.risk.service.utcnow',lambda:now+timedelta(seconds=31))
    assert risk.get('/api/assets/DEMO-CONVEYOR-001/risk').json()['stale']


def test_authored_risk_benchmark():
    from scripts.evaluate_risk_scenarios import run
    report=run()
    assert report['cases']==8 and report['failures']==[]
    assert report['comparisons']['risk_based']['order']==['critical-heat','production-bottleneck','ancillary-nuisance']


def test_rejected_latest_observation_requires_verification(risk):
    now=utcnow()
    r=risk.post('/api/ingestion/demo-import/records',json={'records':[dict(asset_id='DEMO-CONVEYOR-001',tag_id='temperature',timestamp_utc=iso(now),value=9999,unit='degC',quality='good',source='demo-import',operating_state='running')]})
    assert r.json()['results'][0]['status']=='rejected'
    result=risk.post('/api/risks/evaluate',json={'asset_ids':['DEMO-CONVEYOR-001']}).json()[0]
    assert result['priority']=='Insufficient evidence'
    assert any('latest ingestion rejected' in e for e in result['missing_evidence'])


def test_revoked_compatibility_binding_excludes_model(risk):
    risk.post('/api/assets/DEMO-CONVEYOR-001/demo-prediction',json={})
    with transaction() as conn:conn.execute("DELETE FROM model_bindings WHERE asset_id='DEMO-CONVEYOR-001'")
    result=risk.post('/api/risks/evaluate',json={'asset_ids':['DEMO-CONVEYOR-001']}).json()[0]
    assert result['model_output'] is None


def test_confirmed_high_priority_precedes_unestablished_assets(risk):
    risk.post('/api/assets/DEMO-CONVEYOR-001/demo-prediction',json={})
    with transaction() as conn:conn.execute("DELETE FROM telemetry WHERE asset_id='PLA-FAN-001'")
    response=risk.post('/api/risks/evaluate',json={'asset_ids':['PLA-FAN-001','DEMO-CONVEYOR-001']})
    assert response.status_code==200,response.text
    assert [r['priority'] for r in response.json()]==['High priority','Insufficient evidence']
