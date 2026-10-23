from datetime import timedelta
import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.core.database import migrate,transaction,iso,utcnow,ROOT
from backend.core.demo import seed_demo
from backend.ml.registry import initialize_registry,catalog
from backend.ml.synthetic import signal


@pytest.fixture
def ml(tmp_path,monkeypatch):
    monkeypatch.setenv('MAINTENANCEAI_DB',str(tmp_path/'ml.db'));monkeypatch.setenv('MAINTENANCEAI_MODE','demo')
    migrate();seed_demo();initialize_registry()
    c=TestClient(app,raise_server_exceptions=False)
    token=c.post('/api/auth/login',json={'username':'admin','password':'admin-demo-password'}).json()['access_token']
    c.headers['Authorization']='Bearer '+token
    return c


def store(c,asset='DEMO-CONVEYOR-001',**changes):
    data=dict(source_id='demo-simulator',kind='vibration',timestamp_utc=iso(),sample_rate=4096,sensor_position='synthetic bearing',
              dataset_id='synthetic-conveyor-v1',values=signal(50000,'bearing_impulse_like').tolist())
    data.update(changes)
    r=c.post(f'/api/assets/{asset}/signals',json=data)
    assert r.status_code==200,r.text
    return r.json()['id']


def prediction(c,input_id=None,family='conveyor-synthetic',asset='DEMO-CONVEYOR-001',**other):
    r=c.post(f'/api/assets/{asset}/predictions',json={'family':family,'input_id':input_id,**other})
    assert r.status_code==200,r.text
    return r.json()


def reference_asset(asset,equipment,family,dataset):
    with transaction() as conn:
        conn.execute('INSERT INTO assets VALUES(?,?,?,?,?,?,?,?)',(asset,'demo-plant',asset,equipment,'reference','running',1,1))
        conn.execute('INSERT INTO asset_grants VALUES(?,?)',('admin',asset))
        conn.execute('INSERT INTO model_bindings VALUES(?,?,?)',(asset,family,dataset))


def test_real_synthetic_model_and_evidence(ml):
    r=ml.post('/api/assets/DEMO-CONVEYOR-001/demo-prediction',json={'state':'bearing_impulse_like','seed':50000})
    assert r.status_code==200,r.text
    result=r.json()
    assert result['status']=='SUCCEEDED' and result['output']['is_anomaly']
    assert result['output']['fault_class']=='bearing_impulse_like'
    assert len(result['evidence']['spectrum_peaks'])==5 and result['model_version']
    assert result['uncertainty_interval'] is None and result['data_mode']=='simulation'
    history=ml.get('/api/assets/DEMO-CONVEYOR-001/predictions').json()
    assert history[0]['id']==result['id']
    telemetry=ml.get('/api/assets/DEMO-CONVEYOR-001/telemetry').json()
    assert all(v['observation']['timestamp_utc']==result['signal_timestamp'] for v in telemetry['latest'])


def test_cmapss_never_applied_to_conveyor_or_pump(ml):
    for asset in ['DEMO-CONVEYOR-001','PLA-PUMP-001']:
        r=prediction(ml,family='cmapss-FD001',asset=asset)
        assert r['status']=='UNSUPPORTED_ASSET' and r['output']=={}
    assert prediction(ml,family='mimii-pump',asset='PLA-PUMP-001')['status']=='UNSUPPORTED_ASSET'


@pytest.mark.parametrize('changes,status',[
    ({'quality':'bad'},'INVALID_INPUT'),({'operating_state':'stopped'},'INVALID_INPUT'),
    ({'timestamp_utc':iso(utcnow()-timedelta(minutes=3))},'STALE_INPUT'),({'sample_rate':48000},'UNSUPPORTED_INPUT')])
def test_quality_and_applicability(ml,changes,status):
    assert prediction(ml,store(ml,**changes))['status']==status


def test_model_input_corruption_and_drift(ml):
    input_id=store(ml)
    with transaction() as conn:
        path=conn.execute('SELECT content_path FROM signal_inputs WHERE id=?',(input_id,)).fetchone()[0]
    Path(path).write_text('[1]')
    assert prediction(ml,input_id)['status']=='MODEL_ERROR'
    shifted=store(ml,values=(signal(50001,'bearing_impulse_like')*100).tolist())
    result=prediction(ml,shifted)
    assert result['status']=='SUCCEEDED' and result['drift']['flagged']
    monitor=ml.get('/api/ml/monitoring').json()
    assert monitor['observations']==2 and monitor['failures']==1 and monitor['distribution_warnings']==1


def test_missing_input_flat_signal_and_operating_state(ml):
    assert prediction(ml)['status']=='MISSING_INPUT'
    assert prediction(ml,store(ml,values=[0.0]*4096))['status']=='MODEL_ERROR'
    with transaction() as conn:conn.execute("UPDATE assets SET operating_state='stopped' WHERE id='DEMO-CONVEYOR-001'")
    assert prediction(ml)['status']=='UNSUPPORTED_STATE'


def test_registry_activation_rollback_and_restart(ml):
    versions=[m for m in ml.get('/api/ml/models').json() if m['family']=='conveyor-synthetic']
    old=next(m['id'] for m in versions if m['active']);new=next(m['id'] for m in versions if not m['active'])
    r=ml.post(f'/api/ml/models/{new}/activate',json={'reason':'Evaluate registered challenger'})
    assert r.json()['previous_id']==old
    initialize_registry()
    assert prediction(ml,store(ml))['model_id']==new
    assert ml.post(f'/api/ml/models/{old}/activate',json={'reason':'Rollback to evaluated champion'}).status_code==200
    assert prediction(ml,store(ml))['model_id']==old
    assert len(ml.get('/api/ml/activations').json())==2
    token=ml.post('/api/auth/login',json={'username':'engineer','password':'engineer-demo-password'}).json()['access_token']
    ml.headers['Authorization']='Bearer '+token
    assert ml.post(f'/api/ml/models/{new}/activate',json={'reason':'Unauthorized activation'}).status_code==403


def test_prediction_and_monitoring_source_isolation(ml):
    prediction(ml,store(ml))
    with transaction() as conn:conn.execute("DELETE FROM source_grants WHERE user_id='admin' AND source_id='demo-simulator'")
    assert ml.get('/api/assets/DEMO-CONVEYOR-001/predictions').json()==[]
    assert ml.get('/api/ml/monitoring').json()['observations']==0
    with transaction() as conn:conn.execute("DELETE FROM asset_grants WHERE user_id='admin' AND asset_id='DEMO-CONVEYOR-001'")
    assert ml.get('/api/assets/DEMO-CONVEYOR-001/models').status_code==404
    assert ml.post('/api/assets/DEMO-CONVEYOR-001/demo-prediction',json={}).status_code==404


def test_actual_cwru_reference_prediction(ml):
    reference_asset('CWRU-REF','bearing_reference','cwru-bearing','CWRU-48k-load1')
    x=np.load(ROOT/'Data/CWRU/CWRU_48k_load_1_CNN_data.npz')['data'][0].reshape(-1)
    stored=store(ml,asset='CWRU-REF',source_id='demo-import',dataset_id='CWRU-48k-load1',sample_rate=48000,values=x.tolist(),historical=True)
    result=prediction(ml,stored,'cwru-bearing','CWRU-REF')
    assert result['status']=='SUCCEEDED' and len(result['output']['class_scores'])==10
    assert 'uncalibrated' in result['output']['score_interpretation'].lower()
    assert result['drift']['method']=='reference_standardized_distance_v1'


def test_actual_rul_reference_versions(ml):
    reference_asset('CMAPSS-REF','turbofan_reference','cmapss-FD001','C-MAPSS-FD001')
    from src.inference.rul_estimator import RULEstimator
    import pandas as pd
    frame=pd.read_csv(ROOT/'Data/CMaps/test_FD001.txt',sep=r'\s+',header=None)
    frame.columns=['unit','cycle','op1','op2','op3']+[f's{i}' for i in range(1,22)]
    # Stored checkpoint feature names match the training notebook's feature order.
    checkpoint=RULEstimator(str(ROOT/'trained_models/rul_predictor_FD001.pth'))
    data=frame[frame['unit']==1].iloc[:,[2,3,4]+[5+i-1 for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]].to_numpy()
    stored=store(ml,asset='CMAPSS-REF',source_id='demo-import',kind='sequence',dataset_id='C-MAPSS-FD001',sample_rate=None,values=data.tolist(),historical=True,feature_names=list(checkpoint.feature_cols))
    versions=[m for m in catalog() if m['family']=='cmapss-FD001']
    for version in versions:
        result=prediction(ml,stored,'cmapss-FD001','CMAPSS-REF',model_id=version['id'])
        assert result['status']=='SUCCEEDED',result
        assert result['output']['rul']>=0 and result['output']['units']=='cycles'
        assert result['uncertainty_interval'] is None


def test_actual_audio_reference_and_original_metadata(ml):
    reference_asset('MIMII-REF','fan','mimii-fan','MIMII-fan')
    report=json.loads((ROOT/'docs/evaluation/audio_fan.json').read_text())
    wav=ROOT/next(r['path'] for r in report['records'] if r['role']=='source_test' and r['section']=='00')
    r=ml.post('/api/assets/MIMII-REF/signals/audio',params={'timestamp_utc':iso(),'historical':'true'},files={'file':('signal.wav',wav.read_bytes(),'audio/wav')})
    assert r.status_code==200,r.text
    assert r.json()['metadata']['original_channels']>=1
    result=prediction(ml,r.json()['id'],'mimii-fan','MIMII-REF')
    assert result['status']=='SUCCEEDED',result
    assert 0<=result['output']['anomaly_score']<=1 and result['drift']


def test_model_integrity_failure_is_recorded(ml,monkeypatch):
    def corrupt(meta):raise ValueError('Model artifact integrity mismatch')
    monkeypatch.setattr('backend.ml.service.verified_path',corrupt)
    result=prediction(ml,store(ml))
    assert result['status']=='MODEL_ERROR' and result['output']=={}


def test_missing_or_reordered_channels_rejected(ml):
    body=dict(source_id='demo-import',kind='sequence',timestamp_utc=iso(),sample_rate=None,sensor_position='engine',dataset_id='C-MAPSS-FD001',values=[[0]*17])
    assert ml.post('/api/assets/DEMO-CONVEYOR-001/signals',json=body).status_code==422
    body['feature_names']=['sensor_2']*17
    assert ml.post('/api/assets/DEMO-CONVEYOR-001/signals',json=body).status_code==422
    body['values']=[[0]*16]
    assert ml.post('/api/assets/DEMO-CONVEYOR-001/signals',json=body).status_code==422
