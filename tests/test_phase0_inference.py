from pathlib import Path
import json
import numpy as np
import pytest
import scipy.io
import torch
from fastapi.testclient import TestClient
from backend.main import app, plant_intel
from src.inference.rul_estimator import RULEstimator
from src.inference.fault_diagnoser import FaultDiagnoser
from src.inference.anomaly_detector import AudioAnomalyDetector

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def client():
    plant_intel.machines.clear(); plant_intel.history.clear()
    yield TestClient(app, raise_server_exceptions=False)
    plant_intel.machines.clear(); plant_intel.history.clear()

@pytest.mark.parametrize('kind',['predictor','transformer'])
@pytest.mark.parametrize('subset',['FD001','FD002','FD003','FD004'])
def test_real_rul_checkpoint(kind,subset):
    model=RULEstimator(str(ROOT/f'trained_models/rul_{kind}_{subset}.pth'))
    data=np.loadtxt(ROOT/f'Data/CMaps/test_{subset}.txt')
    cols=[2,3,4]+[4+i for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
    seq=data[data[:,0]==1][:,cols][-model.sequence_length:]
    result=model.predict(seq)
    assert np.isfinite(result['rul_cycles']) and result['rul_cycles']>=0
    assert 0<=result['health_score']<=1


def test_padding_matches_training():
    model=RULEstimator(str(ROOT/'trained_models/rul_predictor_FD001.pth'))
    seq=np.tile(model.scaler.mean_,(50,1))
    expected=np.pad(model.scaler.transform(seq),((30,0),(0,0)))
    np.testing.assert_allclose(model.preprocess(seq).cpu()[0],expected,atol=1e-6)


def test_vibration_preprocessing_matches_real_npz():
    data=np.load(ROOT/'Data/CWRU/CWRU_48k_load_1_CNN_data.npz')
    mapping={'Normal':'Time_Normal','Ball_007':'B007','Ball_014':'B014','Ball_021':'B021',
             'IR_007':'IR007','IR_014':'IR014','IR_021':'IR021','OR_007':'OR007','OR_014':'OR014','OR_021':'OR021'}
    model=FaultDiagnoser(str(ROOT/'trained_models/vibration_classifier.pth'))
    for label,pattern in mapping.items():
        mat=scipy.io.loadmat(next((ROOT/'Data/CWRU/raw').glob(pattern+'*.mat')))
        raw=next(v.flatten() for k,v in mat.items() if 'DE_time' in k)
        images=data['data'][data['labels']==label][:3]
        np.testing.assert_allclose(raw[:3072].reshape(3,32,32),images,atol=5.1e-5)
        for i, image in enumerate(images):
            actual=model.preprocess(raw[i*1024:(i+1)*1024]).cpu().numpy()[0,0]
            expected=(image-image.mean())/(image.std()+1e-8)
            np.testing.assert_allclose(actual,expected,atol=.001)
        assert model.predict(raw[:2048])['fault_type'] in model.class_labels


def test_vibration_http(client):
    mat=scipy.io.loadmat(ROOT/'Data/CWRU/raw/Time_Normal_1_098.mat')
    signal=next(v.flatten() for k,v in mat.items() if 'DE_time' in k)[:2048]
    r=client.post('/predict/vibration',json={'signal':signal.tolist()})
    assert r.status_code==200, r.text
    assert 0<=r.json()['confidence']<=1

@pytest.mark.parametrize('sequence',[[],[[]],[[1],[1,2]],[[0]*16]*80,[[0]*18]*80,[[0]*17]*10001])
def test_invalid_rul_shapes(client,sequence):
    assert client.post('/predict/rul',json={'sequence':sequence}).status_code==422

@pytest.mark.parametrize('value',['NaN','Infinity','-Infinity'])
def test_invalid_rul_numbers(client,value):
    body='{"sequence":[['+value+','+','.join(['0']*16)+']]}'
    assert client.post('/predict/rul',content=body,headers={'content-type':'application/json'}).status_code==422

@pytest.mark.parametrize('field,value',[('rul_cycles','nan'),('rul_cycles','inf'),('rul_cycles','-1'),('anomaly_score','nan'),('anomaly_score','2'),('fault_confidence','-1'),('machine_type','unsupported'),('fault_type','bogus')])
def test_invalid_updates_are_atomic(client,field,value):
    before=plant_intel.update_machine('test','pump',area='utilities',rul_cycles=25).to_dict()
    r=client.post('/plant/machines/test/update',params={field:value})
    assert r.status_code==422,r.text
    assert plant_intel.get_machine('test')==before
    assert client.get('/plant/machines').status_code==200


def test_partial_update_preserves_fields(client):
    before=plant_intel.update_machine('test','pump','utilities',.12,'Ball_014',.9,40).to_dict()
    r=client.post('/plant/machines/test/update',params={'rul_cycles':30})
    assert r.status_code==200,r.text
    for key in ['machine_type','area','anomaly_score','fault_type','fault_confidence']:assert r.json()[key]==before[key]
    assert r.json()['rul_cycles']==30


def test_unknown_machine(client):
    assert client.get('/plant/machines/unknown').status_code==404


def test_metrics_are_not_models():
    with pytest.raises(ValueError,match='Not a fitted'):
        AudioAnomalyDetector(str(ROOT/'trained_models/audio_advanced_v2_fan.pkl'))


def test_actual_rul_http(client):
    model=RULEstimator(str(ROOT/'trained_models/rul_predictor_FD001.pth'))
    r=client.post('/predict/rul',json={'sequence':np.tile(model.scaler.mean_,(50,1)).tolist()})
    assert r.status_code==200,r.text


def test_rul_checkpoint_snapshot_and_split():
    notebook=json.loads((ROOT/'notebooks/03_Colab_RUL_Prediction_Training.ipynb').read_text())
    source='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    import ast
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_subset')
    snapshots=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='best_model' for t in n.targets) and isinstance(n.value,ast.DictComp)]
    model=torch.nn.Linear(1,1)
    best=eval(compile(ast.Expression(snapshots[0].value),'<snapshot>','eval'),{'model':model})
    original=best['weight'].clone()
    with torch.no_grad():model.weight.add_(1)
    assert torch.equal(original,best['weight'])
    assert 'preds - y_val' in source and 'preds - rul_true' not in source
    assert "~train_df['unit'].isin(validation_ids)" in source

@pytest.mark.parametrize('machine',['fan','pump','valve'])
@pytest.mark.parametrize('section',['00','01','02'])
def test_trained_audio_http_and_score(client,machine,section):
    path=next((ROOT/f'Data/{machine}/source_test').glob(f'section_{section}_*normal*.wav'))
    model=AudioAnomalyDetector(str(ROOT/f'trained_models/audio_iforest_v1_{machine}.joblib'))
    expected=model.predict(str(path),section)
    with path.open('rb') as f:
        r=client.post(f'/predict/audio?machine_type={machine}&section={section}',files={'file':(path.name,f,'audio/wav')})
    assert r.status_code==200,r.text
    assert r.json()['anomaly_score']==expected[1]
    assert r.json()['is_anomaly']==expected[0]


def test_corrupt_audio_not_normal(client):
    r=client.post('/predict/audio',files={'file':('invalid.wav',b'corrupt recording','audio/wav')})
    assert r.status_code==422,r.text


def test_audio_split_provenance_and_artifacts():
    from scripts.model_artifacts import verify
    verify(ROOT/'trained_models')
    for machine in ['fan','pump','valve']:
        report=json.loads((ROOT/f'docs/evaluation/audio_{machine}.json').read_text())
        roles={}
        for row in report['records']:
            roles.setdefault(row['role'],set()).add(row['sha256'])
        assert len(report['records'])==4209
        assert not roles['train'] & roles['calibration']
        assert not (roles['train']|roles['calibration']) & (roles['source_test']|roles['target_test'])
        assert sum(x['train_count']+x['calibration_count'] for x in report['sections'].values())==3009


def test_artifact_roundtrip_and_corruption(tmp_path):
    from scripts.model_artifacts import export_bundle,import_bundle,verify
    bundle=tmp_path/'models.tar.gz'; destination=tmp_path/'installed'
    export_bundle(ROOT/'trained_models',bundle)
    import_bundle(bundle,destination)
    verify(destination)
    next(destination.glob('*.pth')).write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'):verify(destination)


def test_real_startup_and_websocket():
    with TestClient(app) as client:
        assert client.get('/').json()['models_loaded']['total']==12
        assert len(client.get('/plant/machines').json())==14
        with client.websocket_connect('/ws/updates') as ws:
            assert ws.receive_json()['type']=='connected'
            ws.send_json({'type':'ping'})
            assert ws.receive_json()['type']=='pong'
            client.post('/plant/machines/PLA-FAN-001/update',params={'rul_cycles':0})
            assert ws.receive_json()['type']=='machine_update'
            assert ws.receive_json()['type']=='plant_summary'


def test_rul_training_evidence_uses_disjoint_engines():
    from scripts.verify_rul_training import run
    run()
    checkpoint=torch.load(ROOT/'artifacts/rul_training_smoke/rul_transformer_FD001.pth',map_location='cpu',weights_only=False)
    assert len(checkpoint['training_engine_ids'])==80
    assert len(checkpoint['validation_engine_ids'])==20
    assert not set(checkpoint['training_engine_ids']) & set(checkpoint['validation_engine_ids'])
    assert checkpoint['architecture']=='transformer'
    report=json.loads((ROOT/'docs/evaluation/rul_training_smoke.json').read_text())
    assert report['epochs']==1 and report['runtime_reload_verified']


def test_reported_metrics_have_artifact_and_dataset_identity():
    from scripts.model_artifacts import sha256
    report=json.loads((ROOT/'docs/evaluation/legacy_baseline.json').read_text())
    assert report['vibration']['artifact_sha256']==sha256(ROOT/'trained_models/vibration_classifier.pth')
    assert report['vibration']['count']==920
    assert len(report['rul'])==8
    readme=(ROOT/'README.md').read_text()
    for model,metrics in report['rul'].items():
        assert metrics['artifact_sha256']==sha256(ROOT/f'trained_models/{model}.pth')
    assert '> 0.96' not in readme
    assert f"{report['rul']['rul_predictor_FD001']['rmse_cycles']:.4f}" in readme
    assert f"{report['rul']['rul_transformer_FD001']['rmse_cycles']:.4f}" in readme
