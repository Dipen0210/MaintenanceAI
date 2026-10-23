"""Local model registry equivalent: immutable identities and audited activation/rollback."""
from functools import lru_cache
import hashlib,json,os
from fastapi import HTTPException
from backend.core.database import ROOT,transaction,identifier,iso,audit
from backend.core.security import authorize_asset


def catalog():
    entries=[]
    baseline=json.loads((ROOT/'docs/evaluation/legacy_baseline.json').read_text())
    for artifact in json.loads((ROOT/'models/manifest.json').read_text())['artifacts']:
        name=artifact['filename'];digest=artifact['sha256']
        e=dict(filename=name,sha256=digest,version=digest[:12],operating_states=['running'],mode='reference',
               uncertainty='No validated uncertainty interval. Class scores, if present, are uncalibrated.',
               limitations='Reference dataset only; no evidence of transfer to real plant machinery.')
        if name.startswith('audio_'):
            machine=name.split('_')[-1].split('.')[0]
            e.update(family='mimii-'+machine,kind='anomaly',dataset_id='MIMII-'+machine,
                     preprocessing_version='whole_file_logmel_mfcc_v1',equipment_types=[machine],input_kind='audio',sample_rate=16000,active=True,
                     metrics={'report':f'docs/evaluation/audio_{machine}.json','selection':'Training/calibration split; weak test discrimination documented'})
        elif name.startswith('rul_'):
            dataset=name.split('_')[-1].split('.')[0]
            e.update(family='cmapss-'+dataset,kind='rul',dataset_id='C-MAPSS-'+dataset,
                     preprocessing_version='cmapss-17features-scaled-leftpad-v1',equipment_types=['turbofan_reference'],input_kind='sequence',sample_rate=None,
                     active='predictor' in name,metrics=baseline['rul'][name[:-4]])
        else:
            e.update(family='cwru-bearing',kind='fault',dataset_id='CWRU-48k-load1',
                     preprocessing_version='cwru1024-rowmajor-zscore-v1',equipment_types=['bearing_reference'],input_kind='vibration',sample_rate=48000,active=True,metrics=baseline['vibration'])
        e['id']=e['family']+'-'+digest[:12];entries.append(e)
    synthetic=ROOT/'models/synthetic-registry.json'
    if synthetic.exists():entries.extend(json.loads(synthetic.read_text()))
    return entries


def initialize_registry():
    with transaction() as conn:
        for e in catalog():
            # A restart never overwrites explicit activation or rollback decisions.
            conn.execute('INSERT OR IGNORE INTO model_versions VALUES(?,?,?,?,?,?)',
                         (e['id'],e['family'],e['version'],e['kind'],json.dumps(e),int(e['active'] and not conn.execute('SELECT 1 FROM model_versions WHERE family=? AND active=1',(e['family'],)).fetchone())))
        if os.getenv('MAINTENANCEAI_MODE','demo')=='demo' and conn.execute("SELECT 1 FROM assets WHERE id='DEMO-CONVEYOR-001'").fetchone():
            conn.execute('INSERT OR IGNORE INTO model_bindings VALUES(?,?,?)',('DEMO-CONVEYOR-001','conveyor-synthetic','synthetic-conveyor-v1'))


def verified_path(metadata):
    path=ROOT/'trained_models'/metadata['filename']
    if path.parent != ROOT/'trained_models' or not path.is_file():raise ValueError('Verified model artifact unavailable')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['sha256']:raise ValueError('Model artifact integrity mismatch')
    return path


@lru_cache(maxsize=20)
def load_verified(model_id,digest):
    entry=next((e for e in catalog() if e['id']==model_id and e['sha256']==digest),None)
    if entry is None:raise ValueError('Model identity is not in the trusted catalog')
    path=verified_path(entry)
    if entry['kind']=='condition':
        import joblib
        return joblib.load(path)
    if entry['kind']=='anomaly':
        from src.inference.anomaly_detector import AudioAnomalyDetector
        return AudioAnomalyDetector(str(path))
    if entry['kind']=='rul':
        from src.inference.rul_estimator import RULEstimator
        return RULEstimator(str(path),device='cpu')
    from src.inference.fault_diagnoser import FaultDiagnoser
    return FaultDiagnoser(str(path),device='cpu')


def active_id(conn,family,plant_id):
    row=conn.execute('SELECT model_id FROM plant_model_selections WHERE plant_id=? AND family=?',(plant_id,family)).fetchone()
    if row:return row['model_id']
    row=conn.execute('SELECT id FROM model_versions WHERE family=? AND active=1',(family,)).fetchone()
    return row['id'] if row else None


def activate(model_id,user,reason):
    if user['role'] not in ['administrator','reliability_engineer']:raise HTTPException(403,'Model activation requires reliability or administrator role')
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE')
        actor=conn.execute('SELECT role,plant_id,enabled FROM users WHERE id=?',(user['id'],)).fetchone()
        if not actor or not actor['enabled'] or actor['plant_id']!=user['plant_id'] or actor['role'] not in ['administrator','reliability_engineer']:
            raise HTTPException(403,'Current identity cannot change model selection')
        row=conn.execute('SELECT * FROM model_versions WHERE id=?',(model_id,)).fetchone()
        if not row:raise HTTPException(404,'Model version not found')
        for asset in conn.execute('SELECT a.id FROM assets a JOIN model_bindings b ON a.id=b.asset_id WHERE a.plant_id=? AND b.family=?',(user['plant_id'],row['family'])).fetchall():
            authorize_asset(conn,user,asset['id'])
        metadata=json.loads(row['metadata'])
        try:verified_path(metadata)
        except ValueError as exc:raise HTTPException(409,str(exc)) from exc
        previous=active_id(conn,row['family'],user['plant_id'])
        conn.execute('INSERT INTO plant_model_selections VALUES(?,?,?) ON CONFLICT(plant_id,family) DO UPDATE SET model_id=excluded.model_id',(user['plant_id'],row['family'],model_id))
        event=identifier()
        conn.execute('INSERT INTO model_activations VALUES(?,?,?,?,?,?,?,?)',(event,row['family'],previous,model_id,user['id'],iso(),reason,user['plant_id']))
        audit(conn,user,'activate_model',model_id,reason)
        return {'activation_id':event,'previous_id':previous,'active_id':model_id}
