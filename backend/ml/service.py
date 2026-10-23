"""Authorized, dataset-scoped predictive services with immutable evidence and monitoring."""
from datetime import datetime,timedelta
import hashlib,json,time
import numpy as np
from fastapi import HTTPException
from backend.core.database import ROOT,transaction,identifier,iso,utcnow,audit,database_path
from backend.core.security import authorize_asset,authorize_source
from backend.ml.registry import verified_path,load_verified,active_id
from backend.ml.synthetic import features,spectrum,FEATURES,waveform_summary


def store_signal(asset_id,body,user):
    metadata=body.model_dump(mode='json',exclude={'values'})
    content=json.dumps(body.values,allow_nan=False).encode()
    return save_signal(asset_id,body.source_id,body.kind,metadata,content,user)


def save_signal(asset_id,source_id,kind,metadata,content,user):
    signal_id=identifier();path=database_path().parent/'signals'/signal_id
    end=datetime.fromisoformat(metadata['timestamp_utc'])
    duration=metadata.get('duration_seconds')
    if duration is None and kind=='vibration':duration=len(json.loads(content))/metadata['sample_rate']
    metadata.update(end_timestamp_utc=iso(end),start_timestamp_utc=iso(end-timedelta(seconds=duration)) if duration else None)
    metadata['preprocessing_metadata']='Original signal preserved; each prediction records its model-specific preprocessing version.'
    with transaction() as conn:
        authorize_asset(conn,user,asset_id);source=authorize_source(conn,user,source_id)
        if source['kind'] not in ['file','simulation']:raise HTTPException(403,'Gateway signals require the approved gateway collector')
        if source['kind']=='simulation' and metadata['dataset_id']!='synthetic-conveyor-v1':raise HTTPException(422,'Simulator source only supports its synthetic dataset')
        timestamp=datetime.fromisoformat(metadata['timestamp_utc'])
        if (timestamp-utcnow()).total_seconds()>30:raise HTTPException(422,'Future signal timestamp')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        digest=hashlib.sha256(content).hexdigest()
        try:
            conn.execute('INSERT INTO signal_inputs VALUES(?,?,?,?,?,?,?,?,?)',(signal_id,asset_id,source_id,kind,metadata['timestamp_utc'],json.dumps(metadata),str(path),digest,iso()))
            audit(conn,user,'store_signal',signal_id,'accepted')
        except BaseException:path.unlink(missing_ok=True);raise
    return {'id':signal_id,'asset_id':asset_id,'source_id':source_id,'kind':kind,'sha256':digest,'metadata':metadata}


def distribution(values,mean,std):
    z=np.abs((np.asarray(values)-np.asarray(mean))/np.maximum(np.asarray(std),1e-6))
    return {'method':'reference_standardized_distance_v1','max_z':float(np.max(z)),
            'fraction_over_3':float(np.mean(z>3)),'flagged':bool(np.mean(z>3)>.25),
            'interpretation':'Input distribution warning, not a calibrated failure probability or physical cause.'}


def run_model(meta,input_row,input_meta,content):
    # Verify bytes on every execution, including cached model instances.
    verified_path(meta);model=load_verified(meta['id'],meta['sha256'])
    path=input_row['content_path'];kind=meta['kind'];output={};evidence={};drift={'method':'unavailable','flagged':False,'reason':'No validated reference distribution'}
    if kind=='anomaly':
        section=input_meta.get('section','00')
        anomaly,score,_=model.predict(path,section)
        output={'anomaly_score':score,'is_anomaly':anomaly,'score_interpretation':'Normal calibration percentile, not fault probability'}
        x=model.extract_features(path);scaler=model.model_data['sections'][section]['pipeline'].named_steps['scaler']
        drift=distribution(x,scaler.mean_,scaler.scale_)
        evidence={'feature_count':len(x),'feature_version':meta['preprocessing_version']}
    else:
        x=np.asarray(json.loads(content),dtype=float)
        if kind=='condition':
            f=features(x);raw=float(-model['anomaly'].score_samples(f[None,:])[0]);score=float(np.mean(model['calibration']<=raw))
            scores=model['classifier'].predict_proba(f[None,:])[0];labels=model['classifier'].classes_;best=int(np.argmax(scores))
            output={'anomaly_score':score,'is_anomaly':score>.95,'fault_class':str(labels[best]),'class_scores':dict(zip(labels,map(float,scores))),
                    'score_interpretation':'Uncalibrated classifier scores and normal anomaly percentile. Synthetic domain only.',
                    'health_score_proxy':round(1-score,3)}
            evidence={'signal_features':dict(zip(FEATURES,map(float,f))),'spectrum_peaks':spectrum(x,input_meta['sample_rate'])}
            drift=distribution(f,model['reference_mean'],model['reference_std'])
        elif kind=='fault':
            import torch
            tensor=model.preprocess(x)
            with torch.no_grad():scores=torch.softmax(model.model(tensor),dim=1).mean(0).cpu().numpy()
            if not np.isfinite(scores).all():raise ValueError('Non-finite model output')
            output={'fault_class':str(model.class_labels[int(np.argmax(scores))]),'class_scores':dict(zip(map(str,model.class_labels),map(float,scores))),
                    'score_interpretation':'Uncalibrated softmax scores, not validated class probabilities'}
            evidence={'rms':float(np.sqrt(np.mean(x*x))),'spectrum_peaks':spectrum(x,input_meta['sample_rate'])}
            reference=json.loads((ROOT/'models/cwru-reference.json').read_text())
            drift=distribution(waveform_summary(x),reference['mean'],reference['std'])
        else:
            if input_meta.get('feature_names')!=list(model.feature_cols):raise ValueError('Feature order differs from checkpoint')
            result=model.predict(x)
            output={'rul':result['rul_cycles'],'units':'cycles','health_score_proxy':result['health_score']}
            if model.scaler is not None:drift=distribution(x.mean(0),model.scaler.mean_,model.scaler.scale_)
            evidence={'feature_count':x.shape[1],'timesteps':len(x),'feature_columns':model.feature_cols}
    return output,evidence,drift


def predict(asset_id,body,user):
    started=time.perf_counter();now=utcnow();pid=identifier()
    with transaction() as conn:
        asset=authorize_asset(conn,user,asset_id)
        row=conn.execute('SELECT * FROM model_versions WHERE family=? AND id=?',
                         (body.family,body.model_id or active_id(conn,body.family,user['plant_id']))).fetchone()
        if not row:raise HTTPException(404,'Model family/version unavailable')
        meta=json.loads(row['metadata']);binding=conn.execute('SELECT * FROM model_bindings WHERE asset_id=? AND family=?',(asset_id,body.family)).fetchone()
        plant=conn.execute('SELECT mode FROM plants WHERE id=?',(asset['plant_id'],)).fetchone()
        input_row=None;input_meta=None
        if body.input_id:
            found=conn.execute('SELECT * FROM signal_inputs WHERE id=? AND asset_id=?',(body.input_id,asset_id)).fetchone()
            if not found:raise HTTPException(404,'Signal not found or not authorized')
            input_row=dict(found);authorize_source(conn,user,input_row['source_id']);input_meta=json.loads(input_row['metadata'])
        status='SUCCEEDED';reason=None;output={};evidence={};drift=None
        if not binding or binding['dataset_id']!=meta['dataset_id'] or asset['equipment_type'] not in meta['equipment_types'] or (meta['mode']=='simulation' and plant['mode']!='simulation'):
            status='UNSUPPORTED_ASSET';reason='No approved asset/dataset/model compatibility binding'
        elif asset['operating_state'] not in meta['operating_states']:
            status='UNSUPPORTED_STATE';reason='Asset operating state is outside model scope'
        elif not input_row:
            status='MISSING_INPUT';reason='A referenced compatible signal is required'
        elif input_meta['dataset_id']!=meta['dataset_id'] or input_row['kind']!=meta['input_kind'] or input_meta.get('sample_rate')!=meta['sample_rate']:
            status='UNSUPPORTED_INPUT';reason='Dataset, signal kind or sampling rate is incompatible'
        elif input_meta['quality']!='good' or input_meta['operating_state'] not in meta['operating_states']:
            status='INVALID_INPUT';reason='Signal quality/state is unsuitable'
        elif not input_meta['historical'] and (now-datetime.fromisoformat(input_row['timestamp_utc'])).total_seconds()>60:
            status='STALE_INPUT';reason='Signal exceeds 60-second inference freshness limit'
        else:
            try:
                from pathlib import Path
                content=Path(input_row['content_path']).read_bytes()
                if hashlib.sha256(content).hexdigest()!=input_row['sha256']:raise ValueError('Signal integrity mismatch')
                output,evidence,drift=run_model(meta,input_row,input_meta,content)
            except (ValueError,RuntimeError,KeyError,OSError) as exc:
                status='MODEL_ERROR';reason=type(exc).__name__+': compatible model or input validation failed'
        result={'id':pid,'asset_id':asset_id,'model_id':row['id'],'model_version':row['version'],'family':body.family,
                'dataset_id':meta['dataset_id'],'preprocessing_version':meta['preprocessing_version'],'status':status,'reason':reason,
                'timestamp_utc':iso(now),'input_id':body.input_id,'input_quality':input_meta['quality'] if input_meta else 'missing',
                'historical':input_meta.get('historical',False) if input_meta else False,'signal_timestamp':input_row['timestamp_utc'] if input_row else None,
                'output':output,'evidence':evidence,'drift':drift,'uncertainty_interval':None,'uncertainty':meta['uncertainty'],
                'limitations':meta['limitations'],'data_mode':meta['mode'],'latency_ms':round((time.perf_counter()-started)*1000,3)}
        conn.execute('INSERT INTO predictions VALUES(?,?,?,?,?,?,?,?,?)',(pid,asset_id,row['id'],body.input_id,iso(now),status,json.dumps(result,allow_nan=False),result['latency_ms'],user['id']))
        audit(conn,user,'predict',pid,status)
        return result


def history(conn,user,asset_id,limit=50):
    authorize_asset(conn,user,asset_id)
    return [json.loads(r['result']) for r in conn.execute('''SELECT p.result FROM predictions p LEFT JOIN signal_inputs s ON p.input_id=s.id
        WHERE p.asset_id=? AND (p.input_id IS NULL OR s.source_id IN (SELECT source_id FROM source_grants WHERE user_id=?))
        ORDER BY p.timestamp_utc DESC LIMIT ?''',(asset_id,user['id'],limit))]
