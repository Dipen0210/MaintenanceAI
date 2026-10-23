"""Dataset-scoped prediction and registry endpoints; no industrial controls."""
from datetime import datetime,timezone
import io,json,os
from typing import Literal
from fastapi import APIRouter,Depends,HTTPException,File,UploadFile,Query
from pydantic import BaseModel,Field
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.core.security import principal,require_roles,authorize_asset,authorize_source,allowed_assets
from backend.ml.contracts import SignalInput,PredictionRequest,ActivationRequest,PredictionResponse,SignalResponse,ModelResponse,ActivationResponse,ActivationEvent,BindingsResponse,MonitoringResponse
from backend.ml.service import store_signal,save_signal,predict,history
from backend.ml.registry import activate,verified_path,active_id
from backend.ml.synthetic import signal

router=APIRouter(prefix='/api',tags=['predictive ML'])


@router.get('/ml/models',response_model=list[ModelResponse])
def models(user=Depends(principal)):
    with transaction() as conn:
        rows=[]
        for row in conn.execute('SELECT * FROM model_versions ORDER BY family,id'):
            item=json.loads(row['metadata']);item['active']=row['id']==active_id(conn,row['family'],user['plant_id'])
            try:verified_path(item);item['available']=True
            except ValueError:item['available']=False
            rows.append(item)
        return rows


@router.post('/ml/models/{model_id}/activate',response_model=ActivationResponse)
def activation(model_id:str,body:ActivationRequest,user=Depends(require_roles('administrator','reliability_engineer'))):
    return activate(model_id,user,body.reason)


@router.get('/ml/activations',response_model=list[ActivationEvent])
def activations(user=Depends(require_roles('administrator','reliability_engineer'))):
    with transaction() as conn:return [dict(r) for r in conn.execute('SELECT * FROM model_activations WHERE plant_id=? ORDER BY timestamp_utc DESC LIMIT 100',(user['plant_id'],))]


@router.get('/assets/{asset_id}/models',response_model=BindingsResponse)
def asset_models(asset_id:str,user=Depends(principal)):
    with transaction() as conn:
        asset=authorize_asset(conn,user,asset_id)
        return {'asset_id':asset_id,'equipment_type':asset['equipment_type'],
                'bindings':[dict(r) for r in conn.execute('SELECT * FROM model_bindings WHERE asset_id=?',(asset_id,))],
                'rul_notice':'C-MAPSS is only supported for explicitly registered turbofan reference datasets.'}


@router.post('/assets/{asset_id}/signals',response_model=SignalResponse)
def signals(asset_id:str,body:SignalInput,user=Depends(require_roles('administrator','engineer','reliability_engineer'))):
    return store_signal(asset_id,body,user)


@router.post('/assets/{asset_id}/signals/audio',response_model=SignalResponse)
def audio(asset_id:str,timestamp_utc:datetime,source_id:str='demo-import',dataset_id:str='MIMII-fan',
          sensor_position:str=Query('microphone',min_length=1,max_length=100),section:Literal['00','01','02']='00',
          historical:bool=False,quality:Literal['good','bad','uncertain']='good',
          file:UploadFile=File(...),user=Depends(require_roles('administrator','engineer','reliability_engineer'))):
    import soundfile as sf
    if timestamp_utc.tzinfo is None:raise HTTPException(422,'UTC-aware timestamp required')
    content=file.file.read(20*1024*1024+1)
    if not content or len(content)>20*1024*1024:raise HTTPException(413,'Audio must contain 1 byte to 20 MiB')
    try:
        info=sf.info(io.BytesIO(content))
        if info.format!='WAV' or not 1<=info.duration<=60:raise ValueError('Expected 1–60 second WAV')
    except Exception as exc:raise HTTPException(422,'Invalid WAV recording') from exc
    metadata={'timestamp_utc':iso(timestamp_utc.astimezone(timezone.utc)),'sample_rate':info.samplerate,'sensor_position':sensor_position,
              'dataset_id':dataset_id,'section':section,'historical':historical,'quality':quality,'operating_state':'running',
              'original_channels':info.channels,'duration_seconds':info.duration,'preprocessing':'Preserve original; model converts to mono 16 kHz'}
    return save_signal(asset_id,source_id,'audio',metadata,content,user)


@router.post('/assets/{asset_id}/predictions',response_model=PredictionResponse)
def predictions(asset_id:str,body:PredictionRequest,user=Depends(require_roles('administrator','engineer','reliability_engineer'))):
    return predict(asset_id,body,user)


@router.get('/assets/{asset_id}/predictions',response_model=list[PredictionResponse])
def prediction_history(asset_id:str,limit:int=Query(50,ge=1,le=100),user=Depends(principal)):
    with transaction() as conn:return history(conn,user,asset_id,limit)


@router.get('/ml/monitoring',response_model=MonitoringResponse)
def monitoring(user=Depends(principal)):
    with transaction() as conn:
        results=[r for a in allowed_assets(conn,user) for r in history(conn,user,a['id'],100)]
        latencies=sorted(r['latency_ms'] for r in results)
        return {'observations':len(results),'failures':sum(r['status']!='SUCCEEDED' for r in results),
                'distribution_warnings':sum(bool(r.get('drift') and r['drift']['flagged']) for r in results),
                'latency_p95_ms':latencies[min(len(latencies)-1,int(len(latencies)*.95))] if latencies else None,
                'scope':'Latest 100 predictions per authorized asset and data source',
                'calibration':'No validated predictive uncertainty intervals; classifier scores are uncalibrated.'}


class DemoRun(BaseModel):
    state:Literal['normal','imbalance_like','bearing_impulse_like']='bearing_impulse_like'
    seed:int=Field(default=50000,ge=40000,le=1000000)


@router.post('/assets/{asset_id}/demo-prediction',response_model=PredictionResponse)
def demo_prediction(asset_id:str,body:DemoRun,user=Depends(require_roles('administrator','engineer','reliability_engineer'))):
    with transaction() as conn:
        asset=authorize_asset(conn,user,asset_id)
        plant=conn.execute('SELECT mode FROM plants WHERE id=?',(asset['plant_id'],)).fetchone()
        if os.getenv('MAINTENANCEAI_MODE','demo')!='demo' or plant['mode']!='simulation' or asset_id!='DEMO-CONVEYOR-001':
            raise HTTPException(403,'Synthetic demonstration is restricted to the demo conveyor')
    now=utcnow()
    body_signal=SignalInput(source_id='demo-simulator',kind='vibration',timestamp_utc=now,sample_rate=4096,
                            sensor_position='synthetic drive-end bearing',dataset_id='synthetic-conveyor-v1',
                            values=signal(body.seed,body.state).tolist())
    stored=store_signal(asset_id,body_signal,user)
    # Time-aligned process observations use the same explicit simulator source.
    from backend.ingestion.service import ingest
    ingest([dict(asset_id=asset_id,tag_id=tag,timestamp_utc=iso(now),value=value,unit=unit,quality='good',source='demo-simulator',operating_state='running')
            for tag,value,unit in [('temperature',78 if body.state!='normal' else 45,'degC'),('vibration_rms',7.2 if body.state!='normal' else 1.5,'mm/s'),('current',28 if body.state!='normal' else 12,'A')]],user,'demo-simulator')
    result=predict(asset_id,PredictionRequest(family='conveyor-synthetic',input_id=stored['id']),user)
    with transaction() as conn:
        authorize_asset(conn,user,asset_id);authorize_source(conn,user,'demo-simulator')
        conn.execute("UPDATE alarms SET state='cleared' WHERE asset_id=? AND source='demo-simulator' AND state='active'",(asset_id,))
        alarm_id=identifier();active=body.state!='normal'
        message='Synthetic scenario source event: '+body.state+'. This simulated alarm is separate from the trained prediction and does not establish a physical fault.'
        conn.execute('INSERT INTO alarms VALUES(?,?,?,?,?,?,?)',(alarm_id,asset_id,'warning' if active else 'info','demo-simulator','active' if active else 'cleared',iso(now),message))
        audit(conn,user,'simulate_alarm',alarm_id,'Scenario event only; no industrial alarm or equipment control')
    return result
