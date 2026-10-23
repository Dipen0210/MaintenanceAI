"""Machine onboarding and a source-scoped inbound telemetry gateway.

Credentials permit measurement ingestion only, never equipment control or UI access.
"""
import hashlib,hmac,json,secrets
from datetime import timedelta
from fastapi import APIRouter,Depends,Header,HTTPException,Response
from pydantic import Field
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.core.security import authorize_source
from backend.routes.administration import admin,scope,AssetInput,SensorInput,create_asset
from backend.scheduling.contracts import Strict
from backend.risk.contracts import Criticality
from backend.ingestion.service import ingest
from backend.ingestion.responses import ImportResponse

router=APIRouter(prefix='/api',tags=['machine onboarding'])


class NewSensor(SensorInput):
    source_id:str=''


class Machine(Strict):
    id:str=Field(min_length=3,max_length=100,pattern=r'^[a-zA-Z0-9_.-]+$')
    name:str=Field(min_length=3,max_length=120)
    equipment_type:str=Field(min_length=2,max_length=80)
    area:str=Field(min_length=2,max_length=100)
    sensors:list[NewSensor]=Field(min_length=1,max_length=20)
    criticality:Criticality
    reason:str=Field(min_length=10,max_length=500)


class GatewayBatch(Strict):
    records:list[dict]=Field(min_length=1,max_length=1000)


def credential(conn,user,source_id):
    token='mai_gateway_'+secrets.token_urlsafe(32);expires=iso(utcnow()+timedelta(days=90))
    conn.execute('INSERT INTO gateway_credentials VALUES(?,?,?,?,?,0) ON CONFLICT(source_id) DO UPDATE SET token_hash=excluded.token_hash,owner_id=excluded.owner_id,created_at=excluded.created_at,expires_at=excluded.expires_at,revoked=0',
                 (source_id,hashlib.sha256(token.encode()).hexdigest(),user['id'],iso(),expires))
    return dict(token=token,expires_at=expires)


def gateway_scope(conn,user,source_id):
    scope(conn,user);source=authorize_source(conn,user,source_id)
    if source['kind']!='gateway':raise HTTPException(422,'An inbound machine gateway is required')
    return source


def setup(conn,source_id):
    mappings=[dict(r) for r in conn.execute('SELECT asset_id,tag_id,unit,lower_bound,upper_bound,stale_after_seconds FROM sensors WHERE source_id=? ORDER BY tag_id',(source_id,))]
    state=conn.execute('SELECT expires_at,revoked FROM gateway_credentials WHERE source_id=?',(source_id,)).fetchone()
    return dict(source_id=source_id,endpoint=f'/api/gateways/{source_id}/telemetry',mappings=mappings,
                credential_expires_at=state['expires_at'] if state else None,credential_revoked=bool(state['revoked']) if state else True)


@router.post('/admin/machines')
def onboard(body:Machine,response:Response,user=Depends(admin)):
    response.headers['Cache-Control']='no-store'
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope(conn,user)
        sid=identifier()
        conn.execute("INSERT INTO sources VALUES(?,?,?,'gateway',1,'awaiting data',NULL)",(sid,user['plant_id'],body.name+' gateway'))
        conn.execute('INSERT INTO source_grants VALUES(?,?)',(user['id'],sid))
        data=body.model_dump();data['sensors']=[{**s,'source_id':sid} for s in data['sensors']]
        create_asset(conn,AssetInput.model_validate(data),user)
        key=credential(conn,user,sid)
        audit(conn,user,'register_machine_gateway',sid,body.reason+'; awaiting actual measurements')
        return dict(asset_id=body.id,**setup(conn,sid),**key)


@router.get('/admin/gateways/{source_id}')
def gateway_setup(source_id:str,user=Depends(admin)):
    with transaction() as conn:
        gateway_scope(conn,user,source_id)
        return setup(conn,source_id)


@router.post('/admin/gateways/{source_id}/credential')
def rotate(source_id:str,response:Response,user=Depends(admin)):
    response.headers['Cache-Control']='no-store'
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');gateway_scope(conn,user,source_id)
        key=credential(conn,user,source_id)
        conn.execute("UPDATE sources SET status='awaiting data' WHERE id=?",(source_id,))
        audit(conn,user,'rotate_gateway_credential',source_id,'previous credential invalidated')
        return {**setup(conn,source_id),**key}


@router.delete('/admin/gateways/{source_id}/credential')
def revoke(source_id:str,user=Depends(admin)):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');gateway_scope(conn,user,source_id)
        conn.execute('UPDATE gateway_credentials SET revoked=1 WHERE source_id=?',(source_id,))
        conn.execute("UPDATE sources SET status='revoked' WHERE id=?",(source_id,))
        audit(conn,user,'revoke_gateway_credential',source_id,'ingestion disabled')
        return {'status':'revoked'}


@router.post('/gateways/{source_id}/telemetry',response_model=ImportResponse)
def receive(source_id:str,body:GatewayBatch,authorization:str=Header(default='')):
    if not authorization.startswith('Bearer mai_gateway_'):raise HTTPException(401,'Gateway credential required')
    digest=hashlib.sha256(authorization[7:].encode()).hexdigest()
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row=conn.execute('''SELECT k.*,u.username,u.role,u.plant_id,u.enabled FROM gateway_credentials k
                            JOIN users u ON k.owner_id=u.id WHERE k.source_id=?''',(source_id,)).fetchone()
        if not row or not hmac.compare_digest(row['token_hash'],digest) or row['revoked'] or row['expires_at']<=iso() or not row['enabled'] or row['role']!='administrator':
            raise HTTPException(401,'Gateway credential expired, revoked or invalid')
        user=dict(id=row['owner_id'],username=row['username'],plant_id=row['plant_id'],role=row['role'])
        source=authorize_source(conn,user,source_id)
        if source['kind']!='gateway' or source['read_only']!=1:raise HTTPException(403,'Read-only measurement gateway required')
        # Check the complete batch before ingesting anything; a key cannot target a
        # different machine even if its administrator can access that machine.
        mappings={(r['asset_id'],r['tag_id']) for r in conn.execute('SELECT asset_id,tag_id FROM sensors WHERE source_id=?',(source_id,))}
        for record in body.records:
            if not isinstance(record.get('asset_id'),str) or not isinstance(record.get('tag_id'),str) or record.get('source')!=source_id or (record['asset_id'],record['tag_id']) not in mappings:
                raise HTTPException(403,'Observation is outside this gateway mapping')
        return {'results':ingest(body.records,user,source_id,connection=conn)}
