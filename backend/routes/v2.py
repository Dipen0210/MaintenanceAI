"""Authenticated industrial data foundation. No industrial write operations."""
import asyncio,hashlib,json,os
from fastapi import APIRouter,Depends,File,HTTPException,Query,Request,Response,UploadFile,WebSocket,WebSocketDisconnect
from pydantic import BaseModel,Field,ConfigDict
from starlette.requests import HTTPConnection
from backend.core.database import transaction,audit,iso,utcnow
from backend.core.security import authenticate,principal,require_roles,allowed_source_ids,allowed_assets,authorize_asset,authorize_source
from backend.ingestion.service import ingest,latest_telemetry
from backend.ingestion.adapters import CSVAdapter,ParquetAdapter
from backend.ingestion.responses import UserResponse,LoginResponse,LogoutResponse,AssetResponse,SourceResponse,TelemetryResponse,ImportResponse,AlarmResponse

router=APIRouter(prefix='/api',tags=['industrial data'])


class Login(BaseModel):
    username:str=Field(min_length=1,max_length=100)
    password:str=Field(min_length=1,max_length=256)


class ImportRecords(BaseModel):
    records:list[dict]=Field(min_length=1,max_length=10000)


@router.post('/auth/login',response_model=LoginResponse)
def login(body:Login,response:Response):
    token,user=authenticate(body.username,body.password)
    response.set_cookie('maintenanceai_session',token,httponly=True,samesite='strict',
                        secure=os.getenv('MAINTENANCEAI_MODE','demo')!='demo',max_age=28800)
    return {'user':user,'access_token':token,'token_type':'bearer'}


@router.get('/auth/me',response_model=UserResponse)
def me(user=Depends(principal)):return user


@router.post('/auth/logout',response_model=LogoutResponse)
def logout(request:Request,response:Response,user=Depends(principal)):
    authorization=request.headers.get('authorization','')
    token=authorization[7:] if authorization.startswith('Bearer ') else request.cookies.get('maintenanceai_session','')
    with transaction() as conn:
        conn.execute('DELETE FROM sessions WHERE token_hash=?',(hashlib.sha256(token.encode()).hexdigest(),))
        audit(conn,user,'logout',user['id'],'allowed')
    response.delete_cookie('maintenanceai_session')
    return {'status':'logged_out'}


@router.get('/assets',response_model=list[AssetResponse])
def assets(user=Depends(principal)):
    with transaction() as conn:
        rows=allowed_assets(conn,user)
        from backend.ingestion.condition import describe_asset
        return [describe_asset(conn,user,asset) for asset in rows]



@router.get('/assets/{asset_id}',response_model=AssetResponse)
def asset(asset_id:str,user=Depends(principal)):
    from backend.ingestion.condition import describe_asset
    with transaction() as conn:return describe_asset(conn,user,authorize_asset(conn,user,asset_id))


@router.get('/assets/{asset_id}/telemetry',response_model=TelemetryResponse)
def telemetry(asset_id:str,limit:int=Query(120,ge=1,le=1000),user=Depends(principal)):
    with transaction() as conn:
        current=latest_telemetry(conn,user,asset_id)
        sources=allowed_source_ids(conn,user);slots=','.join('?' for _ in sources)
        history=[dict(r) for r in conn.execute(f'SELECT * FROM telemetry WHERE asset_id=? AND source IN ({slots}) ORDER BY timestamp_utc DESC LIMIT ?',(asset_id,*sources,limit))]
        audit(conn,user,'read_telemetry',asset_id,'allowed')
        return {'asset_id':asset_id,'latest':current,'history':list(reversed(history)),'server_time':iso()}


@router.get('/assets/{asset_id}/alarms',response_model=list[AlarmResponse])
def alarms(asset_id:str,user=Depends(principal)):
    with transaction() as conn:
        authorize_asset(conn,user,asset_id)
        return [dict(r) for r in conn.execute('SELECT a.* FROM alarms a JOIN source_grants g ON a.source=g.source_id WHERE a.asset_id=? AND g.user_id=? ORDER BY a.timestamp_utc DESC',(asset_id,user['id']))]


@router.get('/integrations/status',response_model=list[SourceResponse])
def integration_status(user=Depends(principal)):
    with transaction() as conn:
        rows=[dict(r) for r in conn.execute('''SELECT s.* FROM sources s JOIN source_grants g ON s.id=g.source_id
            WHERE g.user_id=? AND s.plant_id=?''',(user['id'],user['plant_id']))]
        from backend.core.workspaces import matches_source
        rows=[r for r in rows if matches_source(user,r)]
        for row in rows:
            if row['kind'] in ['simulation','gateway','opcua','historian'] and row['last_received'] and row['status'] not in ['paused','revoked']:
                from datetime import datetime
                interval=conn.execute('SELECT MIN(stale_after_seconds) FROM sensors WHERE source_id=?',(row['id'],)).fetchone()[0] or 30
                if (utcnow()-datetime.fromisoformat(row['last_received'])).total_seconds()>interval:row['status']='stale'
        return rows


@router.post('/ingestion/{source_id}/records',response_model=ImportResponse)
def import_records(source_id:str,body:ImportRecords,user=Depends(require_roles('administrator'))):
    with transaction() as conn:
        source=authorize_source(conn,user,source_id)
        if source['kind']!='file':raise HTTPException(422,'Only file sources accept manual imports')
    return {'results':ingest(body.records,user,source_id)}


@router.post('/ingestion/{source_id}/file',response_model=ImportResponse)
def import_file(source_id:str,file:UploadFile=File(...),user=Depends(require_roles('administrator'))):
    with transaction() as conn:
        source=authorize_source(conn,user,source_id)
        if source['kind']!='file':raise HTTPException(422,'Source does not accept file imports')
    content=file.file.read(10*1024*1024+1)
    if len(content)>10*1024*1024:raise HTTPException(413,'File exceeds 10 MiB')
    name=(file.filename or '').lower()
    try:
        if name.endswith('.csv'):rows=CSVAdapter(content).records()
        elif name.endswith('.parquet'):rows=ParquetAdapter(content).records()
        else:raise ValueError('Expected CSV or Parquet')
        if not rows:raise ValueError('No observations in file')
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    except RuntimeError as exc:raise HTTPException(503,str(exc)) from exc
    return {'results':ingest(rows,user,source_id)}


@router.websocket('/ws/telemetry')
async def telemetry_socket(ws:WebSocket):
    # Cookies are accepted only from the same approved application origins.
    if ws.headers.get('origin') not in ['http://localhost:3000','http://127.0.0.1:3000']:
        await ws.close(code=1008);return
    token=ws.cookies.get('maintenanceai_session')
    if not token:await ws.close(code=1008);return
    from backend.core.workspaces import parse_workspace
    try:workspace=parse_workspace(ws.query_params.get('workspace'))
    except HTTPException:
        await ws.close(code=1008);return
    await ws.accept()
    try:
        while True:
            # Re-check expiry, revocation, user state and grants on every push.
            with transaction() as conn:
                row=conn.execute('''SELECT u.id,u.username,u.role,u.plant_id FROM users u JOIN sessions s ON s.user_id=u.id
                    WHERE s.token_hash=? AND s.expires_at>? AND u.enabled=1''',(hashlib.sha256(token.encode()).hexdigest(),iso())).fetchone()
                if row is None:await ws.close(code=1008);return
                user=dict(row)
                if workspace:user['workspace']=workspace
                payload=[{'asset':a,'telemetry':latest_telemetry(conn,user,a['id'])} for a in allowed_assets(conn,user)]
            await ws.send_json({'type':'telemetry_snapshot','timestamp_utc':iso(),'assets':payload})
            await asyncio.sleep(5)
    except (WebSocketDisconnect,RuntimeError):return
