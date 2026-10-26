"""Local identity and registry administration; never connects to or controls equipment."""
import json,sqlite3
from typing import Literal
from zoneinfo import ZoneInfo,ZoneInfoNotFoundError
from fastapi import APIRouter,Depends,HTTPException
from pydantic import Field,FiniteFloat,SecretStr,model_validator,field_validator
from backend.core.database import transaction,identifier,iso,audit
from backend.core.security import require_roles,authorize_asset,authorize_source,hash_password,allowed_assets
from backend.copilot.documents import authorize_document_review
from backend.scheduling.contracts import Strict
from backend.scheduling.service import role
from backend.risk.contracts import Criticality
from backend.ingestion.responses import AssetResponse,SensorResponse

router=APIRouter(prefix='/api/admin',tags=['local administration'])
admin=require_roles('administrator')
Roles=Literal['administrator','engineer','maintenance_manager','reliability_engineer','operator','operations_coordinator','stores_coordinator']


class Access(Strict):
    role:Roles
    enabled:bool=True
    asset_ids:list[str]=Field(max_length=1000)
    source_ids:list[str]=Field(max_length=100)
    document_ids:list[str]=Field(default_factory=list,max_length=1000)
    reason:str=Field(min_length=10,max_length=500)


class CreateUser(Access):
    username:str=Field(min_length=3,max_length=80,pattern=r'^[a-zA-Z0-9_.-]+$')
    password:SecretStr=Field(min_length=14,max_length=256)


class Identity(Strict):
    id:str
    username:str
    role:Roles
    enabled:bool
    asset_ids:list[str]
    source_ids:list[str]
    document_ids:list[str]


class SourceInput(Strict):
    name:str=Field(min_length=3,max_length=120)
    reason:str=Field(min_length=10,max_length=500)


class SensorInput(Strict):
    tag_id:str=Field(min_length=1,max_length=100)
    unit:str=Field(min_length=1,max_length=30)
    lower_bound:FiniteFloat
    upper_bound:FiniteFloat
    stale_after_seconds:int=Field(ge=1,le=86400)
    source_id:str

    @model_validator(mode='after')
    def bounds(self):
        if self.lower_bound>=self.upper_bound:raise ValueError('Sensor bounds must increase')
        return self


class AssetInput(Strict):
    id:str=Field(min_length=3,max_length=100,pattern=r'^[a-zA-Z0-9_.-]+$')
    name:str=Field(min_length=3,max_length=120)
    equipment_type:str=Field(min_length=2,max_length=80)
    area:str=Field(min_length=2,max_length=100)
    sensors:list[SensorInput]=Field(min_length=1,max_length=20)
    criticality:Criticality
    reason:str=Field(min_length=10,max_length=500)


class AssetMetadata(Strict):
    name:str=Field(min_length=3,max_length=120)
    area:str=Field(min_length=2,max_length=100)
    reason:str=Field(min_length=10,max_length=500)


class PlantInput(Strict):
    name:str=Field(min_length=3,max_length=120)
    location:str=Field(min_length=3,max_length=200)
    display_timezone:str=Field(min_length=1,max_length=80)
    reason:str=Field(min_length=10,max_length=500)

    @field_validator('display_timezone')
    @classmethod
    def timezone_known(cls,value):
        try:ZoneInfo(value)
        except (ZoneInfoNotFoundError,ValueError) as exc:raise ValueError('Use an IANA timezone, such as UTC or America/Chicago') from exc
        return value


class PlantResponse(Strict):
    id:str
    name:str
    mode:str
    location:str
    configuration:dict


class ResourceID(Strict):
    id:str


class MappingResponse(AssetResponse):
    sensors:list[SensorResponse]
    bindings:list[dict]


class AuditResponse(Strict):
    id:str
    actor_id:str|None
    plant_id:str|None
    action:str
    resource:str
    timestamp_utc:str
    outcome:str


def scope(conn,user):
    role(conn,user,['administrator'])
    # A registry administrator must see the full registry before replacing user grants.
    all_ids={r['id'] for r in conn.execute('SELECT id FROM assets WHERE plant_id=?',(user['plant_id'],))}
    if all_ids!={a['id'] for a in allowed_assets(conn,user)}:raise HTTPException(403,'Whole-plant asset access is required for registry administration')


def identity(conn,row):
    value={k:row[k] for k in ['id','username','role','enabled']}
    for key,table,column,resources in [('asset_ids','asset_grants','asset_id','assets'),('source_ids','source_grants','source_id','sources'),('document_ids','document_grants','document_id','documents')]:
        value[key]=[r[0] for r in conn.execute(f'SELECT g.{column} FROM {table} g JOIN {resources} r ON r.id=g.{column} WHERE g.user_id=? AND r.plant_id=? ORDER BY g.{column}',(row['id'],row['plant_id']))]
    return value


def validate_grants(conn,user,body):
    for aid in set(body.asset_ids):authorize_asset(conn,user,aid)
    for sid in set(body.source_ids):authorize_source(conn,user,sid)
    for did in set(body.document_ids):authorize_document_review(conn,user,did)


def write_grants(conn,uid,body):
    for values,table,column in [(body.asset_ids,'asset_grants','asset_id'),(body.source_ids,'source_grants','source_id'),(body.document_ids,'document_grants','document_id')]:
        conn.execute(f'DELETE FROM {table} WHERE user_id=?',(uid,))
        conn.executemany(f'INSERT INTO {table}(user_id,{column}) VALUES(?,?)',[(uid,value) for value in set(values)])


def plant_view(conn,user):
    row=conn.execute('SELECT p.id,p.name,p.mode,s.location,s.configuration FROM plants p LEFT JOIN plant_settings s ON s.plant_id=p.id WHERE p.id=?',(user['plant_id'],)).fetchone()
    return dict(id=row['id'],name=row['name'],mode=row['mode'],location=row['location'] or 'Unconfigured',configuration=json.loads(row['configuration'] or '{"display_timezone":"UTC"}'))


@router.get('/plant',response_model=PlantResponse)
def plant(user=Depends(admin)):
    with transaction() as conn:scope(conn,user);return plant_view(conn,user)


@router.put('/plant',response_model=PlantResponse)
def configure_plant(body:PlantInput,user=Depends(admin)):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope(conn,user)
        conn.execute('UPDATE plants SET name=? WHERE id=?',(body.name,user['plant_id']))
        conn.execute('INSERT INTO plant_settings VALUES(?,?,?,?,?) ON CONFLICT(plant_id) DO UPDATE SET location=excluded.location,configuration=excluded.configuration,updated_at=excluded.updated_at,actor_id=excluded.actor_id',
                     (user['plant_id'],body.location,json.dumps({'display_timezone':body.display_timezone}),iso(),user['id']))
        audit(conn,user,'configure_plant_metadata',user['plant_id'],body.reason)
        return plant_view(conn,user)


@router.get('/identities',response_model=list[Identity])
def users(user=Depends(admin)):
    with transaction() as conn:
        scope(conn,user)
        return [identity(conn,row) for row in conn.execute('SELECT id,username,role,enabled,plant_id FROM users WHERE plant_id=? ORDER BY username',(user['plant_id'],)).fetchall()]


@router.post('/identities',response_model=Identity)
def create_user(body:CreateUser,user=Depends(admin)):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope(conn,user);validate_grants(conn,user,body)
        uid=identifier()
        try:conn.execute('INSERT INTO users VALUES(?,?,?,?,?,?)',(uid,body.username,hash_password(body.password.get_secret_value()),body.role,user['plant_id'],int(body.enabled)))
        except sqlite3.IntegrityError as exc:raise HTTPException(409,'Username is unavailable') from exc
        write_grants(conn,uid,body);audit(conn,user,'create_user',uid,body.reason)
        return identity(conn,conn.execute('SELECT * FROM users WHERE id=?',(uid,)).fetchone())


@router.put('/identities/{uid}/access',response_model=Identity)
def access(uid:str,body:Access,user=Depends(admin)):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope(conn,user)
        if not conn.execute('SELECT 1 FROM users WHERE id=? AND plant_id=?',(uid,user['plant_id'])).fetchone():raise HTTPException(404,'User not found')
        if uid==user['id']:raise HTTPException(409,'Use a separate authorized administrator to change your own access')
        validate_grants(conn,user,body)
        conn.execute('UPDATE users SET role=?,enabled=? WHERE id=?',(body.role,int(body.enabled),uid));write_grants(conn,uid,body)
        conn.execute('DELETE FROM sessions WHERE user_id=?',(uid,))
        audit(conn,user,'change_user_access',uid,body.reason+'; existing sessions revoked')
        return identity(conn,conn.execute('SELECT * FROM users WHERE id=?',(uid,)).fetchone())


@router.post('/sources',response_model=ResourceID)
def create_source(body:SourceInput,user=Depends(admin)):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope(conn,user);sid=identifier()
        conn.execute("INSERT INTO sources VALUES(?,?,?,'file',1,'awaiting import',NULL)",(sid,user['plant_id'],body.name))
        conn.execute('INSERT INTO source_grants VALUES(?,?)',(user['id'],sid));audit(conn,user,'register_file_source',sid,body.reason)
        return {'id':sid}


@router.get('/mappings',response_model=list[MappingResponse])
def mappings(user=Depends(admin)):
    with transaction() as conn:
        scope(conn,user);rows=[]
        for asset in allowed_assets(conn,user):
            sensors=[dict(s) for s in conn.execute('''SELECT s.* FROM sensors s JOIN source_grants g ON g.source_id=s.source_id WHERE s.asset_id=? AND g.user_id=?''',(asset['id'],user['id']))]
            rows.append({**asset,'sensors':sensors,'bindings':[dict(b) for b in conn.execute('SELECT * FROM model_bindings WHERE asset_id=?',(asset['id'],))]})
        return rows


@router.post('/assets',response_model=ResourceID)
def register_asset(body:AssetInput,user=Depends(admin)):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope(conn,user)
        return create_asset(conn,body,user)


def create_asset(conn,body,user):
    if len({s.tag_id for s in body.sensors})!=len(body.sensors):raise HTTPException(422,'Duplicate sensor tag')
    sensors={s.tag_id:s for s in body.sensors}
    for s in body.sensors:authorize_source(conn,user,s.source_id)
    for tag,limit in body.criticality.limits.items():
        if tag not in sensors or limit.unit!=sensors[tag].unit or not sensors[tag].lower_bound<=limit.warning<limit.critical<=sensors[tag].upper_bound:
            raise HTTPException(422,'Criticality limits must match the configured sensor contract')
    for dep in body.criticality.dependencies:
        if dep==body.id:raise HTTPException(422,'Asset cannot depend on itself')
        authorize_asset(conn,user,dep)
    try:conn.execute('INSERT INTO assets VALUES(?,?,?,?,?,?,?,?)',(body.id,user['plant_id'],body.name,body.equipment_type,body.area,'unknown',body.criticality.safety,body.criticality.production))
    except sqlite3.IntegrityError as exc:raise HTTPException(409,'Asset identifier unavailable') from exc
    conn.execute('INSERT INTO asset_grants VALUES(?,?)',(user['id'],body.id))
    for s in body.sensors:conn.execute('INSERT INTO sensors VALUES(?,?,?,?,?,?,?,?)',(identifier(),body.id,s.tag_id,s.unit,s.lower_bound,s.upper_bound,s.stale_after_seconds,s.source_id))
    conn.execute('INSERT INTO asset_profiles VALUES(?,?,?,?,?,1)',(identifier(),body.id,json.dumps(body.criticality.model_dump()),user['id'],iso()))
    audit(conn,user,'register_asset',body.id,body.reason+'; no model binding or industrial connection created')
    return {'id':body.id}



@router.put('/assets/{aid}',response_model=ResourceID)
def metadata(aid:str,body:AssetMetadata,user=Depends(admin)):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope(conn,user);authorize_asset(conn,user,aid)
        conn.execute('UPDATE assets SET name=?,area=? WHERE id=?',(body.name,body.area,aid))
        audit(conn,user,'update_asset_metadata',aid,body.reason)
        return {'id':aid}


@router.get('/audit',response_model=list[AuditResponse])
def events(user=Depends(admin)):
    with transaction() as conn:
        scope(conn,user)
        return [dict(r) for r in conn.execute('SELECT * FROM audit_events WHERE plant_id=? ORDER BY timestamp_utc DESC LIMIT 200',(user['plant_id'],))]


@router.get('/sources')
def registered_sources(user=Depends(admin)):
    with transaction() as conn:
        return [dict(r) for r in conn.execute('SELECT s.* FROM sources s JOIN source_grants g ON g.source_id=s.id WHERE g.user_id=? AND s.plant_id=? ORDER BY s.name',(user['id'],user['plant_id']))]


@router.get('/assets/{aid}/telemetry')
def commissioning_telemetry(aid:str,user=Depends(admin)):
    from backend.routes.v2 import telemetry
    return telemetry(aid,limit=120,user=user)
