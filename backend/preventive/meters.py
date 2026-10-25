"""Cumulative meter recurrence. Counter drops require an attributed human reconciliation."""
import json
from datetime import datetime,timedelta
from decimal import Decimal
from typing import Literal
from fastapi import HTTPException
from pydantic import Field,FiniteFloat,model_validator
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.core.security import authorize_asset,authorize_source
from backend.core.workspaces import asset_workspace,source_workspace
from backend.scheduling.service import role,procedures,order
from backend.scheduling.contracts import Strict
from backend.preventive.contracts import PlanInput
from backend.copilot.safety import quarantine_reason


class MeterInput(Strict):
    asset_id:str
    sensor_id:str
    title:str=Field(min_length=5,max_length=200)
    task:str=Field(min_length=15,max_length=2000)
    procedure_ids:list[str]=Field(min_length=1,max_length=20)
    duration_minutes:int=Field(ge=5,le=480)
    required_skills:list[str]=Field(min_length=1,max_length=10)
    team_size:int=Field(default=1,ge=1,le=3)
    parts:dict[str,int]=Field(default_factory=dict,max_length=30)
    first_due:FiniteFloat=Field(ge=0,le=1e12)
    interval:FiniteFloat=Field(gt=0,le=1e9)
    grace_hours:int=Field(default=24,ge=0,le=720)
    reason:str=Field(min_length=10,max_length=500)

    @model_validator(mode='after')
    def task_scope(self):
        # Share existing task validation; this calendar object is never persisted.
        fields=self.model_dump(exclude={'sensor_id','first_due','interval'})
        checked=PlanInput(**fields,start_local='2030-01-01T00:00:00',timezone='UTC',interval_unit='days',interval_count=1)
        self.required_skills=checked.required_skills
        if self.interval<0.001:raise ValueError('Minimum recurrence interval is 0.001 meter units')
        return self


class MeterRevision(MeterInput):
    expected_version:int=Field(ge=1)

    @model_validator(mode='after')
    def task_scope(self):
        fields=self.model_dump(exclude={'sensor_id','first_due','interval','expected_version'})
        checked=PlanInput(**fields,start_local='2030-01-01T00:00:00',timezone='UTC',interval_unit='days',interval_count=1)
        self.required_skills=checked.required_skills
        if self.interval<0.001:raise ValueError('Minimum recurrence interval is 0.001 meter units')
        return self


class Reconcile(Strict):
    expected_version:int=Field(ge=1)
    sample_id:str
    action:Literal['reset','rollover']
    confirmed_total:FiniteFloat|None=Field(default=None,ge=0,le=1e15)
    modulus:FiniteFloat|None=Field(default=None,gt=0,le=1e12)
    reason:str=Field(min_length=15,max_length=1000)


class MeterRegistration(Strict):
    tag_id:str=Field(min_length=1,max_length=100,pattern=r'^[a-zA-Z0-9_.-]+$')
    unit:Literal['h','cycles']
    source_id:str
    maximum:FiniteFloat=Field(gt=0,le=1e12)
    stale_after_seconds:int=Field(ge=1,le=86400)
    reason:str=Field(min_length=10,max_length=500)


def sensor(conn,user,sid,aid=None):
    row=conn.execute('SELECT * FROM sensors WHERE id=?',(sid,)).fetchone()
    if not row or (aid and row['asset_id']!=aid):raise HTTPException(404,'Meter unavailable for this machine')
    authorize_asset(conn,user,row['asset_id']);authorize_source(conn,user,row['source_id'])
    if row['unit'] not in ['h','cycles'] or row['lower_bound']<0:raise HTTPException(422,'Use a nonnegative cumulative meter in h or cycles')
    return dict(row)


def register(aid,body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['administrator']);authorize_asset(conn,user,aid);source=authorize_source(conn,user,body.source_id)
        if source_workspace(source)!=asset_workspace(conn,aid):raise HTTPException(422,'Meter source must belong to the same demo or plant workspace as the machine')
        if len(body.reason.strip())<10:raise HTTPException(422,'Explain the meter mapping')
        if conn.execute('SELECT 1 FROM sensors WHERE asset_id=? AND tag_id=?',(aid,body.tag_id)).fetchone():raise HTTPException(409,'This tag already has a mapping')
        sid=identifier();conn.execute('INSERT INTO sensors VALUES(?,?,?,?,?,?,?,?)',(sid,aid,body.tag_id,body.unit,0,body.maximum,body.stale_after_seconds,body.source_id))
        audit(conn,user,'register_cumulative_meter',sid,body.reason+'; mapping only, no readings or connection fabricated')
        return sensor(conn,user,sid)


def get(conn,user,pid):
    row=conn.execute('SELECT * FROM meter_plans WHERE id=? AND plant_id=?',(pid,user['plant_id'])).fetchone()
    if not row:raise HTTPException(404,'Meter plan unavailable')
    sensor(conn,user,row['sensor_id'],row['asset_id'])
    return dict(row)


def threshold(template,index):return float(Decimal(str(template['first_due']))+Decimal(str(template['interval']))*index)


def latest(conn,s):
    return conn.execute('SELECT * FROM telemetry WHERE asset_id=? AND tag_id=? AND source=? ORDER BY timestamp_utc DESC LIMIT 1',(s['asset_id'],s['tag_id'],s['source_id'])).fetchone()


def freshness(row,s):
    if not row:return 'No meter observations received'
    if row['unit']!=s['unit'] or row['quality']!='good' or row['value']<0:return 'Latest meter observation has invalid quality, unit or value'
    age=(utcnow()-datetime.fromisoformat(row['timestamp_utc'])).total_seconds()
    if age< -5 or age>s['stale_after_seconds']:return 'Meter observation is stale or future-dated'
    return None


def view(conn,user,pid):
    p=get(conn,user,pid);t=json.loads(p['template']);p['template']=t;p['enabled']=bool(p['enabled'])
    s=sensor(conn,user,p['sensor_id']);p['meter']=s;p['reading_problem']=freshness(latest(conn,s),s)
    p['next_threshold']=threshold(t,p['next_index']);p['catch_up_pending']=p['enabled'] and p['total'] is not None and p['total']>=p['next_threshold']
    p['history']=[dict(r) for r in conn.execute('SELECT version,actor_id,created_at,reason,enabled FROM meter_plan_versions WHERE plan_id=? ORDER BY version DESC',(pid,))]
    p['reconciliations']=[dict(r) for r in conn.execute('SELECT * FROM meter_reconciliations WHERE plan_id=? ORDER BY created_at DESC LIMIT 100',(pid,))]
    p['occurrences']=[]
    for r in conn.execute('SELECT * FROM meter_occurrences WHERE plan_id=? ORDER BY occurrence_index DESC LIMIT 100',(pid,)):
        try:w=order(conn,user,r['work_order_id'])
        except HTTPException:continue
        p['occurrences'].append({**dict(r),'status':w['status'],'overdue':w['status']!='completed' and utcnow()>datetime.fromisoformat(r['overdue_at'])})
    if p['blocked_sample_id']:
        r=conn.execute('SELECT id,value,timestamp_utc FROM telemetry WHERE id=?',(p['blocked_sample_id'],)).fetchone();p['blocked_sample']=dict(r) if r else None
    return p


def save(body,user,pid=None):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['maintenance_manager']);s=sensor(conn,user,body.sensor_id,body.asset_id);procedures(conn,user,body.asset_id,body.procedure_ids)
        if quarantine_reason(body.title+' '+body.task):raise HTTPException(422,'Unsafe instructions cannot become a meter task')
        template=body.model_dump(exclude={'asset_id','sensor_id','reason','expected_version'});template.update(unit=s['unit'],source_id=s['source_id'],tag_id=s['tag_id']);now=iso()
        if pid:
            old=get(conn,user,pid);prior=json.loads(old['template'])
            if old['version']!=body.expected_version:raise HTTPException(409,'Plan changed; refresh before editing')
            if old['asset_id']!=body.asset_id or old['sensor_id']!=body.sensor_id or any(prior[k]!=template[k] for k in ['first_due','interval','unit','source_id','tag_id']):raise HTTPException(422,'Meter identity and recurrence are immutable; pause this plan and create a new one to change them')
            version=old['version']+1;enabled=old['enabled'];conn.execute('UPDATE meter_plans SET owner_id=?,version=?,template=?,updated_at=? WHERE id=?',(user['id'],version,json.dumps(template),now,pid))
        else:
            pid=identifier();version=1;enabled=1
            conn.execute('INSERT INTO meter_plans(id,plant_id,asset_id,sensor_id,owner_id,version,enabled,template,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)',(pid,user['plant_id'],body.asset_id,body.sensor_id,user['id'],version,enabled,json.dumps(template),now,now))
        conn.execute('INSERT INTO meter_plan_versions VALUES(?,?,?,?,?,?,?)',(pid,version,user['id'],now,body.reason,json.dumps(template),enabled));audit(conn,user,'configure_meter_plan',pid,body.reason)
        return view(conn,user,pid)


def state(pid,body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['maintenance_manager']);p=get(conn,user,pid)
        if p['version']!=body.expected_version or bool(p['enabled'])==body.enabled:raise HTTPException(409,'Plan changed or already has that state')
        if len(body.reason.strip())<10:raise HTTPException(422,'Explain the plan change')
        if body.enabled:procedures(conn,user,p['asset_id'],json.loads(p['template'])['procedure_ids'])
        v=p['version']+1;now=iso();conn.execute('UPDATE meter_plans SET owner_id=?,version=?,enabled=?,updated_at=? WHERE id=?',(user['id'],v,int(body.enabled),now,pid))
        conn.execute('INSERT INTO meter_plan_versions VALUES(?,?,?,?,?,?,?)',(pid,v,user['id'],now,body.reason,p['template'],int(body.enabled)));audit(conn,user,'change_meter_plan_state',pid,body.reason)
        return view(conn,user,pid)


def generate(pid,user):
    created=[]
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['maintenance_manager']);p=get(conn,user,pid);t=json.loads(p['template']);s=sensor(conn,user,p['sensor_id'])
        if not p['enabled']:raise HTTPException(409,'Plan is paused')
        refs=procedures(conn,user,p['asset_id'],t['procedure_ids'])
        if any(t[k]!=s[v] for k,v in [('unit','unit'),('source_id','source_id'),('tag_id','tag_id')]):raise HTTPException(409,'Meter mapping changed; review the plan')
        problem=freshness(latest(conn,s),s)
        if p['blocked_sample_id']:problem='Counter decreased; a manager must reconcile reset or rollover'
        if problem:
            conn.execute('UPDATE meter_plans SET last_error=? WHERE id=?',(problem,pid));return {'created_work_order_ids':[],'blocked':problem}
        # Inspect every good new sample, not just the newest value, so an observed
        # reset cannot disappear behind a later higher reading. Invalid rows never advance the cursor.
        rows=conn.execute("SELECT * FROM telemetry WHERE asset_id=? AND tag_id=? AND source=? AND unit=? AND quality='good' AND value>=0 AND (? IS NULL OR timestamp_utc>?) ORDER BY timestamp_utc LIMIT 1000",(p['asset_id'],s['tag_id'],s['source_id'],s['unit'],p['last_timestamp'],p['last_timestamp'])).fetchall()
        for r in rows:
            if p['last_raw'] is not None and r['value']<p['last_raw']:
                problem='Counter decreased; a manager must reconcile reset or rollover';conn.execute('UPDATE meter_plans SET blocked_sample_id=?,last_error=? WHERE id=?',(r['id'],problem,pid));return {'created_work_order_ids':[],'blocked':problem}
            p.update(last_raw=r['value'],last_timestamp=r['timestamp_utc'],last_sample_id=r['id'],total=r['value']+p['offset'])
            conn.execute('UPDATE meter_plans SET last_raw=?,last_timestamp=?,last_sample_id=?,total=? WHERE id=?',(p['last_raw'],p['last_timestamp'],p['last_sample_id'],p['total'],pid))
        if len(rows)==1000:return {'created_work_order_ids':[],'catch_up_pending':True}
        if p['total'] is None:return {'created_work_order_ids':[],'blocked':'No usable meter value'}
        # Due time is detection time, not an invented threshold-crossing timestamp.
        now=utcnow();index=p['next_index'];mode=conn.execute('SELECT mode FROM plants WHERE id=?',(user['plant_id'],)).fetchone()[0]
        for _ in range(20):
            due=threshold(t,index)
            if p['total']<due:break
            wid=identifier();oid=identifier();stamp=iso(now);overdue=iso(now+timedelta(hours=t['grace_hours']))
            content={k:t[k] for k in ['title','task','procedure_ids','duration_minutes','required_skills','team_size','parts']}
            content.update(asset_id=p['asset_id'],kind='preventive',dependencies=[],due_at=stamp,priority='Planned maintenance',risk_snapshot=None,investigation_id=None,
                preventive={'trigger':'meter','plan_id':pid,'version':p['version'],'occurrence_id':oid,'threshold':due,'unit':s['unit'],'observed_total':p['total'],'sample_id':p['last_sample_id'],'source_id':s['source_id'],'sensor_id':s['id'],'sample_timestamp':p['last_timestamp'],'due_at':stamp,'overdue_at':overdue},
                procedures=refs,simulation_only=mode=='simulation' and asset_workspace(conn,p['asset_id'])=='demo',safety='Meter-threshold draft detected from recorded readings, not a diagnosis or work authorization. Actual crossing time is unknown. Human review, scheduling and site permits remain required.')
            conn.execute('INSERT INTO work_orders VALUES(?,?,?,?,?,?,NULL,NULL,NULL)',(wid,p['asset_id'],user['id'],stamp,'draft',json.dumps(content)))
            conn.execute('INSERT INTO meter_occurrences VALUES(?,?,?,?,?,?,?,?)',(oid,pid,index,due,wid,stamp,overdue,p['version']))
            created.append(wid);index+=1;audit(conn,user,'generate_meter_draft',wid,'Meter plan '+pid+'; unapproved draft')
        conn.execute('UPDATE meter_plans SET next_index=?,last_error=NULL WHERE id=?',(index,pid))
    return {'created_work_order_ids':created,'catch_up_pending':p['total']>=threshold(t,index)}


def reconcile(pid,body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['maintenance_manager']);p=get(conn,user,pid)
        if p['version']!=body.expected_version or p['blocked_sample_id']!=body.sample_id:raise HTTPException(409,'Counter state changed; review the current blocked observation')
        if len(body.reason.strip())<15 or quarantine_reason(body.reason):raise HTTPException(422,'Record a meaningful, safe reconciliation reason')
        s=sensor(conn,user,p['sensor_id']);r=conn.execute('SELECT * FROM telemetry WHERE id=?',(body.sample_id,)).fetchone()
        if not r or r['source']!=s['source_id'] or r['unit']!=s['unit'] or r['asset_id']!=p['asset_id'] or r['tag_id']!=s['tag_id']:raise HTTPException(409,'Meter evidence changed')
        if body.action=='rollover':
            if body.modulus is None or body.confirmed_total is not None or not max(p['last_raw'],r['value'])<body.modulus:raise HTTPException(422,'Rollover requires a modulus above both raw values and no manual total')
            total=p['total']+body.modulus-p['last_raw']+r['value']
        else:
            if body.confirmed_total is None or body.modulus is not None or body.confirmed_total<p['total']:raise HTTPException(422,'Reset requires an attributed lifetime total at least as large as the prior accepted total')
            total=body.confirmed_total
        version=p['version']+1;now=iso()
        conn.execute('UPDATE meter_plans SET owner_id=?,version=?,last_sample_id=?,last_timestamp=?,last_raw=?,offset=?,total=?,blocked_sample_id=NULL,last_error=NULL,updated_at=? WHERE id=?',(user['id'],version,r['id'],r['timestamp_utc'],r['value'],total-r['value'],total,now,pid))
        conn.execute('INSERT INTO meter_reconciliations VALUES(?,?,?,?,?,?,?,?,?,?)',(identifier(),pid,user['id'],now,r['id'],body.action,p['total'],total,body.reason,version))
        conn.execute('INSERT INTO meter_plan_versions VALUES(?,?,?,?,?,?,?)',(pid,version,user['id'],now,body.reason,p['template'],p['enabled']));audit(conn,user,'reconcile_meter_'+body.action,pid,body.reason)
        return view(conn,user,pid)


def sweep():
    with transaction() as conn:rows=conn.execute('SELECT p.id,u.id AS owner,u.role,u.plant_id FROM meter_plans p JOIN users u ON u.id=p.owner_id WHERE p.enabled=1 ORDER BY p.updated_at,p.id LIMIT 500').fetchall()
    for r in rows:
        try:generate(r['id'],dict(id=r['owner'],role=r['role'],plant_id=r['plant_id']))
        except (HTTPException,ValueError,OverflowError):
            with transaction() as conn:conn.execute('UPDATE meter_plans SET last_error=? WHERE id=?',('Generation blocked: review owner access, procedures and meter mapping.',r['id']))
