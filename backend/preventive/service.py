"""Idempotent, bounded generation of unapproved calendar maintenance drafts."""
import asyncio,json,logging
from datetime import datetime,timedelta
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.core.security import authorize_asset
from backend.core.workspaces import asset_workspace
from backend.scheduling.service import role,procedures,order
from backend.copilot.safety import quarantine_reason
from backend.preventive.calendar import occurrence

LOG=logging.getLogger(__name__)
BATCH_SIZE=20


def plan(conn,user,pid):
    row=conn.execute('SELECT * FROM preventive_plans WHERE id=? AND plant_id=?',(pid,user['plant_id'])).fetchone()
    if not row:raise HTTPException(404,'Preventive plan not found or not authorized')
    authorize_asset(conn,user,row['asset_id'])
    return row


def view(conn,user,pid):
    row=plan(conn,user,pid);result=dict(row);result['template']=json.loads(row['template']);result['enabled']=bool(row['enabled'])
    result['history']=[dict(r) for r in conn.execute('SELECT version,actor_id,created_at,reason,enabled FROM preventive_versions WHERE plan_id=? ORDER BY version DESC',(pid,))]
    result['occurrences']=[]
    for r in conn.execute('SELECT * FROM preventive_occurrences WHERE plan_id=? ORDER BY due_at DESC LIMIT 100',(pid,)):
        try:w=order(conn,user,r['work_order_id'])
        except HTTPException:continue
        result['occurrences'].append({**dict(r),'status':w['status'],'overdue':w['status']!='completed' and utcnow()>datetime.fromisoformat(r['overdue_at'])})
    result['catch_up_pending']=bool(row['enabled'] and datetime.fromisoformat(row['next_due'])<=utcnow())
    return result


def validate_scope(conn,user,body):
    role(conn,user,['maintenance_manager']);authorize_asset(conn,user,body.asset_id)
    procedures(conn,user,body.asset_id,body.procedure_ids)
    if quarantine_reason(body.title+' '+body.task):raise HTTPException(422,'Unsafe instructions cannot become a preventive task')


def save(body,user,pid=None):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');validate_scope(conn,user,body)
        template=body.model_dump(mode='json',exclude={'expected_version','reason','asset_id'});index=0;now=iso()
        if pid:
            prior=plan(conn,user,pid)
            if prior['version']!=body.expected_version:raise HTTPException(409,'Plan changed; refresh before revising')
            if prior['asset_id']!=body.asset_id:raise HTTPException(422,'A plan cannot be moved to another machine')
            old=json.loads(prior['template'])
            if all(old[k]==template[k] for k in ['start_local','timezone','interval_unit','interval_count']):index=prior['next_index']
            version=prior['version']+1;enabled=prior['enabled']
        else:version=1;enabled=1;pid=identifier()
        try:due=occurrence(template,index)
        except (ValueError,OverflowError) as exc:raise HTTPException(422,'Calendar range is unsupported; revise the start and interval') from exc
        if index==0 and due<utcnow()-timedelta(days=31):raise HTTPException(422,'A new calendar anchor may be at most 31 days in the past')
        if version==1:
            conn.execute('INSERT INTO preventive_plans VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL)',(pid,user['plant_id'],body.asset_id,user['id'],version,enabled,json.dumps(template),index,iso(due),now,now))
        else:
            conn.execute('UPDATE preventive_plans SET owner_id=?,version=?,template=?,next_index=?,next_due=?,updated_at=?,last_error=NULL WHERE id=?',(user['id'],version,json.dumps(template),index,iso(due),now,pid))
        conn.execute('INSERT INTO preventive_versions VALUES(?,?,?,?,?,?,?)',(pid,version,user['id'],now,body.reason,json.dumps(template),enabled))
        audit(conn,user,'configure_preventive_plan',pid,body.reason)
        return view(conn,user,pid)


def set_state(pid,body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['maintenance_manager']);row=plan(conn,user,pid)
        if row['version']!=body.expected_version:raise HTTPException(409,'Plan changed; refresh before updating')
        if bool(row['enabled'])==body.enabled:raise HTTPException(409,'Plan is already in that state')
        if len(body.reason.strip())<10:raise HTTPException(422,'Record a meaningful change reason')
        if body.enabled:procedures(conn,user,row['asset_id'],json.loads(row['template'])['procedure_ids'])
        now=iso();version=row['version']+1
        conn.execute('UPDATE preventive_plans SET version=?,enabled=?,owner_id=?,updated_at=?,last_error=NULL WHERE id=?',(version,int(body.enabled),user['id'],now,pid))
        conn.execute('INSERT INTO preventive_versions VALUES(?,?,?,?,?,?,?)',(pid,version,user['id'],now,body.reason,row['template'],int(body.enabled)))
        audit(conn,user,'resume_preventive_plan' if body.enabled else 'pause_preventive_plan',pid,body.reason)
        return view(conn,user,pid)


def generate(pid,user,now=None):
    now=now or utcnow();created=[]
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['maintenance_manager']);row=plan(conn,user,pid)
        if not row['enabled']:raise HTTPException(409,'Plan is paused')
        template=json.loads(row['template']);refs=procedures(conn,user,row['asset_id'],template['procedure_ids'])
        if quarantine_reason(template['title']+' '+template['task']):raise HTTPException(422,'Plan content requires review')
        index=row['next_index'];due=occurrence(template,index)
        mode=conn.execute('SELECT mode FROM plants WHERE id=?',(user['plant_id'],)).fetchone()[0]
        for _ in range(BATCH_SIZE):
            if due>now:break
            if not conn.execute('SELECT 1 FROM preventive_occurrences WHERE plan_id=? AND due_at=?',(pid,iso(due))).fetchone():
                wid=identifier();oid=identifier();stamp=iso(now)
                content={k:template[k] for k in ['title','task','procedure_ids','duration_minutes','required_skills','team_size','parts']}
                content.update(asset_id=row['asset_id'],kind='preventive',dependencies=[],due_at=iso(due),priority='Planned maintenance',risk_snapshot=None,investigation_id=None,
                    preventive={'plan_id':pid,'version':row['version'],'occurrence_id':oid,'timezone':template['timezone'],'due_at':iso(due),'overdue_at':iso(due+timedelta(hours=template['grace_hours']))},
                    procedures=refs,simulation_only=mode=='simulation' and asset_workspace(conn,row['asset_id'])=='demo',
                    safety='Calendar-driven draft, not a condition assessment or work authorization. Manager review, approved scheduling and site permits/isolation remain required.')
                conn.execute('INSERT INTO work_orders VALUES(?,?,?,?,?,?,NULL,NULL,NULL)',(wid,row['asset_id'],user['id'],stamp,'draft',json.dumps(content)))
                conn.execute('INSERT INTO preventive_occurrences VALUES(?,?,?,?,?,?,?)',(oid,pid,row['version'],iso(due),content['preventive']['overdue_at'],wid,stamp))
                audit(conn,user,'generate_preventive_draft',wid,'Calendar plan '+pid+'; no resources reserved or maintenance authorized')
                created.append(wid)
            index+=1;due=occurrence(template,index)
        conn.execute('UPDATE preventive_plans SET next_index=?,next_due=?,last_error=NULL WHERE id=?',(index,iso(due),pid))
    return {'created_work_order_ids':created,'catch_up_pending':due<=now}


def sweep():
    with transaction() as conn:ids=[r[0] for r in conn.execute('SELECT id FROM preventive_plans WHERE enabled=1 AND next_due<=? ORDER BY next_due LIMIT 100',(iso(),))]
    for pid in ids:
        try:
            with transaction() as conn:
                owner=conn.execute('SELECT u.id,u.role,u.plant_id FROM users u JOIN preventive_plans p ON p.owner_id=u.id WHERE p.id=?',(pid,)).fetchone()
            if owner:generate(pid,dict(owner))
        except (HTTPException,ValueError,OverflowError):
            with transaction() as conn:conn.execute('UPDATE preventive_plans SET last_error=? WHERE id=? AND enabled=1',('Generation blocked: a manager must review account access, approved procedures and calendar configuration.',pid))


async def worker():
    while True:
        try:await asyncio.to_thread(sweep)
        except Exception:LOG.exception('Preventive generation pass failed; retrying next minute')
        await asyncio.sleep(60)
