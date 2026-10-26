"""Durable, access-scoped in-app delivery. Acknowledgement never authorizes work."""
import asyncio,json,logging
from datetime import datetime,timedelta
from urllib.parse import quote
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.core.security import authorize_asset,authorize_source

LOG=logging.getLogger(__name__)
LABELS={
 'work_review':'Maintenance candidate needs review','work_assignment':'Maintenance assigned to you',
 'repair_review':'Repair report needs verification','return_review':'Production return needs review',
 'release_review':'Equipment release needs review','work_change':'Maintenance assignment changed',
 'window_review':'Downtime window needs agreement','schedule_review':'Schedule proposal needs review',
 'issue_review':'Reported issue needs triage','issue_assignment':'Issue assigned to you',
 'round_assignment':'Inspection round assigned to you','critical':'Critical sensor readings need attention',
}
AUDIENCES={'manager':['maintenance_manager'],'operations':['operations_coordinator'],
 'engineer':['engineer','reliability_engineer'],'critical':['engineer','reliability_engineer','maintenance_manager']}


def policy(conn,plant_id):
    row=conn.execute('SELECT * FROM notification_policy WHERE plant_id=?',(plant_id,)).fetchone()
    return dict(row) if row else dict(plant_id=plant_id,version=0,escalation_minutes=60,escalation_enabled=True,reason='Default in-app policy',updated_at=None)


def configure(body,user):
    from backend.scheduling.service import role
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['administrator']);old=policy(conn,user['plant_id'])
        if old['version']!=body.expected_version:raise HTTPException(409,'Policy changed; refresh before editing')
        now=iso();version=old['version']+1
        conn.execute('INSERT INTO notification_policy VALUES(?,?,?,?,?,?,?) ON CONFLICT(plant_id) DO UPDATE SET version=excluded.version,escalation_minutes=excluded.escalation_minutes,escalation_enabled=excluded.escalation_enabled,actor_id=excluded.actor_id,updated_at=excluded.updated_at,reason=excluded.reason',(user['plant_id'],version,body.escalation_minutes,int(body.escalation_enabled),user['id'],now,body.reason))
        conn.execute('INSERT INTO notification_policy_history VALUES(?,?,?,?,?,?)',(identifier(),user['plant_id'],version,user['id'],now,body.model_dump_json()))
        audit(conn,user,'configure_notifications',user['plant_id'],body.reason)
        return policy(conn,user['plant_id'])


def context(conn,event,user):
    """Reauthorize the entire linked record; never expose cached titles after revocation."""
    event=dict(event)
    asset=authorize_asset(conn,user,event['asset_id']);kind=event['kind'];rid=event['resource_id'];revision=event['revision']
    active=True;title=asset['name'];url='/operations/assets/'+quote(asset['id'],safe='')
    if kind in ['work_review','work_assignment','repair_review','return_review','release_review','work_change']:
        from backend.scheduling.service import order
        work=order(conn,user,rid);title=work['content']['title'];url='/operations/scheduling?view=work&work='+quote(rid,safe='')
        if kind=='work_review':active=work['status']=='draft'
        elif kind=='repair_review':active=work['status']=='awaiting_verification'
        elif kind=='return_review':active=work['status']=='awaiting_return_to_service'
        elif kind in ['work_assignment','release_review']:
            assignment=conn.execute('SELECT proposal_id FROM schedule_assignments WHERE work_order_id=?',(rid,)).fetchone()
            active=bool(assignment and assignment['proposal_id']==revision and work['status']=='scheduled')
            if kind=='release_review':active=active and bool(work['operations'] and work['operations']['status'] in ['awaiting_release','held'])
    elif kind in ['window_review','schedule_review']:
        from backend.scheduling.service import proposal
        pid=rid
        if kind=='window_review':
            row=conn.execute('SELECT * FROM operations_requests WHERE id=?',(rid,)).fetchone()
            if not row:raise HTTPException(404,'Notification source unavailable')
            pid=row['proposal_id']
            from backend.scheduling.coordination import policy as operations_policy
            active=row['status']=='requested' and operations_policy(conn,asset['id'])['version']==row['policy_version'] and datetime.fromisoformat(row['start_utc'])>utcnow()
        p=proposal(conn,user,pid);current=conn.execute('SELECT active FROM planning_configs WHERE id=?',(p['config_id'],)).fetchone()
        active=active and p['status']=='pending' and p['result']['feasibility']=='feasible' and bool(current and current['active'])
        if any(datetime.fromisoformat(a['start_utc'])<=utcnow() for a in p['result'].get('assignments',[])):active=False
        url='/operations/coordination' if kind=='window_review' else '/operations/scheduling?view=schedule'
    elif kind in ['issue_review','issue_assignment']:
        from backend.routes.issues import view
        item=view(conn,user,rid);title=item['title'];active=item['status']=='reported' if kind=='issue_review' else item['status']!='closed' and item['assignee_id']==revision
        url='/operations/issues?asset='+quote(asset['id'],safe='')
    elif kind=='round_assignment':
        from backend.inspections.service import run
        item=run(conn,user,rid);title=item['content']['title'];active=item['status'] not in ['completed','completed_with_skips','cancelled'] and item['assignee_id']==revision;url='/operations/preventive/rounds'
    elif kind=='critical':
        from backend.ingestion.condition import describe_asset
        for sensor in conn.execute('SELECT source_id FROM sensors WHERE asset_id=?',(asset['id'],)):authorize_source(conn,user,sensor['source_id'])
        observed=describe_asset(conn,user,asset);episode=conn.execute('SELECT * FROM notification_conditions WHERE asset_id=?',(asset['id'],)).fetchone()
        active=bool(episode and str(episode['episode'])==revision and episode['is_open'] and observed['condition']=='critical')
    if event.get('parent_id'):
        parent=dict(conn.execute('SELECT * FROM notification_events WHERE id=?',(event['parent_id'],)).fetchone());outstanding=False
        for delivery in conn.execute("SELECT * FROM notification_deliveries WHERE event_id=? AND acknowledged_at IS NULL AND status IN('delivered','failed')",(parent['id'],)).fetchall():
            try:outstanding=outstanding or eligible(conn,parent,delivery,identity(conn,delivery['user_id']))['active']
            except HTTPException:continue
        active=active and outstanding
    return dict(title=title,asset_name=asset['name'],href=url,active=bool(active))


def identity(conn,uid):
    row=conn.execute('SELECT id,username,role,plant_id FROM users WHERE id=? AND enabled=1',(uid,)).fetchone()
    if not row:raise HTTPException(404,'Notification unavailable')
    return dict(row)


def eligible(conn,event,delivery,user):
    current=identity(conn,user['id'])
    if current['plant_id']!=event['plant_id'] or current['role'] not in AUDIENCES[delivery['audience']]:raise HTTPException(404,'Notification unavailable')
    current.update(workspace=user.get('workspace'))
    return context(conn,event,current)


def enqueue(conn,kind,rid,aid,key,audience,revision='',users=None,parent_id=None,now=None):
    stamp=iso(now);plant=conn.execute('SELECT plant_id FROM assets WHERE id=?',(aid,)).fetchone()
    if not plant:return None
    old=conn.execute('SELECT id FROM notification_events WHERE event_key=?',(key,)).fetchone()
    if old:return old['id']
    eid=identifier();rules=policy(conn,plant['plant_id'])
    event=dict(id=eid,event_key=key,plant_id=plant['plant_id'],asset_id=aid,kind=kind,resource_id=rid,revision=revision,created_at=stamp,parent_id=parent_id,escalate_after=rules['escalation_minutes'] if rules['escalation_enabled'] else 0)
    conn.execute('INSERT INTO notification_events VALUES(?,?,?,?,?,?,?,?,?,?)',tuple(event.values()))
    for row in conn.execute('SELECT id FROM users WHERE plant_id=? AND enabled=1',(plant['plant_id'],)).fetchall():
        if users is not None and row['id'] not in users:continue
        try:eligible(conn,event,dict(audience=audience),identity(conn,row['id']))
        except HTTPException:continue
        conn.execute('INSERT INTO notification_deliveries(id,event_id,user_id,audience,status,next_attempt) VALUES(?,?,?,?,?,?)',(identifier(),eid,row['id'],audience,'pending',stamp))
    return eid


def inbox(user,history=False):
    with transaction() as conn:
        rows=conn.execute('SELECT * FROM notification_deliveries WHERE user_id=? ORDER BY rowid DESC',(user['id'],)).fetchall();items=[];unread=0;unacknowledged=0;failed=0
        for row in rows:
            if row['status']=='suppressed':continue
            event=dict(conn.execute('SELECT * FROM notification_events WHERE id=?',(row['event_id'],)).fetchone())
            try:ctx=eligible(conn,event,row,user)
            except HTTPException:continue
            is_current=ctx['active'] and not row['acknowledged_at']
            if is_current:
                unread+=int(row['status']=='delivered' and not row['read_at']);unacknowledged+=int(row['status']=='delivered');failed+=int(row['status']=='failed')
            if not history and not is_current:continue
            items.append({**dict(row),**ctx,'kind':event['kind'],'label':('Escalation: ' if event['parent_id'] else '')+LABELS[event['kind']],'created_at':event['created_at'],'escalated':bool(event['parent_id']),'asset_id':event['asset_id']})
        worker=conn.execute("SELECT last_success FROM notification_worker WHERE id='in_app'").fetchone()
        return dict(items=items,unread=unread,unacknowledged=unacknowledged,failed=failed,last_worker_success=worker[0] if worker else None,external_delivery='not_configured')


def update(did,body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');row=conn.execute('SELECT * FROM notification_deliveries WHERE id=? AND user_id=?',(did,user['id'])).fetchone()
        if not row or row['status']=='suppressed':raise HTTPException(404,'Notification unavailable')
        event=dict(conn.execute('SELECT * FROM notification_events WHERE id=?',(row['event_id'],)).fetchone());ctx=eligible(conn,event,row,user)
        if body.action=='retry':
            if row['status']!='failed' or not ctx['active']:raise HTTPException(409,'Only a current failed delivery can be retried')
            conn.execute("UPDATE notification_deliveries SET status='pending',attempts=0,next_attempt=?,last_error=NULL WHERE id=?",(iso(),did))
        else:
            if row['status']!='delivered':raise HTTPException(409,'Notification has not been delivered')
            if body.action=='acknowledge':
                if row['acknowledged_at']:return {'status':'acknowledged'}
                if not ctx['active']:raise HTTPException(409,'The linked workflow has changed; review its current record')
                if len((body.reason or '').strip())<10:raise HTTPException(422,'Record an acknowledgement reason of at least 10 characters')
                conn.execute('UPDATE notification_deliveries SET acknowledged_at=?,acknowledgement=? WHERE id=?',(iso(),body.reason.strip(),did))
            conn.execute('UPDATE notification_deliveries SET read_at=COALESCE(read_at,?) WHERE id=?',(iso(),did))
        audit(conn,user,'notification_'+body.action,did,'Personal inbox action; no work or equipment decision')
    return {'status':body.action}


def deliver(conn,row,now):
    event=dict(conn.execute('SELECT * FROM notification_events WHERE id=?',(row['event_id'],)).fetchone())
    try:
        ctx=eligible(conn,event,row,identity(conn,row['user_id']))
        if not ctx['active']:raise HTTPException(409,'Workflow changed')
    except HTTPException:
        conn.execute("UPDATE notification_deliveries SET status='suppressed',last_error='No longer eligible or current' WHERE id=?",(row['id'],));return
    conn.execute("UPDATE notification_deliveries SET status='delivered',delivered_at=?,attempts=attempts+1,last_error=NULL WHERE id=?",(iso(now),row['id']))


def tick(now=None):
    now=now or utcnow()
    from backend.notifications.conditions import scan
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scan(conn,now)
        for row in conn.execute("SELECT * FROM notification_deliveries WHERE status='pending' AND next_attempt<=? ORDER BY next_attempt LIMIT 200",(iso(now),)).fetchall():
            conn.execute('SAVEPOINT delivery')
            try:deliver(conn,row,now)
            except Exception:
                conn.execute('ROLLBACK TO delivery');attempts=row['attempts']+1
                conn.execute('UPDATE notification_deliveries SET status=?,attempts=?,next_attempt=?,last_error=? WHERE id=?',('failed' if attempts>=5 else 'pending',attempts,iso(now+timedelta(seconds=min(3600,30*2**(attempts-1)))),'In-app delivery failed; retry available after five attempts',row['id']))
                LOG.warning('In-app delivery attempt failed for %s',row['id'])
            finally:conn.execute('RELEASE delivery')
        # One durable escalation per original event, not one per unread recipient.
        for e in conn.execute('SELECT * FROM notification_events WHERE parent_id IS NULL AND escalate_after>0 AND created_at<=?',(iso(now-timedelta(minutes=1)),)).fetchall():
            if datetime.fromisoformat(e['created_at'])+timedelta(minutes=e['escalate_after'])>now or not policy(conn,e['plant_id'])['escalation_enabled']:continue
            for d in conn.execute("SELECT * FROM notification_deliveries WHERE event_id=? AND acknowledged_at IS NULL AND status IN('delivered','failed')",(e['id'],)).fetchall():
                try:ctx=eligible(conn,e,d,identity(conn,d['user_id']))
                except HTTPException:continue
                if not ctx['active']:continue
                enqueue(conn,e['kind'],e['resource_id'],e['asset_id'],'escalation:'+e['id'],'manager',e['revision'],parent_id=e['id'],now=now);break
        conn.execute("INSERT INTO notification_worker VALUES('in_app',?) ON CONFLICT(id) DO UPDATE SET last_success=excluded.last_success",(iso(now),))


async def worker():
    while True:
        try:await asyncio.to_thread(tick)
        except Exception:LOG.exception('Notification worker failed; retrying in ten seconds')
        await asyncio.sleep(10)
