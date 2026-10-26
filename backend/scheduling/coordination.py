"""Human operations decisions; no equipment control or substitute for site permits."""
from datetime import datetime
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.core.security import authorize_asset
from backend.copilot.safety import quarantine_reason


def policy(conn,aid):
    row=conn.execute('SELECT * FROM asset_operations_policy WHERE asset_id=?',(aid,)).fetchone()
    return dict(row) if row else dict(asset_id=aid,required=False,version=0,actor_id=None,updated_at=None,reason=None)


def configure(aid,body,user):
    from backend.scheduling.service import role
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['administrator','maintenance_manager']);authorize_asset(conn,user,aid)
        if quarantine_reason(body.reason):raise HTTPException(422,'Unsafe instructions cannot configure operations policy')
        old=policy(conn,aid)
        if old['version']!=body.expected_version:raise HTTPException(409,'Policy changed; refresh before editing')
        # Query status directly, even if a document has since been revoked. An
        # inaccessible active job must not allow its safety gate to be removed.
        if conn.execute("SELECT 1 FROM work_orders w LEFT JOIN work_execution e ON e.work_order_id=w.id WHERE w.asset_id=? AND COALESCE(e.status,w.status) IN('scheduled','in_progress','awaiting_verification','awaiting_return_to_service')",(aid,)).fetchone():
            raise HTTPException(409,'Finish confirmed work before changing this machine policy')
        conn.execute('INSERT INTO asset_operations_policy VALUES(?,?,?,?,?,?) ON CONFLICT(asset_id) DO UPDATE SET required=excluded.required,version=excluded.version,actor_id=excluded.actor_id,updated_at=excluded.updated_at,reason=excluded.reason',(aid,int(body.required),old['version']+1,user['id'],iso(),body.reason))
        conn.execute('INSERT INTO operations_policy_history VALUES(?,?,?,?,?,?)',(aid,old['version']+1,int(body.required),user['id'],iso(),body.reason))
        audit(conn,user,'configure_operations_policy',aid,body.reason)
        return policy(conn,aid)


def create_requests(conn,pid,assignments):
    for a in assignments:
        rule=policy(conn,a['asset_id'])
        if rule['required']:
            conn.execute('INSERT INTO operations_requests VALUES(?,?,?,?,?,\'requested\',?,?,NULL,NULL,NULL)',(identifier(),pid,a['work_order_id'],rule['version'],0,a['start_utc'],a['end_utc']))


def request_row(conn,user,rid):
    from backend.scheduling.service import order
    row=conn.execute('SELECT * FROM operations_requests WHERE id=?',(rid,)).fetchone()
    if not row:raise HTTPException(404,'Operations request unavailable')
    return dict(row),order(conn,user,row['work_order_id'])


def eligible_actor(conn,uid,work_id):
    from backend.scheduling.service import order
    person=conn.execute("SELECT id,role,plant_id FROM users WHERE id=? AND enabled=1 AND role='operations_coordinator'",(uid,)).fetchone()
    if not person:raise HTTPException(409,'Operations decision maker no longer has an enabled coordination role')
    try:order(conn,dict(person),work_id)
    except HTTPException:raise HTTPException(409,'Operations decision maker no longer has the required evidence access')


def decide(rid,body,user):
    from backend.scheduling.service import role
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['operations_coordinator']);r,w=request_row(conn,user,rid)
        if r['version']!=body.expected_version:raise HTTPException(409,'Operations request changed; refresh before deciding')
        p=conn.execute('SELECT status FROM schedule_proposals WHERE id=?',(r['proposal_id'],)).fetchone()
        rule=policy(conn,w['asset_id'])
        if quarantine_reason(body.reason):raise HTTPException(422,'Unsafe instructions cannot approve downtime')
        config=conn.execute('SELECT active FROM planning_configs WHERE id=(SELECT config_id FROM schedule_proposals WHERE id=?)',(r['proposal_id'],)).fetchone()
        if not config or not config['active']:raise HTTPException(409,'Planning resources changed; create a new proposal')
        if p['status']!='pending' or w['status']!='approved' or not rule['required'] or rule['version']!=r['policy_version']:raise HTTPException(409,'Request is no longer current; create a new schedule proposal')
        if datetime.fromisoformat(r['start_utc'])<=utcnow():raise HTTPException(409,'Requested time has passed; propose a new window')
        status='approved' if body.decision=='approved' else 'declined';version=r['version']+1;now=iso()
        conn.execute('UPDATE operations_requests SET status=?,version=?,actor_id=?,decided_at=?,reason=? WHERE id=?',(status,version,user['id'],now,body.reason,rid))
        conn.execute('INSERT INTO operations_events VALUES(?,?,?,?,?,?,?,?)',(identifier(),w['id'],rid,user['id'],now,'window_'+status,body.reason,version))
        audit(conn,user,'operations_window_'+status,rid,body.reason)
        return dict(conn.execute('SELECT * FROM operations_requests WHERE id=?',(rid,)).fetchone())


def confirm(conn,user,pid,assignments):
    for a in assignments:
        rule=policy(conn,a['asset_id'])
        if not rule['required']:continue
        row=conn.execute('SELECT * FROM operations_requests WHERE proposal_id=? AND work_order_id=?',(pid,a['work_order_id'])).fetchone()
        if not row or row['policy_version']!=rule['version'] or row['status']!='approved' or row['start_utc']!=a['start_utc'] or row['end_utc']!=a['end_utc']:
            raise HTTPException(409,'Operations must approve each exact downtime window before schedule confirmation')
        eligible_actor(conn,row['actor_id'],a['work_order_id'])
        conn.execute("INSERT INTO operations_handovers VALUES(?,?,0,'awaiting_release',NULL,?)",(a['work_order_id'],row['id'],iso()))


def handover_snapshot(conn,wid):
    row=conn.execute('SELECT * FROM operations_handovers WHERE work_order_id=?',(wid,)).fetchone()
    if not row:return None
    return {**dict(row),'events':[dict(e) for e in conn.execute('SELECT * FROM operations_events WHERE work_order_id=? ORDER BY created_at,id',(wid,))]}


def require_release(conn,wid):
    row=conn.execute('SELECT * FROM operations_handovers WHERE work_order_id=?',(wid,)).fetchone()
    if not row:return
    if row['status']!='released':raise HTTPException(409,'Operations must release the machine to maintenance before work starts')
    eligible_actor(conn,row['actor_id'],wid)


def handover(wid,body,user):
    from backend.scheduling.service import role,order
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['operations_coordinator']);w=order(conn,user,wid)
        row=handover_snapshot(conn,wid)
        if not row:raise HTTPException(409,'This work has no required operations handover')
        if row['version']!=body.expected_version:raise HTTPException(409,'Handover changed; refresh before deciding')
        if quarantine_reason(body.reason):raise HTTPException(422,'Unsafe instructions cannot authorize a handover')
        action=body.action
        if action in ['release','hold']:
            if w['status']!='scheduled':raise HTTPException(409,'Release or hold applies only before work starts')
            status='released' if action=='release' else 'held'
            # A fresh attributed decision can replace a revoked coordinator's release.
        else:
            if w['status']!='awaiting_return_to_service':raise HTTPException(409,'Independent repair verification must finish before return-to-service review')
            involved={e['actor_id'] for e in w['execution']['events'] if e['action'] in ['report_fixed','verify']}
            if user['id'] in involved:raise HTTPException(403,'Return-to-service review must be independent of repair reporting and verification')
            status='return_confirmed' if action=='confirm_return' else 'return_declined'
            # A fresh attributed decision can replace a revoked coordinator's release.
        now=iso();version=row['version']+1
        conn.execute('UPDATE operations_handovers SET version=?,status=?,actor_id=?,updated_at=? WHERE work_order_id=?',(version,status,user['id'],now,wid))
        conn.execute('INSERT INTO operations_events VALUES(?,?,?,?,?,?,?,?)',(identifier(),wid,row['request_id'],user['id'],now,action,body.reason,version))
        if action=='confirm_return':
            ev=w['execution']['version']+1
            conn.execute("UPDATE work_execution SET status='completed',version=?,updated_at=? WHERE work_order_id=?",(ev,now,wid))
            conn.execute('INSERT INTO work_events VALUES(?,?,?,?,?,?,?,?)',(identifier(),wid,user['id'],now,'operations_return','completed',body.reason,ev))
        audit(conn,user,'operations_'+action,wid,body.reason+'; human record only, no equipment command')
        return order(conn,user,wid)


def reopen(conn,wid,user,reason):
    row=handover_snapshot(conn,wid)
    if not row:raise HTTPException(409,'Operations handover unavailable')
    from backend.core.workspaces import asset_workspace
    work=conn.execute('SELECT asset_id FROM work_orders WHERE id=?',(wid,)).fetchone()
    mode=conn.execute('SELECT mode FROM plants WHERE id=?',(user['plant_id'],)).fetchone()[0]
    window=conn.execute('SELECT end_utc FROM schedule_assignments WHERE work_order_id=?',(wid,)).fetchone()
    if (mode!='simulation' or asset_workspace(conn,work['asset_id'])!='demo') and (not window or datetime.fromisoformat(window['end_utc'])<=utcnow()):
        raise HTTPException(409,'Original work window expired. Plan separate approved corrective work before recording a production return decision.')
    version=row['version']+1;now=iso()
    conn.execute("UPDATE operations_handovers SET version=?,status='held',actor_id=?,updated_at=? WHERE work_order_id=?",(version,user['id'],now,wid))
    conn.execute('INSERT INTO operations_events VALUES(?,?,?,?,?,?,?,?)',(identifier(),wid,row['request_id'],user['id'],now,'repair_reopened',reason,version))


def requests(user):
    from backend.scheduling.service import active_config
    with transaction() as conn:
        result=[]
        rows=conn.execute('SELECT r.id FROM operations_requests r JOIN work_orders w ON w.id=r.work_order_id JOIN assets a ON a.id=w.asset_id WHERE a.plant_id=? ORDER BY r.start_utc DESC LIMIT 500',(user['plant_id'],)).fetchall()
        for row in rows:
            try:r,w=request_row(conn,user,row['id'])
            except HTTPException:continue
            p=conn.execute('SELECT status,config_id FROM schedule_proposals WHERE id=?',(r['proposal_id'],)).fetchone();rule=policy(conn,w['asset_id'])
            current=p['status']=='pending' and w['status']=='approved' and rule['required'] and rule['version']==r['policy_version'] and datetime.fromisoformat(r['start_utc'])>utcnow()
            try:current=current and active_config(conn,user)['id']==p['config_id']
            except HTTPException:current=False
            result.append({**r,'asset_id':w['asset_id'],'title':w['content']['title'],'work_status':w['status'],'current':bool(current)})
        return result
