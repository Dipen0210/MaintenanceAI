"""Manager-directed changes to unstarted work, with immutable reservation history."""
import json
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,audit
from backend.copilot.safety import quarantine_reason


def history(conn,wid):
    return [{**dict(r),'snapshot':json.loads(r['snapshot'])} for r in conn.execute('SELECT * FROM work_lifecycle_events WHERE work_order_id=? ORDER BY version',(wid,))]


def change(wid,body,user):
    from backend.scheduling.service import order,role
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['maintenance_manager']);work=order(conn,user,wid)
        if work['execution']['version']!=body.expected_version:raise HTTPException(409,'Work changed; review the latest record before changing its schedule')
        if not body.not_started_confirmed:raise HTTPException(422,'Confirm with the field team that physical work has not started')
        if quarantine_reason(body.reason):raise HTTPException(422,'Unsafe instructions cannot authorize a lifecycle change')
        if work['status'] not in ['draft','approved','scheduled'] or any(e['action']=='start' for e in work['execution']['events']):
            raise HTTPException(409,'Only work that has never started can be cancelled or returned to planning')
        if body.action=='replan' and work['status']!='scheduled':raise HTTPException(409,'Only confirmed, unstarted work can be returned to planning')
        ops=work['operations']
        if ops and ops['status']=='released':raise HTTPException(409,'Operations must place released equipment on hold before cancelling or replanning')
        blocked=['scheduled','in_progress','awaiting_verification','awaiting_return_to_service'] if body.action=='replan' else ['draft','approved','scheduled','in_progress','awaiting_verification','awaiting_return_to_service']
        for row in conn.execute('SELECT w.id,w.content,COALESCE(e.status,w.status) AS status FROM work_orders w JOIN assets a ON a.id=w.asset_id LEFT JOIN work_execution e ON e.work_order_id=w.id WHERE a.plant_id=?',(user['plant_id'],)):
            if row['status'] in blocked and wid in json.loads(row['content'])['dependencies']:
                raise HTTPException(409,'Resolve dependent open work first; changing this job would invalidate its dependency')
        assignments=[dict(r) for r in conn.execute('SELECT * FROM schedule_assignments WHERE work_order_id=?',(wid,))]
        users=[dict(r) for r in conn.execute('SELECT * FROM work_user_assignments WHERE work_order_id=?',(wid,))]
        now=iso();version=work['execution']['version']+1;disposition='cancelled' if body.action=='cancel' else 'released_for_replanning'
        before=dict(status=work['status'],assignments=assignments,user_assignments=users,operations=ops)
        for a in assignments:
            conn.execute('INSERT INTO schedule_assignment_archive VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(*[a[k] for k in ['id','proposal_id','work_order_id','asset_id','start_utc','end_utc','technician_ids','parts']],disposition,user['id'],now,body.reason))
        conn.execute('DELETE FROM schedule_assignments WHERE work_order_id=?',(wid,));conn.execute('DELETE FROM work_user_assignments WHERE work_order_id=?',(wid,))
        # Invalidate every pending proposal containing this job, including older alternatives.
        for p in conn.execute("SELECT id,snapshot FROM schedule_proposals WHERE plant_id=? AND status='pending'",(user['plant_id'],)).fetchall():
            if any(w['id']==wid for w in json.loads(p['snapshot'])['orders']):
                conn.execute("UPDATE schedule_proposals SET status='rejected',reviewed_by=?,reviewed_at=?,review_reason=? WHERE id=?",(user['id'],now,'Superseded by '+body.action+': '+body.reason,p['id']))
        if ops:
            ov=ops['version']+1
            conn.execute('UPDATE operations_handovers SET status=?,version=?,actor_id=?,updated_at=? WHERE work_order_id=?',('cancelled' if body.action=='cancel' else 'replanning',ov,user['id'],now,wid))
            conn.execute('INSERT INTO operations_events VALUES(?,?,?,?,?,?,?,?)',(identifier(),wid,ops['request_id'],user['id'],now,body.action,body.reason,ov))
        status='cancelled' if body.action=='cancel' else 'approved'
        if body.action=='replan':conn.execute("UPDATE work_orders SET status='approved' WHERE id=?",(wid,))
        conn.execute('INSERT INTO work_execution VALUES(?,?,?,?) ON CONFLICT(work_order_id) DO UPDATE SET status=excluded.status,version=excluded.version,updated_at=excluded.updated_at',(wid,status,version,now))
        conn.execute('INSERT INTO work_events VALUES(?,?,?,?,?,?,?,?)',(identifier(),wid,user['id'],now,body.action,status,body.reason,version))
        conn.execute('INSERT INTO work_lifecycle_events VALUES(?,?,?,?,?,?,?,?)',(identifier(),wid,version,user['id'],now,body.action,body.reason,json.dumps(before)))
        audit(conn,user,'work_lifecycle_'+body.action,wid,body.reason+'; unstarted reservations released, evidence retained')
        return order(conn,user,wid)


def record_reconfirmation(conn,wid,user):
    """Advance the shared version so stale field requests cannot act on a new schedule."""
    row=conn.execute('SELECT * FROM work_execution WHERE work_order_id=?',(wid,)).fetchone()
    if not row:return
    version=row['version']+1;now=iso()
    conn.execute("UPDATE work_execution SET status='scheduled',version=?,updated_at=? WHERE work_order_id=?",(version,now,wid))
    conn.execute('INSERT INTO work_events VALUES(?,?,?,?,?,?,?,?)',(identifier(),wid,user['id'],now,'schedule_reconfirmed','scheduled','Manager confirmed a replacement schedule; any required operations release must be recorded again.',version))
