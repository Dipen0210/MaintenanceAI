"""Human field-work records, independent from physical equipment and sensor condition."""
from datetime import datetime
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.copilot.safety import quarantine_reason


def snapshot(conn,wid):
    state=conn.execute('SELECT * FROM work_execution WHERE work_order_id=?',(wid,)).fetchone()
    events=[dict(r) for r in conn.execute('SELECT id,actor_id,created_at,action,status,notes,version FROM work_events WHERE work_order_id=? ORDER BY version',(wid,))]
    assignment=conn.execute('SELECT start_utc,end_utc FROM schedule_assignments WHERE work_order_id=?',(wid,)).fetchone()
    return dict(version=state['version'] if state else 0,status=state['status'] if state else None,
                updated_at=state['updated_at'] if state else None,events=events,window=dict(assignment) if assignment else None)


def update(wid,body,user):
    from backend.scheduling.service import order,role
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE')
        role(conn,user,['administrator','engineer','reliability_engineer','maintenance_manager'])
        work=order(conn,user,wid);state=work['execution'];status=work['status']
        if state['version']!=body.expected_version:raise HTTPException(409,'Work changed. Refresh and review the latest handover before updating it.')
        if status not in ['scheduled','in_progress','awaiting_verification','completed']:raise HTTPException(409,'Field updates require a manager-confirmed schedule')
        if quarantine_reason(body.notes):raise HTTPException(422,'Unsafe instructions cannot be recorded as work guidance')
        action=body.action
        if action in ['start','report_fixed']:
            assigned={r[0] for r in conn.execute('SELECT user_id FROM work_user_assignments WHERE work_order_id=?',(wid,))}
            if assigned and user['id'] not in assigned:raise HTTPException(403,'Only an assigned engineer may execute this work')
        if action=='start':
            role(conn,user,['engineer','reliability_engineer'])
            if status!='scheduled':raise HTTPException(409,'Only scheduled work can be started')
            if not body.site_clearance_confirmed:raise HTTPException(422,'Confirm operations coordination, site permits and isolation before recording a start')
            window=state['window']
            if not window:raise HTTPException(409,'A confirmed assignment is required')
            mode=conn.execute('SELECT mode FROM plants WHERE id=?',(user['plant_id'],)).fetchone()[0]
            from backend.core.workspaces import asset_workspace
            if (mode!='simulation' or asset_workspace(conn,work['asset_id'])!='demo') and not datetime.fromisoformat(window['start_utc'])<=utcnow()<datetime.fromisoformat(window['end_utc']):
                raise HTTPException(409,'Work must start within its approved window. Coordinate an approved maintenance window with operations.')
            for dependency in work['content']['dependencies']:
                if order(conn,user,dependency)['status']!='completed':raise HTTPException(409,'Dependent maintenance must be verified complete before starting')
            status='in_progress'
        elif action=='report_fixed':
            role(conn,user,['engineer','reliability_engineer'])
            if status!='in_progress':raise HTTPException(409,'Start the scheduled work before reporting a repair')
            status='awaiting_verification'
        elif action in ['verify','return_to_work']:
            role(conn,user,['maintenance_manager'])
            if status!='awaiting_verification':raise HTTPException(409,'A repair report must be awaiting verification')
            report=next(e for e in reversed(state['events']) if e['action']=='report_fixed')
            if report['actor_id']==user['id']:raise HTTPException(403,'A different maintenance manager must verify this repair report')
            status='completed' if action=='verify' else 'in_progress'
        elif action=='note' and status=='completed':raise HTTPException(409,'Verified work is closed. Create a new work order for further issues.')
        version=state['version']+1;now=iso()
        conn.execute('INSERT INTO work_execution VALUES(?,?,?,?) ON CONFLICT(work_order_id) DO UPDATE SET status=excluded.status,version=excluded.version,updated_at=excluded.updated_at',(wid,status,version,now))
        conn.execute('INSERT INTO work_events VALUES(?,?,?,?,?,?,?,?)',(identifier(),wid,user['id'],now,action,status,body.notes,version))
        audit(conn,user,'field_work_'+action,wid,f'{status}; version={version}; human record only, sensor evidence unchanged')
        return order(conn,user,wid)
