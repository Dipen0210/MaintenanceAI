"""Evidence-linked maintenance planning. Only manager approval reserves resources."""
import json
from datetime import datetime
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.core.workspaces import asset_workspace
from backend.core.security import authorize_asset,authorize_source,allowed_assets
from backend.copilot.documents import authorize_document
from backend.copilot.safety import quarantine_reason
from backend.risk.service import latest
from backend.scheduling.solver import solve,validate


def role(conn,user,roles):
    current=conn.execute('SELECT role,enabled,plant_id FROM users WHERE id=?',(user['id'],)).fetchone()
    if not current or not current['enabled'] or current['role'] not in roles or current['plant_id']!=user['plant_id']:
        raise HTTPException(403,'Current role does not permit this planning action')


def procedures(conn,user,asset_id,ids):
    result=[]
    for cid in ids:
        row=conn.execute("SELECT * FROM document_chunks WHERE id=? AND reviewed=1 AND kind='check'",(cid,)).fetchone()
        if not row:raise HTTPException(404,'Approved procedure reference unavailable')
        doc=authorize_document(conn,user,row['document_id'],asset_id)
        if quarantine_reason(row['content']):raise HTTPException(422,'Unsafe procedure content')
        result.append(dict(id=cid,document_id=doc['id'],title=doc['title'],revision=doc['revision'],page=row['page'],text=row['content']))
    return result


def order(conn,user,wid):
    row=conn.execute('SELECT * FROM work_orders WHERE id=?',(wid,)).fetchone()
    if not row:raise HTTPException(404,'Work order not found or not authorized')
    authorize_asset(conn,user,row['asset_id'])
    value=dict(row);value['content']=json.loads(row['content']);data=value['content']
    for source in (data.get('risk_snapshot') or {}).get('source_ids',[]):authorize_source(conn,user,source)
    if data.get('preventive',{}).get('trigger')=='meter':authorize_source(conn,user,data['preventive']['source_id'])
    for dep in data['dependencies']:
        prior=conn.execute('SELECT asset_id FROM work_orders WHERE id=?',(dep,)).fetchone()
        if not prior:raise HTTPException(404,'Dependency unavailable')
        authorize_asset(conn,user,prior['asset_id'])
    procedures(conn,user,row['asset_id'],data['procedure_ids'])
    # A report link never grants access to its private notes; its source evidence is reauthorized.
    if data.get('investigation_id'):
        from backend.copilot.evidence import resolve
        report=conn.execute('SELECT evidence_ids FROM investigations WHERE id=?',(data['investigation_id'],)).fetchone()
        if not report:raise HTTPException(404,'Investigation unavailable')
        for reference in json.loads(report['evidence_ids']):resolve(conn,user,reference['asset_id'],reference['id'])
    from backend.scheduling.execution import snapshot
    value['execution']=snapshot(conn,wid)
    if value['execution']['status']:value['status']=value['execution']['status']
    return value


def list_orders(conn,user,asset_id=None,limit=100):
    if asset_id:authorize_asset(conn,user,asset_id)
    result=[]
    rows=conn.execute('''SELECT w.id FROM work_orders w JOIN assets a ON a.id=w.asset_id
                        WHERE a.plant_id=? AND (? IS NULL OR w.asset_id=?) ORDER BY w.created_at DESC LIMIT 1000''',
                      (user['plant_id'],asset_id,asset_id)).fetchall()
    for row in rows:
        try:result.append(order(conn,user,row['id']))
        except HTTPException:continue
        if len(result)>=limit:break
    return result


def create_draft(body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['administrator','engineer','reliability_engineer','maintenance_manager'])
        asset=authorize_asset(conn,user,body.asset_id)
        if quarantine_reason(body.title+' '+body.task):raise HTTPException(422,'Unsafe or embedded instructions cannot become a maintenance draft')
        risk=latest(conn,user,body.asset_id)
        if not risk or risk['id']!=body.risk_id or risk['stale']:raise HTTPException(409,'Re-evaluate current risk before creating a work order')
        refs=procedures(conn,user,body.asset_id,body.procedure_ids)
        for wid in body.dependencies:
            prior=order(conn,user,wid)
            if prior['status']=='rejected':raise HTTPException(409,'Rejected work cannot be a dependency')
        if body.investigation_id:
            report=conn.execute('SELECT * FROM investigations WHERE id=? AND asset_id=? AND author_id=?',
                                (body.investigation_id,body.asset_id,user['id'])).fetchone()
            if not report:raise HTTPException(404,'Investigation not found or not authorized')
            from backend.copilot.evidence import resolve
            for reference in json.loads(report['evidence_ids']):resolve(conn,user,reference['asset_id'],reference['id'])
        if body.due_at<=utcnow():raise HTTPException(422,'New work requires a future due time')
        mode=conn.execute('SELECT mode FROM plants WHERE id=?',(asset['plant_id'],)).fetchone()[0]
        data={**body.model_dump(mode='json'),'priority':risk['priority'],'risk_snapshot':risk,'procedures':refs,
              'simulation_only':mode=='simulation' and asset_workspace(conn,body.asset_id)=='demo','safety':'Draft scope only. A qualified person must confirm site isolation, permits and approved procedures before any physical work. Schedule approval does not operate equipment.'}
        wid=identifier();conn.execute('INSERT INTO work_orders VALUES(?,?,?,?,?,?,NULL,NULL,NULL)',
                                     (wid,body.asset_id,user['id'],iso(),'draft',json.dumps(data)))
        audit(conn,user,'draft_work_order',wid,'Evidence-linked draft; no resources reserved')
        return order(conn,user,wid)


def review_order(wid,body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['maintenance_manager'])
        current=order(conn,user,wid)
        if current['status']!='draft':raise HTTPException(409,'Only draft work orders may be reviewed')
        conn.execute('UPDATE work_orders SET status=?,reviewed_by=?,reviewed_at=?,review_reason=? WHERE id=?',
                     (body.decision,user['id'],iso(),body.reason,wid))
        audit(conn,user,'review_work_order',wid,body.decision+': '+body.reason)
        return order(conn,user,wid)


def active_config(conn,user):
    row=conn.execute('SELECT * FROM planning_configs WHERE plant_id=? AND active=1',(user['plant_id'],)).fetchone()
    if not row:raise HTTPException(409,'Configure planning resources and maintenance windows first')
    return dict(id=row['id'],created_at=row['created_at'],config=json.loads(row['config']))


def read_config(user):
    with transaction() as conn:
        result=active_config(conn,user);ids={a['id'] for a in allowed_assets(conn,user)}
        for key in ['equipment_windows','blackouts']:result['config'][key]=[w for w in result['config'][key] if w['asset_id'] in ids]
        return result


def save_config(body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['administrator','maintenance_manager'])
        from backend.routes.issues import eligible
        identities={u['id'] for u in eligible(conn,user)}
        mapped=[t.user_id for t in body.technicians if t.user_id]
        if len(mapped)!=len(set(mapped)) or any(uid not in identities for uid in mapped):raise HTTPException(422,'Map each enabled engineer to at most one technician resource')
        assets={r['id'] for r in conn.execute('SELECT id FROM assets WHERE plant_id=?',(user['plant_id'],))}
        if assets!={a['id'] for a in allowed_assets(conn,user)}:raise HTTPException(403,'Whole-plant configuration requires access to every plant asset')
        for row in [*body.equipment_windows,*body.blackouts]:authorize_asset(conn,user,row.asset_id)
        mode=conn.execute('SELECT mode FROM plants WHERE id=?',(user['plant_id'],)).fetchone()[0]
        if body.simulation_only!=(mode=='simulation'):raise HTTPException(422,'Planning mode must match the plant mode')
        rid=identifier();conn.execute('UPDATE planning_configs SET active=0 WHERE plant_id=?',(user['plant_id'],))
        conn.execute('INSERT INTO planning_configs VALUES(?,?,?,?,?,1)',(rid,user['plant_id'],json.dumps(body.model_dump(mode='json')),user['id'],iso()))
        audit(conn,user,'configure_planning',rid,body.reason)
        return active_config(conn,user)


def reservations(conn,plant_id):
    rows=conn.execute('''SELECT s.* FROM schedule_assignments s JOIN assets a ON a.id=s.asset_id WHERE a.plant_id=?''',(plant_id,))
    return [{**dict(r),'technician_ids':json.loads(r['technician_ids']),'parts':json.loads(r['parts'])} for r in rows]


def proposal(conn,user,pid):
    row=conn.execute('SELECT * FROM schedule_proposals WHERE id=? AND plant_id=?',(pid,user['plant_id'])).fetchone()
    if not row:raise HTTPException(404,'Schedule proposal not found or not authorized')
    snapshot=json.loads(row['snapshot'])
    for w in snapshot['orders']:order(conn,user,w['id'])
    value={k:json.loads(row[k]) if k=='result' else row[k] for k in ['id','created_at','config_id','status','result','reviewed_by','reviewed_at','review_reason']}
    for assignment in value['result'].get('assignments',[]):assignment['work_status']=order(conn,user,assignment['work_order_id'])['status']
    return value


def list_proposals(user):
    with transaction() as conn:
        result=[]
        for row in conn.execute('SELECT id FROM schedule_proposals WHERE plant_id=? ORDER BY created_at DESC LIMIT 100',(user['plant_id'],)).fetchall():
            try:result.append(proposal(conn,user,row['id']))
            except HTTPException:continue
        return result


def propose(body,user):
    if body.start_utc<=utcnow():raise HTTPException(422,'Planning must start in the future')
    with transaction() as conn:
        role(conn,user,['administrator','engineer','reliability_engineer','maintenance_manager'])
        config=active_config(conn,user);orders=[order(conn,user,wid) for wid in body.work_order_ids]
        if config['config']['simulation_only'] and any(not w['content']['simulation_only'] for w in orders):raise HTTPException(409,'Physical maintenance requires commissioned plant planning resources, not demo resources')
        if any(w['status']!='approved' for w in orders):raise HTTPException(409,'Every candidate must have maintenance manager approval')
        snapshot=dict(config=config['config'],orders=orders,existing=reservations(conn,user['plant_id']),
                      start_utc=iso(body.start_utc),end_utc=iso(body.end_utc))
    result=solve(snapshot)
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE')
        role(conn,user,['administrator','engineer','reliability_engineer','maintenance_manager'])
        for w in orders:order(conn,user,w['id'])
        pid=identifier();conn.execute('INSERT INTO schedule_proposals VALUES(?,?,?,?,?,?,?,?,NULL,NULL,NULL)',
            (pid,user['plant_id'],user['id'],iso(),config['id'],json.dumps(snapshot),json.dumps(result),'pending'))
        audit(conn,user,'propose_schedule',pid,result['feasibility']+'; no resources reserved')
        return proposal(conn,user,pid)


def review_proposal(pid,body,user):
    with transaction() as conn:
        # Serialize confirmation and independently check CURRENT reservations, preventing approval races.
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['maintenance_manager'])
        current=proposal(conn,user,pid)
        if current['status']!='pending':raise HTTPException(409,'Only pending proposals may be reviewed')
        if body.decision=='approved':
            if current['result']['feasibility']!='feasible':raise HTTPException(409,'Only feasible schedules may be confirmed')
            config=active_config(conn,user)
            if config['id']!=current['config_id']:raise HTTPException(409,'Planning configuration changed; generate a new proposal')
            snapshot=json.loads(conn.execute('SELECT snapshot FROM schedule_proposals WHERE id=?',(pid,)).fetchone()[0])
            for previous in snapshot['orders']:
                now=order(conn,user,previous['id'])
                if now['status']!='approved' or now['content']!=previous['content']:raise HTTPException(409,'Work order changed or already scheduled; generate a new proposal')
            assignments=current['result']['assignments']
            if any(datetime.fromisoformat(a['start_utc'])<=utcnow() for a in assignments):raise HTTPException(409,'Proposed time has passed; generate a new proposal')
            snapshot['existing']=reservations(conn,user['plant_id'])
            issues=validate(assignments,snapshot)
            if issues:raise HTTPException(409,'Resources changed; generate a new proposal. '+ '; '.join(issues))
            # Validate and freeze named users from the exact approved resource snapshot.
            technicians={t['id']:t for t in snapshot['config']['technicians']}
            bindings=[]
            for a in assignments:
                for tid in a['technician_ids']:
                    uid=technicians[tid].get('user_id')
                    if not uid:continue
                    person=conn.execute("SELECT id,username,role,plant_id FROM users WHERE id=? AND enabled=1 AND role IN('engineer','reliability_engineer') AND plant_id=?",(uid,user['plant_id'])).fetchone()
                    if not person:raise HTTPException(409,'Assigned technician identity is no longer available; revise resources')
                    order(conn,dict(person),a['work_order_id'])
                    bindings.append((a['work_order_id'],tid,uid,user['id'],iso()))
            conn.executemany('INSERT INTO work_user_assignments VALUES(?,?,?,?,?)',bindings)
            for a in assignments:
                conn.execute('INSERT INTO schedule_assignments VALUES(?,?,?,?,?,?,?,?)',(identifier(),pid,a['work_order_id'],a['asset_id'],a['start_utc'],a['end_utc'],json.dumps(a['technician_ids']),json.dumps(a['parts'])))
                conn.execute("UPDATE work_orders SET status='scheduled' WHERE id=?",(a['work_order_id'],))
        conn.execute('UPDATE schedule_proposals SET status=?,reviewed_by=?,reviewed_at=?,review_reason=? WHERE id=?',
                     (body.decision,user['id'],iso(),body.reason,pid))
        audit(conn,user,'review_schedule',pid,body.decision+': '+body.reason)
        return proposal(conn,user,pid)


def internal_records(conn,user,asset_id,kind,limit=20):
    """Read-only evidence for the copilot; private reports and other asset reservations are omitted."""
    rows=list_orders(conn,user,asset_id,100);result=[]
    for w in rows:
        if kind=='work_order':
            result.append(dict(id=w['id'],evidence_kind='work_order',external_id=w['id'],timestamp_utc=w['execution']['updated_at'] or w['created_at'],status=w['status']))
        elif w['status'] in ['scheduled','in_progress','awaiting_verification']:
            a=conn.execute('SELECT * FROM schedule_assignments WHERE work_order_id=?',(w['id'],)).fetchone()
            if a:result.append(dict(id=a['id'],evidence_kind='schedule',external_id=w['id'],timestamp_utc=a['start_utc'],status='approved'))
    return result[:limit]
