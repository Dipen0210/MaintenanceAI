"""Counted stock and work custody. All quantities are integer units, never estimated consumption."""
import json
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,audit
from backend.core.security import allowed_assets
from backend.core.workspaces import asset_workspace

STORES=['administrator','maintenance_manager','stores_coordinator']


def workspace(user):
    value=user.get('workspace')
    if value not in ['demo','plant']:raise HTTPException(422,'Select the Demo or Plant workspace for stock operations')
    return value


def enabled(conn,plant,scope):return bool(conn.execute('SELECT 1 FROM inventory_scopes WHERE plant_id=? AND workspace=?',(plant,scope)).fetchone())


def scope_access(conn,user,roles=STORES):
    from backend.scheduling.service import role
    role(conn,user,roles);scope=workspace(user)
    all_ids={r['id'] for r in conn.execute('SELECT id FROM assets WHERE plant_id=?',(user['plant_id'],)) if asset_workspace(conn,r['id'])==scope}
    if not all_ids.issubset({a['id'] for a in allowed_assets(conn,user)}):raise HTTPException(403,'Shared stock requires access to every asset in this workspace')
    return scope


def status(user):
    with transaction() as conn:
        scope=workspace(user);manage=False
        try:scope_access(conn,user);manage=True
        except HTTPException:pass
        row=conn.execute('SELECT * FROM inventory_scopes WHERE plant_id=? AND workspace=?',(user['plant_id'],scope)).fetchone()
        return dict(enabled=bool(row),workspace=scope,can_manage=manage,activated_at=row['created_at'] if row else None)


def sync_catalog(conn,plant,scope,parts):
    for p in parts:conn.execute('INSERT OR IGNORE INTO inventory_parts(plant_id,workspace,part_id,name) VALUES(?,?,?,?)',(plant,scope,p['id'],p['name']))


def activate(body,user):
    from backend.scheduling.service import active_config
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope=scope_access(conn,user,['administrator','maintenance_manager'])
        if enabled(conn,user['plant_id'],scope):return {'enabled':True}
        if not body.counting_confirmed:raise HTTPException(422,'Confirm that opening quantities will be recorded from a counted stocktake')
        for row in conn.execute("SELECT w.id,w.asset_id,w.content,COALESCE(e.status,w.status) AS current_status FROM work_orders w JOIN assets a ON a.id=w.asset_id LEFT JOIN work_execution e ON e.work_order_id=w.id WHERE a.plant_id=?",(user['plant_id'],)):
            if asset_workspace(conn,row['asset_id'])==scope and row['current_status'] in ['scheduled','in_progress','awaiting_verification','awaiting_return_to_service'] and json.loads(row['content'])['parts']:
                raise HTTPException(409,'Finish or return unstarted parts-bearing work to planning before enabling counted inventory')
        config=active_config(conn,user)
        conn.execute('INSERT INTO inventory_scopes VALUES(?,?,?,?,?)',(user['plant_id'],scope,user['id'],iso(),body.reason));sync_catalog(conn,user['plant_id'],scope,config['config']['parts'])
        audit(conn,user,'activate_inventory',scope,body.reason+'; zero balances, no counted stock fabricated')
    return {'enabled':True}


def part(conn,plant,scope,pid):
    row=conn.execute('SELECT * FROM inventory_parts WHERE plant_id=? AND workspace=? AND part_id=?',(plant,scope,pid)).fetchone()
    if not row:raise HTTPException(409,'Part is not tracked in this workspace; add it to planning resources after activating inventory')
    return dict(row)


def allocation(conn,wid,pid):
    row=conn.execute('SELECT * FROM inventory_work_parts WHERE work_order_id=? AND part_id=?',(wid,pid)).fetchone()
    return dict(row) if row else dict(work_order_id=wid,part_id=pid,reserved=0,issued=0,consumed=0,returned=0)


def movement(conn,plant,scope,pid,wid,action,quantity,user,reason,key):
    existing=conn.execute('SELECT id FROM inventory_movements WHERE event_key=?',(key,)).fetchone()
    if existing:return existing['id']
    stock=part(conn,plant,scope,pid);work=allocation(conn,wid,pid) if wid else None;before=dict(stock=dict(stock),allocation=dict(work) if work else None)
    if action in ['receipt','adjust']:stock['on_hand']+=quantity
    elif action=='reserve':stock['reserved']+=quantity;work['reserved']+=quantity
    elif action=='release':stock['reserved']-=quantity;work['reserved']-=quantity
    elif action=='issue':stock['on_hand']-=quantity;stock['reserved']-=quantity;work['reserved']-=quantity;work['issued']+=quantity
    elif action=='consume':work['issued']-=quantity;work['consumed']+=quantity
    elif action=='return':stock['on_hand']+=quantity;work['issued']-=quantity;work['returned']+=quantity
    else:raise ValueError('Unknown stock action')
    if stock['on_hand']<stock['reserved'] or min(stock['on_hand'],stock['reserved'])<0 or stock['on_hand']>1000000000 or work and min(work[k] for k in ['reserved','issued','consumed','returned'])<0:
        raise HTTPException(409,'Insufficient available, reserved or issued stock; review the latest balances')
    stock['version']+=1
    conn.execute('UPDATE inventory_parts SET on_hand=?,reserved=?,version=? WHERE plant_id=? AND workspace=? AND part_id=?',(stock['on_hand'],stock['reserved'],stock['version'],plant,scope,pid))
    if work:conn.execute('INSERT INTO inventory_work_parts VALUES(?,?,?,?,?,?) ON CONFLICT(work_order_id,part_id) DO UPDATE SET reserved=excluded.reserved,issued=excluded.issued,consumed=excluded.consumed,returned=excluded.returned',tuple(work[k] for k in ['work_order_id','part_id','reserved','issued','consumed','returned']))
    mid=identifier();conn.execute('INSERT INTO inventory_movements VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(mid,key,plant,scope,pid,wid,action,quantity,user['id'],iso(),reason,json.dumps(before),json.dumps(dict(stock=stock,allocation=work))))
    audit(conn,user,'inventory_'+action,mid,reason)
    return mid


def work_scope(conn,wid):
    row=conn.execute('SELECT a.plant_id,w.asset_id,w.content FROM work_orders w JOIN assets a ON a.id=w.asset_id WHERE w.id=?',(wid,)).fetchone()
    return row['plant_id'],asset_workspace(conn,row['asset_id']),json.loads(row['content'])


def summary(conn,wid):
    plant,scope,content=work_scope(conn,wid)
    if not enabled(conn,plant,scope):return None
    rows=[]
    for pid,planned in content['parts'].items():
        a=allocation(conn,wid,pid);p=conn.execute('SELECT name FROM inventory_parts WHERE plant_id=? AND workspace=? AND part_id=?',(plant,scope,pid)).fetchone();rows.append({**a,'name':p['name'] if p else pid,'planned':planned})
    return dict(workspace=scope,parts=rows,history=[dict(r) for r in conn.execute('SELECT id,part_id,action,quantity,actor_id,created_at,reason FROM inventory_movements WHERE work_order_id=? ORDER BY rowid',(wid,))])


def reserve(conn,wid,user,key):
    plant,scope,content=work_scope(conn,wid)
    if not enabled(conn,plant,scope):return
    for pid,quantity in content['parts'].items():
        a=allocation(conn,wid,pid);remaining=quantity-a['reserved']-a['issued']-a['consumed']
        if remaining>0:movement(conn,plant,scope,pid,wid,'reserve',remaining,user,'Reserve approved work scope',key+':'+pid)


def release(conn,wid,user,key):
    plant,scope,_=work_scope(conn,wid)
    if not enabled(conn,plant,scope):return
    rows=conn.execute('SELECT * FROM inventory_work_parts WHERE work_order_id=?',(wid,)).fetchall()
    if any(r['issued'] for r in rows):raise HTTPException(409,'Reconcile issued parts first: record actual consumption or return unused parts to stores')
    for r in rows:
        if r['reserved']:movement(conn,plant,scope,r['part_id'],wid,'release',r['reserved'],user,'Release unused work reservation',key+':'+r['part_id'])


def planning_stock(conn,user,config,orders):
    scopes={asset_workspace(conn,w['asset_id']) for w in orders}
    if not any(enabled(conn,user['plant_id'],s) for s in scopes):return config
    if len(scopes)!=1:raise HTTPException(409,'Counted inventory schedules cannot mix Demo and Plant stock')
    scope=next(iter(scopes));parts=[]
    for p in config['parts']:
        s=part(conn,user['plant_id'],scope,p['id']);parts.append({**p,'quantity':s['on_hand']})
    return {**config,'parts':parts,'inventory_workspace':scope}


def reserved_parts(conn,wid,original):
    plant,scope,_=work_scope(conn,wid)
    if not enabled(conn,plant,scope):return original
    return {r['part_id']:r['reserved'] for r in conn.execute('SELECT * FROM inventory_work_parts WHERE work_order_id=?',(wid,)) if r['reserved']}


def catalog(user):
    from backend.scheduling.service import order
    with transaction() as conn:
        scope=scope_access(conn,user);rows=[{**dict(r),'available':r['on_hand']-r['reserved']} for r in conn.execute('SELECT * FROM inventory_parts WHERE plant_id=? AND workspace=? ORDER BY name',(user['plant_id'],scope))];history=[]
        for r in conn.execute('SELECT id,part_id,work_order_id,action,quantity,actor_id,created_at,reason FROM inventory_movements WHERE plant_id=? AND workspace=? ORDER BY rowid DESC LIMIT 200',(user['plant_id'],scope)):
            if r['work_order_id']:
                try:order(conn,user,r['work_order_id'])
                except HTTPException:continue
            history.append(dict(r))
        return dict(parts=rows,history=history)


def apply(pid,body,user,wid=None):
    from backend.scheduling.service import role,order
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope=workspace(user);w=None
        if wid:
            w=order(conn,user,wid);plant,actual_scope,content=work_scope(conn,wid)
            if actual_scope!=scope:raise HTTPException(404,'Work not in the selected workspace')
            if body.action=='consume':
                role(conn,user,['engineer','reliability_engineer','stores_coordinator','maintenance_manager'])
                current=conn.execute('SELECT role FROM users WHERE id=?',(user['id'],)).fetchone()[0]
                if current in ['engineer','reliability_engineer']:
                    if not conn.execute('SELECT 1 FROM work_user_assignments WHERE work_order_id=? AND user_id=?',(wid,user['id'])).fetchone():raise HTTPException(403,'Only a named assigned engineer may record parts usage')
                else:scope_access(conn,user)
            else:scope_access(conn,user)
        else:scope_access(conn,user);plant=user['plant_id']
        if not enabled(conn,plant,scope):raise HTTPException(409,'Activate counted inventory first')
        payload=json.dumps(dict(part_id=pid,work_order_id=wid,workspace=scope,**body.model_dump()),sort_keys=True)
        old=conn.execute('SELECT * FROM inventory_commands WHERE actor_id=? AND request_key=?',(user['id'],body.request_key)).fetchone()
        if old:
            if old['payload']!=payload:raise HTTPException(409,'Request key was already used for different stock movement details')
            return {'movement_id':old['movement_id'],'replayed':True}
        stock=part(conn,plant,scope,pid)
        if wid:
            if pid not in content['parts']:raise HTTPException(422,'Part is outside the approved work scope')
            if body.expected_version!=w['execution']['version']:raise HTTPException(409,'Work changed; refresh before recording parts')
            states=['in_progress','awaiting_verification'] if body.action=='consume' else ['scheduled','in_progress','awaiting_verification']
            if body.action not in ['issue','consume','return'] or w['status'] not in states:raise HTTPException(409,'This stock movement is not permitted at the current work stage')
        else:
            if body.action not in ['receipt','adjust']:raise HTTPException(422,'This movement requires a work order')
            if body.expected_version!=stock['version']:raise HTTPException(409,'Stock changed; refresh the current balance before recording a movement')
        mid=movement(conn,plant,scope,pid,wid,body.action,body.quantity,user,body.reason,'command:'+user['id']+':'+body.request_key)
        if wid:
            version=w['execution']['version']+1;now=iso()
            conn.execute('INSERT INTO work_execution VALUES(?,?,?,?) ON CONFLICT(work_order_id) DO UPDATE SET version=excluded.version,updated_at=excluded.updated_at',(wid,w['status'],version,now))
            conn.execute('INSERT INTO work_events VALUES(?,?,?,?,?,?,?,?)',(identifier(),wid,user['id'],now,'parts_'+body.action,w['status'],body.reason,version))
        conn.execute('INSERT INTO inventory_commands VALUES(?,?,?,?)',(user['id'],body.request_key,payload,mid))
        return {'movement_id':mid,'replayed':False}
