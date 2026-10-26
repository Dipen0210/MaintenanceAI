"""Human-reviewed supplier facts and quotations. No web scraping or autonomous purchases."""
import json
from datetime import datetime
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.inventory.service import scope_access,enabled,part
from backend.inventory.catalog import details


def access(conn,user,roles=None):
    scope=scope_access(conn,user,roles) if roles else scope_access(conn,user)
    if not enabled(conn,user['plant_id'],scope):raise HTTPException(409,'Activate counted inventory first')
    return scope


def event(conn,user,scope,rid,action,reason,snapshot):
    conn.execute('INSERT INTO procurement_events VALUES(?,?,?,?,?,?,?,?,?)',(identifier(),user['plant_id'],scope,rid,action,user['id'],iso(),reason,json.dumps(snapshot)))
    audit(conn,user,action,rid,reason)


def replay(conn,user,body,operation):
    payload=json.dumps(dict(operation=operation,plant_id=user['plant_id'],workspace=user['workspace'],body=body.model_dump(mode='json')),sort_keys=True)
    old=conn.execute('SELECT * FROM procurement_commands WHERE actor_id=? AND request_key=?',(user['id'],body.request_key)).fetchone()
    if old and old['payload']!=payload:raise HTTPException(409,'Request key was already used for different details')
    return payload,json.loads(old['response']) if old else None


def remember(conn,user,key,payload,result):
    conn.execute('INSERT INTO procurement_commands VALUES(?,?,?,?)',(user['id'],key,payload,json.dumps(result)));return result


def resource(conn,table,rid,user):
    if table not in ['procurement_suppliers','procurement_quotes','procurement_requests','procurement_orders']:raise ValueError('Unknown resource')
    row=conn.execute(f'SELECT * FROM {table} WHERE id=? AND plant_id=? AND workspace=?',(rid,user['plant_id'],user['workspace'])).fetchone()
    if not row:raise HTTPException(404,'Procurement record unavailable in this workspace')
    return {**dict(row),'content':json.loads(row['content'])}


def supplier_create(body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope=access(conn,user);payload,old=replay(conn,user,body,'supplier')
        if old:return old
        rid=identifier();data=body.model_dump(exclude={'request_key','reason'})
        conn.execute('INSERT INTO procurement_suppliers VALUES(?,?,?,?,?,1,?,NULL,?)',(rid,user['plant_id'],scope,json.dumps(data),'draft',user['id'],iso()))
        event(conn,user,scope,rid,'supplier_created',body.reason,data)
        return remember(conn,user,body.request_key,payload,dict(id=rid))


def supplier_review(sid,body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope=access(conn,user,['maintenance_manager']);s=resource(conn,'procurement_suppliers',sid,user)
        if s['version']!=body.expected_version:raise HTTPException(409,'Supplier changed; refresh before review')
        if body.action=='approve' and not body.verified:raise HTTPException(422,'Confirm supplier identity and contact details were checked')
        status='approved' if body.action=='approve' else 'suspended'
        conn.execute('UPDATE procurement_suppliers SET status=?,version=version+1,reviewed_by=? WHERE id=?',(status,user['id'],sid))
        event(conn,user,scope,sid,'supplier_'+status,body.reason,s['content']);return dict(id=sid,status=status)


def quote_current(conn,user,q):
    s=resource(conn,'procurement_suppliers',q['supplier_id'],user);p=part(conn,user['plant_id'],user['workspace'],q['part_id']);unit=details(conn,user['plant_id'],user['workspace'],p['part_id'])['content']['unit']
    reasons=[]
    if q['status']!='verified':reasons.append('Quotation is not verified')
    if s['status']!='approved':reasons.append('Supplier is not approved')
    if q['content']['supplier_version']!=s['version']:reasons.append('Supplier review changed; enter and verify a new quotation')
    if datetime.fromisoformat(q['content']['valid_until'])<=utcnow():reasons.append('Quotation expired')
    if q['content']['unit']!=unit:reasons.append('Stock unit changed')
    return reasons


def supplier_list(user):
    with transaction() as conn:
        scope=access(conn,user)
        suppliers=[resource(conn,'procurement_suppliers',r[0],user) for r in conn.execute('SELECT id FROM procurement_suppliers WHERE plant_id=? AND workspace=? ORDER BY created_at DESC',(user['plant_id'],scope))]
        quotes=[]
        for row in conn.execute('SELECT id FROM procurement_quotes WHERE plant_id=? AND workspace=? ORDER BY created_at DESC',(user['plant_id'],scope)):
            q=resource(conn,'procurement_quotes',row[0],user);q['limitations']=quote_current(conn,user,q);quotes.append(q)
        return dict(suppliers=suppliers,quotes=quotes)


def quote_create(body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope=access(conn,user);s=resource(conn,'procurement_suppliers',body.supplier_id,user);part(conn,user['plant_id'],scope,body.part_id)
        payload,old=replay(conn,user,body,'quote')
        if old:return old
        if s['status']!='approved':raise HTTPException(409,'Approve the supplier before recording a quotation')
        if body.valid_until<=utcnow():raise HTTPException(422,'Quotation must have a future expiry')
        data=body.model_dump(mode='json',exclude={'supplier_id','part_id','request_key','reason'});data.update(supplier_version=s['version'],unit=details(conn,user['plant_id'],scope,body.part_id)['content']['unit'])
        rid=identifier();conn.execute('INSERT INTO procurement_quotes VALUES(?,?,?,?,?,?,?,1,?,NULL,?)',(rid,user['plant_id'],scope,body.supplier_id,body.part_id,json.dumps(data),'draft',user['id'],iso()))
        event(conn,user,scope,rid,'quote_recorded',body.reason,data);return remember(conn,user,body.request_key,payload,dict(id=rid))


def quote_review(qid,body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');scope=access(conn,user,['maintenance_manager']);q=resource(conn,'procurement_quotes',qid,user)
        if q['version']!=body.expected_version:raise HTTPException(409,'Quotation changed; refresh before review')
        if body.action=='verify':
            if q['status']!='draft' or not body.verified:raise HTTPException(409,'Verify a draft quotation against its actual supplier reference')
            reasons=quote_current(conn,user,{**q,'status':'verified'})
            if reasons:raise HTTPException(409,'; '.join(reasons))
        status='verified' if body.action=='verify' else 'withdrawn'
        conn.execute('UPDATE procurement_quotes SET status=?,version=version+1,verified_by=? WHERE id=?',(status,user['id'],qid))
        event(conn,user,scope,qid,'quote_'+status,body.reason,q['content']);return dict(id=qid,status=status)


def compare(user,pid,quantity):
    if not 1<=quantity<=100000:raise HTTPException(422,'Quantity must be from 1 to 100000 whole stock units')
    data=supplier_list(user);result=[]
    for q in data['quotes']:
        c=q['content']
        if q['part_id']!=pid or q['limitations'] or quantity<c['minimum_quantity'] or quantity>c['maximum_quantity']:continue
        result.append(dict(quote_id=q['id'],supplier_id=q['supplier_id'],supplier_name=next(s['content']['name'] for s in data['suppliers'] if s['id']==q['supplier_id']),currency=c['currency'],quantity=quantity,unit=c['unit'],unit_price_minor=c['unit_price_minor'],total_minor=c['unit_price_minor']*quantity+c['shipping_minor']+c['tax_minor'],lead_time_days=c['lead_time_days'],reference=c['reference'],valid_until=c['valid_until']))
    return dict(options=sorted(result,key=lambda q:(q['currency'],q['total_minor'],q['lead_time_days'])),basis='Verified supplier quotations only. Compare totals within the same currency; no exchange-rate assumptions. Tax and freight are fixed quoted amounts.')
