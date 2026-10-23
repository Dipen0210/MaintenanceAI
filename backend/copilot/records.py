"""CMMS CSV adapter for actual imported records, including explicitly simulated fixtures."""
import csv,io,json,sqlite3
from fastapi import HTTPException
from pydantic import ValidationError
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.core.security import authorize_asset,authorize_source
from backend.copilot.contracts import RecordInput
from backend.copilot.safety import quarantine_reason


def import_csv(content,source_id,user):
    if len(content)>5*1024*1024:raise HTTPException(413,'CMMS CSV exceeds 5 MiB')
    try:
        rows=list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
        if not rows or len(rows)>10000:raise ValueError('Expected 1–10000 CMMS rows')
        records=[]
        for raw in rows:
            raw['details']=json.loads(raw.get('details') or '{}')
            record=RecordInput.model_validate(raw)
            if record.timestamp_utc>utcnow():raise ValueError('Record timestamps describe historical events, not future observations')
            if len(json.dumps(record.details))>10000:raise ValueError('CMMS detail object is too large')
            if quarantine_reason(record.summary+' '+json.dumps(record.details)):raise ValueError('Embedded instructions are not accepted as maintenance records')
            records.append(record)
    except (ValueError,UnicodeError,ValidationError) as exc:raise HTTPException(422,'Invalid CMMS record: '+str(exc)[:250]) from exc
    ids=[]
    with transaction() as conn:
        source=authorize_source(conn,user,source_id)
        if source['kind']!='file':raise HTTPException(422,'CMMS import requires an approved file source')
        for record in records:
            authorize_asset(conn,user,record.asset_id);rid=identifier()
            try:
                conn.execute('INSERT INTO maintenance_records VALUES(?,?,?,?,?,?,?,?,?)',
                             (rid,record.asset_id,source_id,record.external_id,record.record_type,iso(record.timestamp_utc),record.summary,json.dumps(record.details),iso()))
            except sqlite3.IntegrityError as exc:raise HTTPException(409,'Duplicate CMMS source/external identifier') from exc
            ids.append(rid)
        audit(conn,user,'import_cmms',source_id,str(len(ids))+' records')
    return {'imported':len(ids),'ids':ids,'source_id':source_id}


def records(conn,user,asset_id,kinds=None,limit=20):
    authorize_asset(conn,user,asset_id)
    rows=conn.execute('''SELECT r.* FROM maintenance_records r JOIN source_grants g ON r.source_id=g.source_id
                        WHERE r.asset_id=? AND g.user_id=? ORDER BY r.timestamp_utc DESC LIMIT 1000''',(asset_id,user['id']))
    return [{**dict(row),'details':json.loads(row['details'])} for row in rows if kinds is None or row['record_type'] in kinds][:limit]
