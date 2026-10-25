"""Immutable observations and auditable quality decisions; never invent missing data."""
from datetime import datetime,timedelta
from contextlib import nullcontext
import sqlite3
from pydantic import ValidationError
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.core.security import authorize_asset,authorize_source,allowed_source_ids
from backend.ingestion.contracts import TelemetryRecord


def ingest(records,user,source_id,now=None,connection=None):
    now=now or utcnow();results=[]
    with (nullcontext(connection) if connection is not None else transaction()) as conn:
        source=authorize_source(conn,user,source_id)
        for raw in records:
            asset_id=raw.get('asset_id','')
            if not isinstance(asset_id,str) or not asset_id:
                raise HTTPException(422,'A valid asset identifier is required')
            authorize_asset(conn,user,asset_id)
            record=None;status='accepted';reason='valid'
            try:
                record=TelemetryRecord.model_validate(raw)
                if record.source != source_id:raise ValueError('Record source does not match approved import source')
                sensor=conn.execute('SELECT * FROM sensors WHERE asset_id=? AND tag_id=?',(record.asset_id,record.tag_id)).fetchone()
                if sensor is None:raise ValueError('Unmapped sensor')
                mapped=conn.execute('SELECT kind FROM sources WHERE id=?',(sensor['source_id'],)).fetchone()
                if sensor['source_id']!=source_id and (source['kind']!='file' or mapped['kind']=='gateway'):
                    raise ValueError('Source does not match the sensor gateway mapping')
                if record.unit != sensor['unit']:raise ValueError('Unit does not match sensor contract')
                if not sensor['lower_bound'] <= record.value <= sensor['upper_bound']:raise ValueError('Outside configured physical bounds')
                if record.timestamp_utc > now+timedelta(seconds=30):raise ValueError('Future observation')
                previous=conn.execute('SELECT * FROM telemetry WHERE asset_id=? AND tag_id=? ORDER BY timestamp_utc DESC LIMIT 1',(record.asset_id,record.tag_id)).fetchone()
                if previous and record.timestamp_utc < datetime.fromisoformat(previous['timestamp_utc']):raise ValueError('Out-of-order observation')
                if previous and previous['timestamp_utc']==iso(record.timestamp_utc):raise ValueError('Duplicate observation')
                if record.quality!='good':status='flagged';reason='Sensor quality '+record.quality
                elif (now-record.timestamp_utc).total_seconds()>sensor['stale_after_seconds']:status='flagged';reason='Stale observation'
                elif previous and previous['operating_state']!=record.operating_state:status='flagged';reason='Operating state changed; review applicability'
                conn.execute('INSERT INTO telemetry VALUES(?,?,?,?,?,?,?,?,?,?)',
                             (identifier(),record.asset_id,record.tag_id,iso(record.timestamp_utc),record.value,
                              record.unit,record.quality,record.source,record.operating_state,iso(now)))
                conn.execute('UPDATE assets SET operating_state=? WHERE id=?',(record.operating_state,record.asset_id))
                conn.execute('UPDATE sources SET last_received=?,status=? WHERE id=?',(iso(now),'connected',source_id))
            except (ValidationError,ValueError,sqlite3.IntegrityError) as exc:
                status='rejected';reason=str(exc)[:600]
            conn.execute('INSERT INTO ingestion_events VALUES(?,?,?,?,?,?,?,?)',
                         (identifier(),source['plant_id'],asset_id,str(raw.get('tag_id','')),source_id,iso(now),status,reason))
            results.append({'asset_id':asset_id,'tag_id':raw.get('tag_id') if isinstance(raw.get('tag_id'),str) else None,'status':status,'reason':reason})
        audit(conn,user,'ingest',source_id,f'{sum(r["status"]!="rejected" for r in results)}/{len(results)} accepted')
    return results


def latest_telemetry(conn,user,asset_id,now=None):
    authorize_asset(conn,user,asset_id);now=now or utcnow();output=[]
    sources=allowed_source_ids(conn,user);slots=','.join('?' for _ in sources)
    for sensor in conn.execute('SELECT * FROM sensors WHERE asset_id=? ORDER BY tag_id',(asset_id,)):
        row=conn.execute(f'SELECT * FROM telemetry WHERE asset_id=? AND tag_id=? AND source IN ({slots}) ORDER BY timestamp_utc DESC LIMIT 1',(asset_id,sensor['tag_id'],*sources)).fetchone()
        event=conn.execute(f'SELECT status,reason FROM ingestion_events WHERE asset_id=? AND tag_id=? AND source_id IN ({slots}) ORDER BY timestamp_utc DESC,rowid DESC LIMIT 1',(asset_id,sensor['tag_id'],*sources)).fetchone()
        data=dict(row) if row else None
        quality='missing' if not row else row['quality']
        if row and (now-datetime.fromisoformat(row['timestamp_utc'])).total_seconds()>sensor['stale_after_seconds']:quality='stale'
        output.append({'sensor':dict(sensor),'observation':data,'effective_quality':quality,'last_ingestion':dict(event) if event else None})
    return output
