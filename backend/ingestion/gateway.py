"""Worker entry point for approved gateway reads into the shared validation path."""
import inspect
from backend.core.database import transaction,audit
from backend.core.security import authorize_source
from backend.ingestion.service import ingest


async def collect(adapter,user,source_id):
    with transaction() as conn:
        source=authorize_source(conn,user,source_id)
        if source['kind'] not in ['opcua','historian'] or source['read_only']!=1:
            raise ValueError('Approved read-only gateway source required')
        audit(conn,user,'gateway_read',source_id,'started')
    try:
        records=adapter.records()
        if inspect.isawaitable(records):records=await records
        records=list(records)
        if len(records)>10000:raise ValueError('Gateway batch exceeds limit')
        return {'status':'processed','results':ingest(records,user,source_id)}
    except Exception as exc:
        with transaction() as conn:
            conn.execute('UPDATE sources SET status=? WHERE id=?',('disconnected',source_id))
            audit(conn,user,'gateway_read',source_id,'failed:'+type(exc).__name__)
        # Exception details may contain connection credentials: never expose them.
        return {'status':'unavailable','error':'Gateway read failed; no measurements invented'}
