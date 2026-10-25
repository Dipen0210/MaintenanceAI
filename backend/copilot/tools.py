"""Eleven narrow read-only tools. No SQL, URL, credential, filesystem or control tool."""
import time
from fastapi import HTTPException
from backend.core.database import transaction,audit,iso
from backend.core.security import authorize_asset
from backend.ingestion.service import latest_telemetry
from backend.ml.service import history
from backend.risk.service import latest as latest_risk,config_row
from backend.copilot.records import records
from backend.copilot.documents import search
from backend.copilot.contracts import ToolArguments
from backend.scheduling.service import internal_records
from backend.inspections.service import records as inspection_records

TOOLS=frozenset(['get_asset_details','get_latest_telemetry','get_sensor_trends','get_recent_alarms','get_model_predictions',
                 'get_fault_history','search_approved_manuals','get_existing_work_orders','get_asset_risk','get_maintenance_schedule','draft_investigation_report'])


def invoke(name,arguments,user,trace=None):
    start=time.perf_counter()
    if name not in TOOLS:raise HTTPException(403,'Unsupported tool; industrial write and arbitrary access tools are forbidden')
    args=ToolArguments.model_validate(arguments)
    try:
        with transaction() as conn:
            asset=authorize_asset(conn,user,args.asset_id)
            if name=='get_asset_details':
                result={**asset,'criticality':config_row(conn,'asset_profiles','asset_id',args.asset_id),
                        'model_bindings':[dict(r) for r in conn.execute('SELECT * FROM model_bindings WHERE asset_id=?',(args.asset_id,))]}
            elif name=='get_latest_telemetry':result=latest_telemetry(conn,user,args.asset_id)
            elif name=='get_sensor_trends':
                result=[dict(r) for r in conn.execute('''SELECT t.* FROM telemetry t JOIN source_grants g ON t.source=g.source_id
                         WHERE t.asset_id=? AND g.user_id=? ORDER BY t.timestamp_utc DESC LIMIT ?''',(args.asset_id,user['id'],args.limit))]
            elif name=='get_recent_alarms':
                result=[dict(r) for r in conn.execute('''SELECT a.* FROM alarms a JOIN source_grants g ON a.source=g.source_id
                         WHERE a.asset_id=? AND g.user_id=? ORDER BY a.timestamp_utc DESC LIMIT ?''',(args.asset_id,user['id'],args.limit))]
            elif name=='get_model_predictions':result=history(conn,user,args.asset_id,args.limit)
            elif name=='get_fault_history':result=sorted(inspection_records(conn,user,args.asset_id,args.limit)+records(conn,user,args.asset_id,['fault','inspection'],args.limit),key=lambda r:r['timestamp_utc'],reverse=True)[:args.limit]
            elif name=='get_existing_work_orders':result=(internal_records(conn,user,args.asset_id,'work_order',args.limit)+records(conn,user,args.asset_id,['work_order'],args.limit))[:args.limit]
            elif name=='get_maintenance_schedule':result=(internal_records(conn,user,args.asset_id,'schedule',args.limit)+records(conn,user,args.asset_id,['schedule'],args.limit))[:args.limit]
            elif name=='get_asset_risk':result=latest_risk(conn,user,args.asset_id)
            elif name=='draft_investigation_report':
                result={'asset_id':args.asset_id,'status':'unsaved draft','as_of':iso(),'observations':latest_telemetry(conn,user,args.asset_id),
                        'predictions':history(conn,user,args.asset_id,5),'maintenance_records':records(conn,user,args.asset_id,limit=10),
                        'risk':latest_risk(conn,user,args.asset_id),'human_review_required':True}
            else:result=None
            audit(conn,user,'copilot_tool',name+':'+args.asset_id,'allowed')
        if name=='search_approved_manuals':result=search(user,args.asset_id,args.query,min(args.limit,12))
        if trace is not None:trace.append({'tool':name,'asset_id':args.asset_id,'status':'succeeded','latency_ms':round((time.perf_counter()-start)*1000,3)})
        return result
    except Exception as exc:
        if trace is not None:trace.append({'tool':name,'asset_id':args.asset_id,'status':'failed','error_type':type(exc).__name__,'latency_ms':round((time.perf_counter()-start)*1000,3)})
        raise
