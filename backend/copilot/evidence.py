"""Resolve every citation through current asset, source and document authorization."""
import json
from fastapi import HTTPException
from backend.core.security import authorize_asset,authorize_source
from backend.copilot.documents import authorize_document


def resolve(conn,user,asset_id,evidence_id):
    authorize_asset(conn,user,asset_id)
    kind,separator,rid=evidence_id.partition(':')
    if not separator:raise HTTPException(404,'Evidence not found')
    base=dict(id=evidence_id,asset_id=asset_id,kind=kind,title='',text='',timestamp_utc=None,document_id=None,revision=None,page=None,section=None,model_version=None)
    if kind=='trend':
        first,sep,last=rid.partition('~')
        if not sep:raise HTTPException(404,'Trend evidence not found')
        rows=[]
        for key in [first,last]:
            row=conn.execute('SELECT * FROM telemetry WHERE id=? AND asset_id=?',(key,asset_id)).fetchone()
            if not row:raise HTTPException(404,'Trend evidence not found')
            authorize_source(conn,user,row['source']);rows.append(row)
        a,b=rows
        if a['tag_id']!=b['tag_id'] or a['unit']!=b['unit'] or a['timestamp_utc']>=b['timestamp_utc']:raise HTTPException(422,'Incompatible trend observations')
        return {**base,'title':a['tag_id']+' recorded change','role':'observation','timestamp_utc':b['timestamp_utc'],
                'text':f'{a["tag_id"]} changed from {a["value"]:g} {a["unit"]} at {a["timestamp_utc"]} to {b["value"]:g} {b["unit"]} at {b["timestamp_utc"]}; difference {b["value"]-a["value"]:g} {a["unit"]}. This comparison does not establish a cause.'}
    if kind=='document':
        row=conn.execute('SELECT * FROM document_chunks WHERE id=? AND reviewed=1',(rid,)).fetchone()
        if not row:raise HTTPException(404,'Evidence not found')
        doc=authorize_document(conn,user,row['document_id'],asset_id)
        return {**base,'title':doc['title'],'text':row['content'],'document_id':doc['id'],'revision':doc['revision'],
                'page':row['page'],'section':row['section'],'role':row['kind'],
                'required_fault_classes':json.loads(doc['metadata']).get('chunk_fault_classes',{}).get(rid,[])}
    if kind=='telemetry':
        row=conn.execute('SELECT * FROM telemetry WHERE id=? AND asset_id=?',(rid,asset_id)).fetchone()
        if not row:raise HTTPException(404,'Evidence not found')
        authorize_source(conn,user,row['source'])
        return {**base,'title':row['tag_id'],'text':f'{row["tag_id"]} measured {row["value"]:g} {row["unit"]} at {row["timestamp_utc"]}; source {row["source"]}, quality {row["quality"]}.',
                'timestamp_utc':row['timestamp_utc'],'role':'observation'}
    if kind=='prediction':
        row=conn.execute('SELECT * FROM predictions WHERE id=? AND asset_id=?',(rid,asset_id)).fetchone()
        if not row:raise HTTPException(404,'Evidence not found')
        if row['input_id']:
            source=conn.execute('SELECT source_id FROM signal_inputs WHERE id=?',(row['input_id'],)).fetchone()
            authorize_source(conn,user,source['source_id'])
        result=json.loads(row['result']);output=result['output']
        parts=[f'{result["family"]} version {result["model_version"]} reported {result["status"]}.']
        if output.get('fault_class'):parts.append('Candidate class: '+output['fault_class']+'.')
        if output.get('anomaly_score') is not None:parts.append(f'Normal-calibration anomaly percentile: {output["anomaly_score"]*100:.1f}%, not a failure probability.')
        if output.get('rul') is not None:parts.append(f'Dataset-specific RUL: {output["rul"]} cycles; no validated uncertainty interval.')
        parts.append(result['limitations'])
        return {**base,'title':result['family'],'text':' '.join(parts),'model_version':result['model_version'],'timestamp_utc':row['timestamp_utc'],'role':'prediction',
                'fault_class':output.get('fault_class'),'is_anomaly':output.get('is_anomaly',False)}
    if kind=='inspection':
        from backend.inspections.service import evidence
        return {**base,**evidence(conn,user,asset_id,rid)}
    if kind=='maintenance':
        row=conn.execute('SELECT * FROM maintenance_records WHERE id=? AND asset_id=?',(rid,asset_id)).fetchone()
        if not row:raise HTTPException(404,'Evidence not found')
        authorize_source(conn,user,row['source_id'])
        return {**base,'title':row['record_type']+' '+row['external_id'],'text':row['summary'],'timestamp_utc':row['timestamp_utc'],'role':'history'}
    if kind in ['work_order','schedule']:
        from backend.scheduling.service import order
        if kind=='schedule':
            assignment=conn.execute('SELECT * FROM schedule_assignments WHERE id=? AND asset_id=?',(rid,asset_id)).fetchone()
            if not assignment:raise HTTPException(404,'Schedule evidence not found')
            work=order(conn,user,assignment['work_order_id'])
            text=f'Internally approved maintenance window {assignment["start_utc"]} to {assignment["end_utc"]}; work order {work["id"]}; technicians '+', '.join(json.loads(assignment['technician_ids']))+'. No physical maintenance completion or equipment control is implied.'
            return {**base,'title':'Approved maintenance schedule','text':text,'timestamp_utc':assignment['start_utc'],'role':'history'}
        work=order(conn,user,rid)
        if work['asset_id']!=asset_id:raise HTTPException(404,'Work-order evidence not found')
        updates=' '.join(f'Human record by {e["actor_id"]} at {e["created_at"]}: {e["action"]}; {e["notes"]}' for e in work['execution']['events'][-5:])
        risk=work['content'].get('risk_snapshot')
        basis=f'Reviewed risk priority: {work["content"]["priority"]}, assessed at {risk["timestamp_utc"]}.' if risk else 'Calendar preventive task. No condition-based risk assessment is implied.'
        meter=work['content'].get('preventive',{})
        if meter.get('trigger')=='meter':
            basis=f'Meter preventive task at {meter["threshold"]} {meter["unit"]}; accepted cumulative total {meter["observed_total"]} from sample {meter["sample_id"]} at {meter["sample_timestamp"]}, including any human-reconciled counter offset. Actual threshold-crossing time is unknown. No condition-based risk assessment is implied.'
        return {**base,'title':'Work order '+work['id'],'text':f'{work["content"]["title"]}. Status: {work["status"]}. {basis} Scope: {work["content"]["task"]}. Physical work remains subject to site safety approval. '+updates+' Work completion records do not establish current equipment health or clear sensor alarms.',
                'timestamp_utc':work['execution']['updated_at'] or work['created_at'],'role':'history'}
    if kind=='risk':
        row=conn.execute('SELECT * FROM risk_assessments WHERE id=? AND asset_id=?',(rid,asset_id)).fetchone()
        if not row:raise HTTPException(404,'Evidence not found')
        result=json.loads(row['result'])
        for source in result['source_ids']:authorize_source(conn,user,source)
        components=result['components']
        return {**base,'title':'Maintenance risk assessment','text':f'{asset_id} assessment priority: {result["priority"]}. Risk proxy: {result["score"]}; degradation proxy {components["degradation_proxy"]}, weighted failure consequence {components["failure_consequence"]}. Safety criticality {result["criticality"]["safety"]}/5; production criticality {result["criticality"]["production"]}/5. This is not a failure probability. '+result['suggested_action'],
                'timestamp_utc':row['timestamp_utc'],'role':'risk'}
    if kind=='alarm':
        row=conn.execute('SELECT * FROM alarms WHERE id=? AND asset_id=?',(rid,asset_id)).fetchone()
        if not row:raise HTTPException(404,'Evidence not found')
        authorize_source(conn,user,row['source'])
        return {**base,'title':row['severity']+' alarm','text':f'Recorded alarm state: {row["state"]}. '+row['message'],'timestamp_utc':row['timestamp_utc'],'role':'alarm'}
    raise HTTPException(404,'Unsupported evidence type')
