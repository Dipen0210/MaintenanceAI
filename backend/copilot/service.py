"""Evidence-selection RAG: an actual LLM selects grounded material; backend validates every claim."""
from datetime import datetime
import hashlib,json,re,time
from fastapi import HTTPException
from pydantic import ValidationError
from backend.core.database import ROOT,transaction,identifier,iso,utcnow,audit
from backend.core.security import authorize_asset,allowed_assets
from backend.copilot.contracts import Selection,ChatResponse,Citation
from backend.copilot.providers import language_provider
from backend.copilot.tools import invoke
from backend.copilot.evidence import resolve
from backend.copilot.safety import unsafe_request,REFUSAL
from backend.copilot.configuration import register,RETRIEVAL

PROMPT_PATH=ROOT/'backend/copilot/prompts/evidence_selection_v3.txt'
CONFIG_VERSION='copilot-evidence-selection-v3'


def parse_selection(text):
    stripped=text.strip()
    if stripped.startswith('```'):
        stripped=stripped.split('\n',1)[1].rsplit('```',1)[0].strip()
    return Selection.model_validate_json(stripped)


def collect(user,asset_id,question,trace,mode='tools_rag'):
    sources=[];missing=[];calls={};question=question[:2000]
    names=['get_asset_details','get_latest_telemetry','get_sensor_trends','get_recent_alarms','get_model_predictions','get_fault_history','get_existing_work_orders','get_asset_risk','get_maintenance_schedule'] if mode=='tools_rag' else ['get_asset_details']
    for name in names:calls[name]=invoke(name,{'asset_id':asset_id,'query':question,'limit':15},user,trace)
    if mode!='no_rag':calls['search_approved_manuals']=invoke('search_approved_manuals',{'asset_id':asset_id,'query':question,'limit':RETRIEVAL['maximum_chunks']},user,trace)
    if 'report' in question.lower() and mode=='tools_rag':calls['draft_investigation_report']=invoke('draft_investigation_report',{'asset_id':asset_id},user,trace)
    def add(kind,rid,target_asset=None):
        with transaction() as conn:source=resolve(conn,user,target_asset or asset_id,kind+':'+rid)
        if source['id'] not in {s['id'] for s in sources}:sources.append(source)
    now=utcnow()
    for r in calls.get('get_latest_telemetry',[]):
        tag=r['sensor']['tag_id']
        if r['effective_quality']!='good' or not r['observation']:
            missing.append(tag+': '+r['effective_quality']);continue
        if r.get('last_ingestion') and r['last_ingestion']['status']=='rejected':
            missing.append(tag+': latest ingestion rejected');continue
        add('telemetry',r['observation']['id'])
    seen=set();bindings={(b['family'],b['dataset_id']) for b in calls['get_asset_details']['model_bindings']}
    for p in calls.get('get_model_predictions',[]):
        if p['family'] in seen:continue
        seen.add(p['family'])
        if p['status']!='SUCCEEDED' or p['historical'] or (p['family'],p['dataset_id']) not in bindings:
            missing.append(p['family']+': no compatible current successful prediction');continue
        if not p['signal_timestamp'] or (now-datetime.fromisoformat(p['signal_timestamp'])).total_seconds()>RETRIEVAL['current_prediction_seconds']:
            missing.append(p['family']+': prediction evidence is stale');continue
        if p.get('drift') and p['drift']['flagged']:
            missing.append(p['family']+': input distribution warning requires verification');continue
        add('prediction',p['id'])
    if mode=='tools_rag' and not any(s['role']=='prediction' for s in sources):missing.append('No compatible current predictive result; unsupported RUL must not be inferred.')
    trends={}
    for row in calls.get('get_sensor_trends',[]):
        if row['quality']=='good':trends.setdefault(row['tag_id'],[]).append(row)
    for rows in trends.values():
        rows.sort(key=lambda r:r['timestamp_utc'])
        if len(rows)>1:add('trend',rows[0]['id']+'~'+rows[-1]['id'])
    for name in ['get_fault_history','get_existing_work_orders','get_maintenance_schedule']:
        for record in calls.get(name,[])[:3]:
            add(record.get('evidence_kind','maintenance'),record['id'])
            age=(now-datetime.fromisoformat(record['timestamp_utc'])).days
            if age>365:missing.append(f'Maintenance record {record["external_id"]} is {age} days old; current relevance is unverified.')
    for alarm in calls.get('get_recent_alarms',[])[:3]:add('alarm',alarm['id'])
    risk=calls.get('get_asset_risk')
    if risk:
        if risk['stale']:missing.append('Maintenance risk assessment is stale; re-evaluation is required.')
        else:add('risk',risk['id'])
    if mode=='tools_rag' and any(term in question.lower() for term in ['which machines','which assets','review first','priority ranking']):
        with transaction() as conn:assets=allowed_assets(conn,user)
        for other in assets[:100]:
            if other['id']==asset_id:continue
            result=invoke('get_asset_risk',{'asset_id':other['id']},user,trace)
            if result and not result['stale']:add('risk',result['id'],other['id'])
    for doc in calls.get('search_approved_manuals',[]):add('document',doc['id'])
    if not any(s['kind']=='document' for s in sources):missing.append('No relevant approved and authorized equipment documentation was retrieved.')
    if mode!='tools_rag':missing.append('Structured equipment tools are disabled in this evaluation configuration.')
    question_lower=question.lower()
    preferred=['observation','prediction','risk','history','hypothesis','alternative','check','reference','safety','alarm']
    if any(term in question_lower for term in ['history','previous','similar problems','before']):preferred=['history','observation','prediction','risk','reference','hypothesis','alternative','check','safety','alarm']
    if any(term in question_lower for term in ['priority','rank','review first']):preferred=['risk','prediction','observation','history','reference','hypothesis','alternative','check','safety','alarm']
    if any(term in question_lower for term in ['manual','section','documentation']):preferred=['reference','hypothesis','alternative','check','safety','observation','prediction','history','risk','alarm']
    changed=any(term in question_lower for term in ['changed','trend','before the alarm'])
    sources.sort(key=lambda s:0 if changed and s['kind']=='trend' else preferred.index('history')+.5 if s['kind']=='trend' else preferred.index(s['role']))
    manual_lookup=any(word in question_lower for word in ['manual','section','guide','documentation']) and not any(word in question_lower for word in ['why','my motor','this motor','current cause'])
    if not manual_lookup:
        faults={s.get('fault_class') for s in sources if s['role']=='prediction'}-{'normal','Normal',None}
        if any(s.get('is_anomaly') for s in sources if s['role']=='prediction'):faults.add('ANOMALY')
        sources=[s for s in sources if s['role']!='hypothesis' or bool(set(s.get('required_fault_classes',[]))&faults)]
    trace.append({'retrieval':{'mode':mode,'evidence_ids':[s['id'] for s in sources],
                               'documents':[{'chunk_id':d['id'],'score':d['score'],'embedding_version':d['embedding_version']} for d in calls.get('search_approved_manuals',[])]}})
    return sources,missing


def validate_selection(selection,short_sources,user,asset_id):
    # The model cannot reclassify raw telemetry as an approved procedure or invent citations.
    groups={'evidence_ids':{'observation','prediction','history','risk','alarm','reference','safety'},
            'hypothesis_ids':{'hypothesis'},'alternative_ids':{'alternative'},'check_ids':{'check'}}
    resolved={}
    with transaction() as conn:
        for group,roles in groups.items():
            for sid in getattr(selection,group):
                source=short_sources.get(sid)
                if source is None or source['role'] not in roles:raise ValueError('Unsupported citation or evidence role')
                current=resolve(conn,user,source['asset_id'],source['id'])
                if current['text']!=source['text']:raise ValueError('Source changed during generation')
                if unsafe_request(current['text']):raise ValueError('Unsafe evidence instruction')
                resolved[sid]=current
    return resolved


def selection_schema(short):
    roles={'evidence_ids':{'observation','prediction','history','risk','alarm','reference','safety'},
           'hypothesis_ids':{'hypothesis'},'alternative_ids':{'alternative'},'check_ids':{'check'}}
    properties={}
    for field,allowed in roles.items():
        ids=[key for key,source in short.items() if source['role'] in allowed]
        # LM Format Enforcer 0.11.3 may allow a first item despite maxItems=0.
        # Omit unsupported roles altogether; Selection defaults them to empty lists.
        if not ids:continue
        properties[field]={'type':'array','items':{'type':'string','enum':ids},
                           'maxItems':min(len(ids),8 if field=='evidence_ids' else 3)}
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def chat(body,user,mode='tools_rag',provider=None):
    started=time.perf_counter();now=iso();trace=[];trace_id=identifier();message_id=identifier();previous=[]
    config=register(mode)
    with transaction() as conn:
        asset=authorize_asset(conn,user,body.asset_id)
        if body.session_id:
            session=conn.execute('SELECT * FROM copilot_sessions WHERE id=? AND user_id=? AND asset_id=?',(body.session_id,user['id'],body.asset_id)).fetchone()
            if not session:raise HTTPException(404,'Copilot session not found or not authorized')
            sid=body.session_id
            prior=list(conn.execute('SELECT role,content FROM copilot_messages WHERE session_id=? ORDER BY rowid DESC LIMIT 4',(sid,)))
            for row in reversed(prior):
                if row['role']=='user':previous.append({'role':'user','text':row['content'][:600]})
                else:
                    old=json.loads(row['content'])
                    try:
                        for source in old['citations']:resolve(conn,user,source['asset_id'],source['id'])
                    except HTTPException:continue
                    previous.append({'role':'assistant','text':old['answer'],'hypotheses':[c['text'][:300] for c in old['hypotheses'][:2]]})
        else:
            sid=identifier();conn.execute('INSERT INTO copilot_sessions VALUES(?,?,?,?)',(sid,user['id'],body.asset_id,now))
        text='[Unsafe/control/credential request redacted]' if unsafe_request(body.message) else body.message
        conn.execute('INSERT INTO copilot_messages VALUES(?,?,?,?,?,?)',(identifier(),sid,'user',text,now,trace_id))
    response=dict(session_id=sid,message_id=message_id,trace_id=trace_id,asset_id=body.asset_id,status='abstained',answer='',
                  observations=[],predictions=[],hypotheses=[],alternatives=[],approved_checks=[],maintenance_history=[],manual_references=[],missing_information=[],citations=[],
                  human_review_required=True,uncertainty='Hypotheses require qualified human verification. No calibrated diagnostic certainty is available.',model_id=None,as_of=now)
    completion=None;validation='not run';sources=[]
    if unsafe_request(body.message):
        response.update(status='refused',answer=REFUSAL);validation='deterministic unsafe-request refusal'
    elif re.search(r'\b(rul(?:_cycles)?|remaining useful life)\b',body.message.lower()) and asset['equipment_type']!='turbofan_reference':
        response.update(status='abstained',answer='UNSUPPORTED_ASSET: the available C-MAPSS RUL models support registered turbofan reference datasets, not this equipment. No RUL estimate is provided.',missing_information=['A validated RUL model for this equipment and operating domain is unavailable.'])
        validation='deterministic model-applicability abstention'
    else:
        try:
            last_question=next((m['text'] for m in reversed(previous) if m['role']=='user'),'')
            retrieval_question=body.message+(' Context: '+last_question if last_question else '')
            sources,missing=collect(user,body.asset_id,retrieval_question,trace,mode)
            response['missing_information']=missing
            short={f'E{i+1}':s for i,s in enumerate(sources)}
            prompt=PROMPT_PATH.read_text()
            payload={'asset':{'id':body.asset_id,'name':asset['name'],'type':asset['equipment_type']},'question':body.message,
                     'conversation_context':previous,'missing_information':missing,'evidence':[{'id':k,'role':v['role'],'text':v['text']} for k,v in short.items()]}
            completion=(provider or language_provider()).complete([{'role':'system','content':prompt},{'role':'user','content':json.dumps(payload)}],256,schema=selection_schema(short))
            response['model_id']=completion.model_id
            selection=parse_selection(completion.text)
            selected=validate_selection(selection,short,user,body.asset_id)
            validation='passed citation, role, applicability and current authorization checks'
            has_observations=any(s['role']=='observation' for s in sources)
            has_documents=any(s['kind']=='document' for s in sources)
            has_prediction=any(s['role']=='prediction' for s in sources)
            can_diagnose=has_observations and has_documents and has_prediction and not any('stale' in m or 'rejected' in m or ': missing' in m or ': bad' in m for m in missing)
            used=set()
            def claim(source):
                used.add(source['id']);return {'text':source['text'],'citations':[source['id']]}
            for key in selection.evidence_ids:
                source=selected[key];role=source['role']
                if role in ['observation','risk','alarm']:response['observations'].append(claim(source))
                elif role=='prediction':response['predictions'].append(claim(source))
                elif role=='history':response['maintenance_history'].append(claim(source))
                else:response['manual_references'].append(claim(source))
            if can_diagnose:
                response['hypotheses']=[claim(selected[k]) for k in selection.hypothesis_ids]
                response['alternatives']=[claim(selected[k]) for k in selection.alternative_ids]
                response['approved_checks']=[claim(selected[k]) for k in selection.check_ids]
            # Data-only lookup/history can be answered without pretending to diagnose a fault.
            diagnostic=any(word in body.message.lower() for word in ['why','fault','cause','diagnos','abnormal','flagged','investigat'])
            manual_lookup=any(word in body.message.lower() for word in ['manual','section','guide','documentation']) and not any(word in body.message.lower() for word in ['why','my motor','this motor','current cause'])
            if manual_lookup:
                diagnostic=False
                for key in selection.hypothesis_ids+selection.alternative_ids+selection.check_ids:
                    source=selected[key]
                    if source['id'] not in used:response['manual_references'].append(claim(source))
            content=any(response[key] for key in ['observations','predictions','hypotheses','maintenance_history','manual_references'])
            if diagnostic and (not can_diagnose or not response['hypotheses']):
                response.update(status='abstained',answer='The available authorized evidence is insufficient to explain the current condition. Review the missing information below; no fault is confirmed.')
            elif not content:
                response.update(status='abstained',answer='No supported answer could be established from the authorized evidence.')
            else:
                response.update(status='answered',answer=f'Evidence review for {asset["name"]}. Observations and model outputs are listed separately from unconfirmed hypotheses. A qualified person must review any maintenance proposal.')
            response['citations']=[Citation.model_validate(s).model_dump() for s in sources if s['id'] in used]
        except HTTPException:
            # Re-checks can revoke access while the model is running. Do not return earlier evidence.
            response.update(status='unavailable',answer='Evidence access changed or a required data service is unavailable. Refresh and try again.',
                            observations=[],predictions=[],hypotheses=[],alternatives=[],approved_checks=[],maintenance_history=[],manual_references=[],citations=[])
            validation='authorization or service validation failed'
        except Exception as exc:
            response.update(status='unavailable',answer='The language-model response or evidence could not be validated. No unsupported answer is substituted.',
                            observations=[],predictions=[],hypotheses=[],alternatives=[],approved_checks=[],maintenance_history=[],manual_references=[],citations=[])
            validation='failed: '+type(exc).__name__
    result=ChatResponse.model_validate(response).model_dump(mode='json')
    details={'configuration_id':config['id'],'request_sha256':hashlib.sha256(body.message.encode()).hexdigest(),'prompt_sha256':hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(),
             'retrieval_mode':mode,'conversation_messages':len(previous),'operations':trace,'validation':validation,'total_latency_ms':round((time.perf_counter()-started)*1000,3),
             'model_id':completion.model_id if completion else None,'input_tokens':completion.input_tokens if completion else 0,
             'output_tokens':completion.output_tokens if completion else 0,'inference_latency_ms':completion.latency_ms if completion else None,
             'estimated_cost_usd':completion.estimated_cost_usd if completion else None,'cost_basis':completion.cost_basis if completion else 'No inference performed',
             'selected_citation_ids':[c['id'] for c in result['citations']]}
    with transaction() as conn:
        authorize_asset(conn,user,body.asset_id)
        conn.execute('INSERT INTO copilot_traces VALUES(?,?,?,?,?,?,?,?)',(trace_id,sid,user['id'],body.asset_id,now,CONFIG_VERSION,result['status'],json.dumps(details)))
        conn.execute('INSERT INTO copilot_messages VALUES(?,?,?,?,?,?)',(message_id,sid,'assistant',json.dumps(result),iso(),trace_id))
        audit(conn,user,'copilot_chat',trace_id,result['status'])
    return result
