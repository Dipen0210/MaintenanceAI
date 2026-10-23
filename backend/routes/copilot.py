"""Authorized document administration and evidence-grounded investigation workspace."""
import json
from pathlib import Path
from fastapi import APIRouter,Depends,File,Form,HTTPException,Query,UploadFile,Response
from pydantic import ValidationError
from backend.core.database import transaction,identifier,iso,audit
from backend.core.security import principal,require_roles,authorize_asset,allowed_assets
from backend.copilot.contracts import (DocumentMetadata,DocumentApproval,ChatRequest,ChatResponse,Citation,
    DocumentImportResponse,ApprovalResponse,DocumentResponse,CMMSResponse,ToolRequest,ToolResponse,SessionResponse,
    FeedbackRequest,InvestigationRequest,IDResponse,RevokeRequest,SessionSummary,UserSummary)
from backend.copilot.documents import import_document,approve,authorize_document,authorize_document_review
from backend.copilot.records import import_csv
from backend.copilot.tools import invoke
from backend.copilot.service import chat
from backend.copilot.evidence import resolve

router=APIRouter(prefix='/api',tags=['engineer copilot'])


def document_view(row,chunks=None):
    return {k:row[k] for k in ['id','title','revision','status','created_at','approved_at','approved_by']}|{
        'equipment_types':json.loads(row['equipment_types']),'asset_ids':json.loads(row['asset_ids']),
        'metadata':json.loads(row['metadata']),'chunks':chunks or []}


@router.post('/documents/import',response_model=DocumentImportResponse)
def upload_document(metadata:str=Form(...),file:UploadFile=File(...),user=Depends(require_roles('administrator'))):
    try:body=DocumentMetadata.model_validate_json(metadata)
    except ValidationError as exc:raise HTTPException(422,'Invalid document metadata') from exc
    return import_document(file.file.read(20*1024*1024+1),file.filename or '',body,user)


@router.get('/documents',response_model=list[DocumentResponse])
def documents(user=Depends(principal)):
    with transaction() as conn:
        assets=allowed_assets(conn,user);ids={a['id'] for a in assets};types={a['equipment_type'] for a in assets}
        rows=conn.execute('''SELECT d.* FROM documents d JOIN document_grants g ON d.id=g.document_id
                            WHERE g.user_id=? AND d.plant_id=? ORDER BY d.created_at DESC''',(user['id'],user['plant_id']))
        return [document_view(r) for r in rows if set(json.loads(r['equipment_types']))&types and (not json.loads(r['asset_ids']) or set(json.loads(r['asset_ids']))&ids)]


@router.get('/documents/{document_id}',response_model=DocumentResponse)
def document(document_id:str,asset_id:str,user=Depends(principal)):
    with transaction() as conn:
        doc=authorize_document(conn,user,document_id,asset_id,require_approved=False)
        chunks=[{k:r[k] for k in ['id','page','section','content','kind','reviewed','embedding_version']} for r in conn.execute('SELECT * FROM document_chunks WHERE document_id=? ORDER BY rowid',(document_id,))]
        return document_view(doc,chunks)


@router.post('/documents/{document_id}/approve',response_model=ApprovalResponse)
def approve_document(document_id:str,body:DocumentApproval,user=Depends(require_roles('administrator','maintenance_manager','reliability_engineer'))):
    return approve(document_id,body,user)


@router.post('/documents/{document_id}/revoke',response_model=IDResponse)
def revoke_document(document_id:str,body:RevokeRequest,user=Depends(require_roles('administrator','maintenance_manager','reliability_engineer'))):
    with transaction() as conn:
        doc=authorize_document_review(conn,user,document_id)
        if doc['status']!='approved':raise HTTPException(409,'Only approved documents can be revoked')
        conn.execute("UPDATE documents SET status='revoked' WHERE id=?",(document_id,))
        audit(conn,user,'revoke_document',document_id,body.reason)
    return {'id':document_id,'status':'revoked'}


@router.post('/maintenance/import/{source_id}',response_model=CMMSResponse)
def cmms(source_id:str,file:UploadFile=File(...),user=Depends(require_roles('administrator'))):
    return import_csv(file.file.read(5*1024*1024+1),source_id,user)


@router.post('/copilot/tools',response_model=ToolResponse)
def tool(body:ToolRequest,user=Depends(principal)):
    return {'name':body.name,'result':invoke(body.name,body.arguments.model_dump(),user)}


@router.post('/copilot/chat',response_model=ChatResponse)
def chat_endpoint(body:ChatRequest,user=Depends(principal)):
    return chat(body,user)


@router.get('/evidence/{evidence_id}',response_model=Citation)
def evidence(evidence_id:str,asset_id:str,user=Depends(principal)):
    with transaction() as conn:
        result=resolve(conn,user,asset_id,evidence_id)
        audit(conn,user,'read_evidence',evidence_id,'allowed')
        return result


def visible_answer(conn,user,result):
    try:
        for citation in result['citations']:resolve(conn,user,citation['asset_id'],citation['id'])
    except HTTPException:
        result.update(status='unavailable',answer='Access to supporting evidence changed. This prior response is no longer available.',
                      observations=[],predictions=[],hypotheses=[],alternatives=[],approved_checks=[],maintenance_history=[],manual_references=[],citations=[],missing_information=[])
    return result


@router.get('/copilot/sessions/{session_id}',response_model=SessionResponse)
def session(session_id:str,user=Depends(principal)):
    with transaction() as conn:
        row=conn.execute('SELECT * FROM copilot_sessions WHERE id=? AND user_id=?',(session_id,user['id'])).fetchone()
        if not row:raise HTTPException(404,'Session not found or not authorized')
        authorize_asset(conn,user,row['asset_id']);messages=[]
        for m in conn.execute('SELECT * FROM copilot_messages WHERE session_id=? ORDER BY rowid',(session_id,)):
            content=m['content'] if m['role']=='user' else visible_answer(conn,user,json.loads(m['content']))
            messages.append({'id':m['id'],'role':m['role'],'content':content,'timestamp_utc':m['timestamp_utc'],'trace_id':m['trace_id']})
        return {'id':session_id,'asset_id':row['asset_id'],'messages':messages}


@router.get('/copilot/sessions',response_model=list[SessionSummary])
def sessions(asset_id:str,user=Depends(principal)):
    with transaction() as conn:
        authorize_asset(conn,user,asset_id)
        return [dict(r) for r in conn.execute('SELECT id,asset_id,created_at FROM copilot_sessions WHERE user_id=? AND asset_id=? ORDER BY created_at DESC LIMIT 50',(user['id'],asset_id))]


@router.post('/copilot/feedback',response_model=IDResponse)
def feedback(body:FeedbackRequest,user=Depends(principal)):
    with transaction() as conn:
        row=conn.execute('''SELECT s.asset_id FROM copilot_messages m JOIN copilot_sessions s ON m.session_id=s.id
            WHERE m.id=? AND s.user_id=? AND m.role='assistant' ''',(body.message_id,user['id'])).fetchone()
        if not row:raise HTTPException(404,'Response not found or not authorized')
        authorize_asset(conn,user,row['asset_id']);fid=identifier()
        conn.execute('INSERT INTO copilot_feedback VALUES(?,?,?,?,?,?)',(fid,body.message_id,user['id'],body.rating,body.comment,iso()))
        audit(conn,user,'copilot_feedback',body.message_id,str(body.rating))
        return {'id':fid,'status':'recorded'}


def report_markdown(result,notes):
    lines=['# Maintenance investigation draft','',f'Asset: {result["asset_id"]}',f'Evidence as of: {result["as_of"]}',
           'Status: requires qualified human review. No maintenance execution is authorized.','',result['answer']]
    for key,title in [('observations','Observed evidence'),('predictions','Model predictions'),('hypotheses','Unconfirmed hypotheses'),('alternatives','Alternative explanations'),('approved_checks','Approved reference checks'),('maintenance_history','Maintenance history'),('manual_references','Manual references')]:
        lines+=['','## '+title,'']
        lines += ['- '+item['text']+' ['+', '.join(item['citations'])+']' for item in result[key]] or ['No supported entries.']
    lines+=['','## Missing information','']+['- '+m for m in result['missing_information']]
    lines+=['','## Engineer notes','',notes,'','## Sources','']
    lines += [f'- {c["id"]}: {c["title"]}; revision {c.get("revision") or "n/a"}; page {c.get("page") or "n/a"}; time {c.get("timestamp_utc") or "n/a"}.' for c in result['citations']]
    return '\n'.join(lines)+'\n'


@router.post('/investigations',response_model=IDResponse)
def investigation(body:InvestigationRequest,user=Depends(require_roles('administrator','engineer','reliability_engineer','maintenance_manager'))):
    with transaction() as conn:
        authorize_asset(conn,user,body.asset_id)
        row=conn.execute('''SELECT m.content FROM copilot_messages m JOIN copilot_sessions s ON m.session_id=s.id
            WHERE s.id=? AND s.user_id=? AND s.asset_id=? AND m.role='assistant' ORDER BY m.rowid DESC LIMIT 1''',(body.session_id,user['id'],body.asset_id)).fetchone()
        if not row:raise HTTPException(409,'A reviewed copilot response is required before drafting a report')
        result=visible_answer(conn,user,json.loads(row['content']))
        if result['status'] in ['unavailable','refused']:raise HTTPException(409,'No validated response is available for a report')
        rid=identifier();references=[{'id':c['id'],'asset_id':c['asset_id']} for c in result['citations']]
        conn.execute('INSERT INTO investigations VALUES(?,?,?,?,?,?,?,?,?)',(rid,body.asset_id,user['id'],body.session_id,body.notes,
                     report_markdown(result,body.notes),json.dumps(references),iso(),'draft'))
        audit(conn,user,'draft_investigation',rid,'human review required')
        return {'id':rid,'status':'draft'}


@router.get('/investigations/{investigation_id}/report')
def report(investigation_id:str,user=Depends(principal)):
    with transaction() as conn:
        row=conn.execute('SELECT * FROM investigations WHERE id=? AND author_id=?',(investigation_id,user['id'])).fetchone()
        if not row:raise HTTPException(404,'Investigation not found or not authorized')
        authorize_asset(conn,user,row['asset_id'])
        for reference in json.loads(row['evidence_ids']):resolve(conn,user,reference['asset_id'],reference['id'])
        audit(conn,user,'export_investigation',investigation_id,'allowed')
        return Response(row['report'],media_type='text/markdown',headers={'Content-Disposition':f'attachment; filename="investigation-{investigation_id}.md"'})


@router.get('/admin/users',response_model=list[UserSummary])
def users(user=Depends(require_roles('administrator'))):
    with transaction() as conn:
        return [dict(r) for r in conn.execute('SELECT id,username,role,enabled FROM users WHERE plant_id=? ORDER BY username',(user['plant_id'],))]
