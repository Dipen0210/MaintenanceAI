"""Scoped trace inspection, measured evaluations and auditable human-reviewed release decisions."""
import json
from typing import Literal
from fastapi import APIRouter,Depends,HTTPException,Query
from pydantic import BaseModel,ConfigDict,Field
from backend.core.database import transaction,identifier,iso,audit
from backend.core.security import principal,require_roles,authorize_asset
from backend.copilot.evidence import resolve
from backend.copilot.evaluation import release_reasons

router=APIRouter(prefix='/api',tags=['copilot observability'])
REVIEW_ROLES=('administrator','reliability_engineer','maintenance_manager')


class TraceSummary(BaseModel):
    id:str
    asset_id:str
    timestamp_utc:str
    status:str
    config_version:str
    configuration_id:str|None
    model_id:str|None
    input_tokens:int
    output_tokens:int
    total_latency_ms:float
    estimated_cost_usd:float|None


class TraceDetail(TraceSummary):
    details:dict
    feedback:list[dict]


class EvaluationSummary(BaseModel):
    id:str
    config_id:str
    dataset_version:str
    mode:str
    finished_at:str
    status:str
    metrics:dict


class EvaluationDetail(EvaluationSummary):
    report:dict
    reviews:list[dict]


class ReviewRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    qualification:str=Field(min_length=20,max_length=1000)
    attest_qualified:Literal[True]
    decision:Literal['accepted','rejected']
    notes:str=Field(min_length=30,max_length=5000)
    reviewed_case_ids:list[str]=Field(min_length=1,max_length=100)


class ReleaseRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    run_id:str
    baseline_id:str|None=None


class DecisionResponse(BaseModel):
    id:str
    status:str
    reasons:list[str]=Field(default_factory=list)


def trace_access(conn,user,row):
    if not row:raise HTTPException(404,'Trace not found or not authorized')
    authorize_asset(conn,user,row['asset_id'])
    if row['user_id']!=user['id'] and user['role'] not in REVIEW_ROLES:raise HTTPException(404,'Trace not found or not authorized')
    details=json.loads(row['details'])
    # Reject rather than exposing private document/source identifiers through a trace.
    for operation in details['operations']:
        if 'retrieval' in operation:
            for evidence_id in operation['retrieval']['evidence_ids']:
                asset_id=row['asset_id']
                # Portfolio questions legitimately retrieve risk evidence for other
                # authorized assets. Resolve each record against its actual asset.
                if evidence_id.startswith('risk:'):
                    risk=conn.execute('SELECT asset_id FROM risk_assessments WHERE id=?',(evidence_id[5:],)).fetchone()
                    if not risk:raise HTTPException(404,'Trace evidence is no longer available')
                    asset_id=risk['asset_id']
                resolve(conn,user,asset_id,evidence_id)
            for doc in operation['retrieval']['documents']:resolve(conn,user,row['asset_id'],'document:'+doc['chunk_id'])
    return details


def trace_summary(row,details):
    return {k:row[k] for k in ['id','asset_id','timestamp_utc','status','config_version']}|{k:details.get(k) for k in ['configuration_id','model_id','input_tokens','output_tokens','total_latency_ms','estimated_cost_usd']}


@router.get('/copilot/traces',response_model=list[TraceSummary])
def traces(asset_id:str,limit:int=Query(50,ge=1,le=200),user=Depends(principal)):
    with transaction() as conn:
        authorize_asset(conn,user,asset_id);result=[]
        for row in conn.execute('SELECT * FROM copilot_traces WHERE asset_id=? ORDER BY timestamp_utc DESC LIMIT 500',(asset_id,)):
            try:details=trace_access(conn,user,row)
            except HTTPException:continue
            result.append(trace_summary(row,details))
            if len(result)>=limit:break
        return result


@router.get('/copilot/traces/{trace_id}',response_model=TraceDetail)
def trace(trace_id:str,user=Depends(principal)):
    with transaction() as conn:
        row=conn.execute('SELECT * FROM copilot_traces WHERE id=?',(trace_id,)).fetchone();details=trace_access(conn,user,row)
        feedback=[dict(r) for r in conn.execute('''SELECT f.rating,f.comment,f.created_at FROM copilot_feedback f JOIN copilot_messages m ON f.message_id=m.id WHERE m.trace_id=?''',(trace_id,))]
        audit(conn,user,'read_copilot_trace',trace_id,'authorized')
        return trace_summary(row,details)|{'details':details,'feedback':feedback}


def evaluation_access(conn,user,run_id):
    row=conn.execute('SELECT * FROM evaluation_runs WHERE id=? AND plant_id=?',(run_id,user['plant_id'])).fetchone()
    if not row:raise HTTPException(404,'Evaluation not found or not authorized')
    report=json.loads(row['report'])
    for case in report['cases']:authorize_asset(conn,user,case['response']['asset_id'])
    return row,report


def evaluation_summary(row,report):
    return {k:row[k] for k in ['id','config_id','dataset_version','mode','finished_at','status']}|{'metrics':report['metrics']}


@router.get('/evaluations',response_model=list[EvaluationSummary])
def evaluations(user=Depends(require_roles(*REVIEW_ROLES))):
    with transaction() as conn:
        results=[]
        for row in conn.execute('SELECT * FROM evaluation_runs WHERE plant_id=? ORDER BY finished_at DESC LIMIT 100',(user['plant_id'],)):
            try:_,report=evaluation_access(conn,user,row['id'])
            except HTTPException:continue
            results.append(evaluation_summary(row,report))
        return results


@router.get('/evaluations/{run_id}',response_model=EvaluationDetail)
def evaluation(run_id:str,user=Depends(require_roles(*REVIEW_ROLES))):
    with transaction() as conn:
        row,report=evaluation_access(conn,user,run_id)
        reviews=[dict(r) for r in conn.execute('SELECT * FROM evaluation_reviews WHERE run_id=? ORDER BY created_at DESC',(run_id,))]
        return evaluation_summary(row,report)|{'report':report,'reviews':reviews}


@router.post('/evaluations/{run_id}/review',response_model=DecisionResponse)
def review(run_id:str,body:ReviewRequest,user=Depends(require_roles(*REVIEW_ROLES))):
    with transaction() as conn:
        _,report=evaluation_access(conn,user,run_id)
        if set(body.reviewed_case_ids)!={case['id'] for case in report['cases']} or len(body.reviewed_case_ids)!=len(report['cases']):
            raise HTTPException(422,'Review must explicitly cover every measured scenario once')
        rid=identifier();notes=json.dumps({'notes':body.notes,'reviewed_case_ids':body.reviewed_case_ids,'qualification_attested_by_user':True})
        conn.execute('INSERT INTO evaluation_reviews VALUES(?,?,?,?,?,?,?)',(rid,run_id,user['id'],body.qualification,body.decision,notes,iso()))
        audit(conn,user,'review_copilot_evaluation',run_id,body.decision)
        return {'id':rid,'status':body.decision}


@router.post('/copilot/releases',response_model=DecisionResponse)
def release(body:ReleaseRequest,user=Depends(require_roles('administrator','reliability_engineer'))):
    with transaction() as conn:
        _,report=evaluation_access(conn,user,body.run_id);baseline=None
        if body.baseline_id:
            _,baseline=evaluation_access(conn,user,body.baseline_id)
            if not conn.execute("SELECT 1 FROM copilot_releases WHERE run_id=? AND plant_id=? AND status='accepted'",(body.baseline_id,user['plant_id'])).fetchone():
                raise HTTPException(422,'Comparison baseline must have an accepted release decision')
        else:
            previous=conn.execute("SELECT run_id FROM copilot_releases WHERE plant_id=? AND status='accepted' ORDER BY created_at DESC LIMIT 1",(user['plant_id'],)).fetchone()
            if previous:_,baseline=evaluation_access(conn,user,previous['run_id'])
        review=conn.execute('SELECT * FROM evaluation_reviews WHERE run_id=? ORDER BY created_at DESC LIMIT 1',(body.run_id,)).fetchone()
        reasons=release_reasons(report,baseline,expert_review=dict(review) if review else None)
        rid=identifier();status='blocked' if reasons else 'accepted'
        conn.execute('INSERT INTO copilot_releases VALUES(?,?,?,?,?,?,?,?)',(rid,user['plant_id'],body.run_id,baseline['id'] if baseline else None,user['id'],status,json.dumps(reasons),iso()))
        audit(conn,user,'copilot_release',rid,status)
        return {'id':rid,'status':status,'reasons':reasons}
