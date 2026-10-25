"""Ordered, attributed field observations. Findings open issues, never maintenance approvals."""
import json
from datetime import datetime,timezone
from typing import Literal
from fastapi import HTTPException
from pydantic import Field,FiniteFloat,model_validator,field_validator
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.core.security import authorize_asset
from backend.core.workspaces import asset_workspace
from backend.scheduling.contracts import Strict
from backend.scheduling.service import role,procedures
from backend.copilot.safety import quarantine_reason


class Step(Strict):
    asset_id:str
    title:str=Field(min_length=5,max_length=160)
    instructions:str=Field(min_length=15,max_length=1500)
    procedure_ids:list[str]=Field(min_length=1,max_length=10)
    kind:Literal['numeric','check']
    unit:str=Field(default='',max_length=30)
    lower:FiniteFloat|None=None
    upper:FiniteFloat|None=None

    @model_validator(mode='after')
    def valid(self):
        if len(self.title.strip())<5 or len(self.instructions.strip())<15:raise ValueError('Describe the observation and approved scope')
        if self.kind=='numeric' and (not self.unit.strip() or self.lower is None or self.upper is None or self.lower>self.upper):raise ValueError('Numeric checks need a unit and ordered lower/upper limits')
        if self.kind=='check' and (self.unit or self.lower is not None or self.upper is not None):raise ValueError('Pass/fail checks do not use numeric limits')
        return self


class TemplateInput(Strict):
    title:str=Field(min_length=5,max_length=160)
    steps:list[Step]=Field(min_length=1,max_length=20)
    enabled:bool=True
    reason:str=Field(min_length=15,max_length=500)


class TemplateRevision(TemplateInput):
    expected_version:int=Field(ge=1)


class Assignment(Strict):
    expected_template_version:int=Field(ge=1)
    assignee_id:str
    due_at:datetime
    request_key:str=Field(min_length=16,max_length=100)

    @field_validator('due_at')
    @classmethod
    def aware(cls,v):
        if v.tzinfo is None:raise ValueError('A timezone-aware due date is required')
        return v.astimezone(timezone.utc)


class Update(Strict):
    expected_version:int=Field(ge=1)
    action:Literal['start','record','complete','cancel']
    step_index:int|None=Field(default=None,ge=0,le=19)
    outcome:Literal['measured','pass','fail','skipped']|None=None
    value:FiniteFloat|None=None
    notes:str=Field(min_length=15,max_length=1500)
    site_clearance_confirmed:bool=False


def scope(conn,user,content):
    origins=set()
    for step in content['steps']:
        authorize_asset(conn,user,step['asset_id']);procedures(conn,user,step['asset_id'],step['procedure_ids'])
        origins.add(asset_workspace(conn,step['asset_id']))
    if len(origins)>1:raise HTTPException(422,'A round cannot mix demo and plant equipment')


def template(conn,user,tid):
    row=conn.execute('SELECT * FROM inspection_templates WHERE id=? AND plant_id=?',(tid,user['plant_id'])).fetchone()
    if not row:raise HTTPException(404,'Inspection template unavailable')
    p=dict(row);p['content']=json.loads(p['content']);p['enabled']=bool(p['enabled']);scope(conn,user,p['content'])
    # Old versions may contain equipment to which access has since been revoked.
    p['history']=[]
    for r in conn.execute('SELECT * FROM inspection_template_versions WHERE template_id=? ORDER BY version DESC',(tid,)):
        try:scope(conn,user,json.loads(r['content']))
        except HTTPException:continue
        p['history'].append({k:r[k] for k in ['version','actor_id','created_at','reason']})
    return p


def save(body,user,tid=None):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['maintenance_manager'])
        content=body.model_dump(exclude={'expected_version','reason','enabled'});scope(conn,user,content)
        if len(body.title.strip())<5 or len(body.reason.strip())<15:raise HTTPException(422,'Use a meaningful title and revision reason')
        if quarantine_reason(json.dumps(content)):raise HTTPException(422,'Unsafe instructions cannot become an inspection round')
        now=iso();version=1
        if tid:
            prior=template(conn,user,tid)
            if prior['version']!=body.expected_version:raise HTTPException(409,'Template changed; refresh before revising')
            version=prior['version']+1;conn.execute('UPDATE inspection_templates SET version=?,enabled=?,content=?,updated_at=? WHERE id=?',(version,int(body.enabled),json.dumps(content),now,tid))
        else:
            tid=identifier();conn.execute('INSERT INTO inspection_templates VALUES(?,?,?,?,?,?,?)',(tid,user['plant_id'],1,int(body.enabled),json.dumps(content),now,now))
        conn.execute('INSERT INTO inspection_template_versions VALUES(?,?,?,?,?,?)',(tid,version,user['id'],now,body.reason,json.dumps(content)));audit(conn,user,'configure_inspection_round',tid,body.reason)
        return template(conn,user,tid)


def run(conn,user,rid):
    row=conn.execute('SELECT * FROM inspection_runs WHERE id=? AND plant_id=?',(rid,user['plant_id'])).fetchone()
    if not row:raise HTTPException(404,'Inspection run unavailable')
    p=dict(row);p.pop('request_key');p['content']=json.loads(p['content']);scope(conn,user,p['content'])
    for step in p['content']['steps']:step['procedures']=procedures(conn,user,step['asset_id'],step['procedure_ids'])
    p['results']=[dict(step_index=r['step_index'],**json.loads(r['result']),issue_id=r['issue_id']) for r in conn.execute('SELECT * FROM inspection_results WHERE run_id=? ORDER BY step_index',(rid,))]
    p['events']=[{**dict(r),'detail':json.loads(r['detail'])} for r in conn.execute('SELECT version,actor_id,created_at,action,detail FROM inspection_events WHERE run_id=? ORDER BY version',(rid,))]
    p['overdue']=p['status'] not in ['completed','completed_with_skips','cancelled'] and datetime.fromisoformat(p['due_at'])<utcnow()
    return p


def assign(tid,body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');role(conn,user,['maintenance_manager']);p=template(conn,user,tid)
        existing=conn.execute('SELECT * FROM inspection_runs WHERE created_by=? AND request_key=?',(user['id'],body.request_key)).fetchone()
        if existing:
            if existing['template_id']!=tid or existing['template_version']!=body.expected_template_version or existing['assignee_id']!=body.assignee_id or existing['due_at']!=iso(body.due_at):raise HTTPException(409,'Request key was already used for a different assignment')
            return run(conn,user,existing['id'])
        if not p['enabled'] or p['version']!=body.expected_template_version:raise HTTPException(409,'Template is disabled or changed; refresh before assigning')
        if body.due_at<=utcnow():raise HTTPException(422,'New runs need a future due date')
        person=conn.execute('SELECT id,role,plant_id FROM users WHERE id=? AND enabled=1 AND plant_id=?',(body.assignee_id,user['plant_id'])).fetchone()
        if not person or person['role'] not in ['engineer','reliability_engineer']:raise HTTPException(422,'Assign an enabled field engineer')
        scope(conn,dict(person),p['content']);rid=identifier();now=iso()
        conn.execute('INSERT INTO inspection_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(rid,user['plant_id'],tid,p['version'],json.dumps(p['content']),body.assignee_id,user['id'],now,iso(body.due_at),'assigned',1,now,body.request_key))
        conn.execute('INSERT INTO inspection_events VALUES(?,?,?,?,?,?,?)',(identifier(),rid,1,user['id'],now,'assigned',json.dumps({'assignee_id':body.assignee_id})))
        audit(conn,user,'assign_inspection_round',rid,'Inspection assignment only; no corrective work authorized')
        return run(conn,user,rid)


def update(rid,body,user):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');p=run(conn,user,rid)
        if p['version']!=body.expected_version:raise HTTPException(409,'Round changed; refresh before saving')
        if len(body.notes.strip())<15:raise HTTPException(422,'Record the observed evidence or a meaningful skip reason')
        if p['status'] in ['completed','completed_with_skips','cancelled']:raise HTTPException(409,'Closed round evidence is immutable; assign a new run for follow-up')
        if body.action=='cancel':role(conn,user,['maintenance_manager'])
        else:
            role(conn,user,['engineer','reliability_engineer'])
            if p['assignee_id']!=user['id']:raise HTTPException(403,'Only the assigned engineer may record this round')
        if body.action!='record' and any(v is not None for v in [body.step_index,body.outcome,body.value]):raise HTTPException(422,'Measurements belong to a record action')
        status=p['status'];detail={'notes':body.notes}
        if body.action=='start':
            if status!='assigned' or not body.site_clearance_confirmed:raise HTTPException(409,'Start requires assigned status and confirmed site clearance for this inspection scope')
            status='in_progress';detail['site_clearance_confirmed']=True
        elif body.action=='record':
            if status!='in_progress':raise HTTPException(409,'Start the round before recording evidence')
            index=body.step_index
            if index is None or index>=len(p['content']['steps']):raise HTTPException(422,'Choose a valid inspection step')
            if any(not any(r['step_index']==i for r in p['results']) for i in range(index)):raise HTTPException(409,'Record or explicitly skip preceding checks first')
            step=p['content']['steps'][index];outcome=body.outcome;abnormal=False
            if outcome=='skipped':
                if body.value is not None:raise HTTPException(422,'A skipped check cannot claim a measurement')
            elif step['kind']=='numeric':
                if outcome!='measured' or body.value is None:raise HTTPException(422,'Record a finite numeric measurement')
                abnormal=body.value<step['lower'] or body.value>step['upper']
            elif outcome not in ['pass','fail'] or body.value is not None:raise HTTPException(422,'Record pass or fail without a numeric value')
            else:abnormal=outcome=='fail'
            previous=next((r for r in p['results'] if r['step_index']==index),None);iid=previous['issue_id'] if previous else None
            result=dict(outcome=outcome,value=body.value,notes=body.notes,abnormal=abnormal,actor_id=user['id'],recorded_at=iso())
            if abnormal and not iid:
                iid=identifier();now=iso();description=f'Inspection run {rid}, step {index+1}: {step["title"]}. Recorded {body.value} {step["unit"]}.' if step['kind']=='numeric' else f'Inspection run {rid}, step {index+1}: {step["title"]}. Reported failed check.'
                description+=' '+body.notes+' Human observation requires triage; no diagnosis or corrective-work approval is implied.'
                conn.execute('INSERT INTO issues VALUES(?,?,?,?,?,?,?,?,?,NULL,1,?,?)',(iid,user['plant_id'],step['asset_id'],user['id'],now,('Inspection finding: '+step['title'])[:160],description,'medium','reported',now,f'round:{rid}:{index}'))
                conn.execute('INSERT INTO issue_events VALUES(?,?,?,?,?,?,1)',(identifier(),iid,user['id'],now,'reported',description));audit(conn,user,'inspection_issue',iid,'Linked human finding for triage')
            conn.execute('INSERT INTO inspection_results VALUES(?,?,?,?) ON CONFLICT(run_id,step_index) DO UPDATE SET result=excluded.result,issue_id=excluded.issue_id',(rid,index,json.dumps(result),iid))
            detail.update(step_index=index,before=previous,after=result,issue_id=iid)
        elif body.action=='complete':
            if status!='in_progress' or len(p['results'])!=len(p['content']['steps']):raise HTTPException(409,'Record every check or an explicit skip reason before completion')
            skipped=sum(r['outcome']=='skipped' for r in p['results']);status='completed_with_skips' if skipped else 'completed';detail['skipped_steps']=skipped
        else:status='cancelled'
        version=p['version']+1;now=iso();conn.execute('UPDATE inspection_runs SET status=?,version=?,updated_at=? WHERE id=?',(status,version,now,rid))
        conn.execute('INSERT INTO inspection_events VALUES(?,?,?,?,?,?,?)',(identifier(),rid,version,user['id'],now,body.action,json.dumps(detail)));audit(conn,user,'inspection_'+body.action,rid,'Human inspection record; equipment state and maintenance approval unchanged')
        return run(conn,user,rid)


def evidence(conn,user,asset_id,event_id):
    """An immutable saved observation, reauthorized against the whole round scope."""
    row=conn.execute("SELECT e.*,r.content,r.plant_id FROM inspection_events e JOIN inspection_runs r ON r.id=e.run_id WHERE e.id=? AND e.action='record' AND r.plant_id=?",(event_id,user['plant_id'])).fetchone()
    if not row:raise HTTPException(404,'Inspection evidence unavailable')
    content=json.loads(row['content']);scope(conn,user,content);detail=json.loads(row['detail']);step=content['steps'][detail['step_index']]
    if step['asset_id']!=asset_id:raise HTTPException(404,'Inspection evidence is for another machine')
    observation=detail['after'];notes=observation['notes']
    if quarantine_reason(notes):notes='Notes withheld from AI context because they contain unsafe or embedded instructions.'
    text=f'Human inspection record by {row["actor_id"]} at {row["created_at"]}: {step["title"]}; outcome {observation["outcome"]}.'
    if observation['value'] is not None:text+=f' Recorded {observation["value"]} {step["unit"]}; checklist range {step["lower"]} to {step["upper"]}.'
    text+=' '+notes+' This is a historical human observation, not validated machine health, a diagnosis or authorization for corrective maintenance. Later corrections do not rewrite this event.'
    return dict(title='Inspection: '+step['title'],text=text,timestamp_utc=row['created_at'],role='history')


def records(conn,user,asset_id,limit=20):
    authorize_asset(conn,user,asset_id);seen=set();result=[]
    rows=conn.execute("SELECT e.id,e.run_id,e.detail,e.created_at FROM inspection_events e JOIN inspection_runs r ON r.id=e.run_id WHERE r.plant_id=? AND e.action='record' ORDER BY e.created_at DESC LIMIT 1000",(user['plant_id'],))
    for row in rows:
        detail=json.loads(row['detail']);key=(row['run_id'],detail['step_index'])
        if key in seen:continue
        seen.add(key)
        try:evidence(conn,user,asset_id,row['id'])
        except HTTPException:continue
        result.append(dict(id=row['id'],evidence_kind='inspection',external_id=row['run_id'],timestamp_utc=row['created_at']))
        if len(result)>=limit:break
    return result
