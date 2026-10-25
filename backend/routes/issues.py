"""Human-reported issues and named queues. An assignment never authorizes maintenance."""
import io
from typing import Literal
from fastapi import APIRouter,Depends,HTTPException,UploadFile,File,Response
from PIL import Image,ImageOps,UnidentifiedImageError
from pydantic import Field
from backend.core.database import transaction,identifier,iso,audit
from backend.core.security import principal,authorize_asset
from backend.scheduling.contracts import Strict
from backend.scheduling.service import role,order

router=APIRouter(prefix='/api',tags=['field issue intake'])


class Report(Strict):
    asset_id:str
    title:str=Field(min_length=5,max_length=160)
    description:str=Field(min_length=15,max_length=3000)
    reported_severity:Literal['low','medium','high','urgent']='medium'
    request_key:str=Field(min_length=16,max_length=100)


class Change(Strict):
    action:Literal['assign','start','note','close','reopen']
    expected_version:int=Field(ge=1)
    assignee_id:str|None=None
    notes:str=Field(min_length=15,max_length=2000)


def view(conn,user,iid):
    row=conn.execute('SELECT * FROM issues WHERE id=? AND plant_id=?',(iid,user['plant_id'])).fetchone()
    if not row:raise HTTPException(404,'Issue not found or not authorized')
    authorize_asset(conn,user,row['asset_id'])
    result=dict(row);result.pop('request_key')
    result['assignee_name']=None
    if row['assignee_id']:
        person=conn.execute('SELECT username,enabled FROM users WHERE id=?',(row['assignee_id'],)).fetchone()
        result['assignee_name']=person['username'] if person else None
        result['assignee_enabled']=bool(person and person['enabled'])
    result['events']=[dict(r) for r in conn.execute('SELECT actor_id,created_at,action,notes,version FROM issue_events WHERE issue_id=? ORDER BY version',(iid,))]
    result['photos']=[dict(r) for r in conn.execute('SELECT id,created_at FROM issue_photos WHERE issue_id=? ORDER BY created_at',(iid,))]
    return result


def eligible(conn,user,asset_id=None):
    if asset_id:authorize_asset(conn,user,asset_id)
    result=[]
    for r in conn.execute("SELECT id,username,role,plant_id FROM users WHERE plant_id=? AND enabled=1 AND role IN('engineer','reliability_engineer') ORDER BY username",(user['plant_id'],)):
        try:
            if asset_id:authorize_asset(conn,dict(r),asset_id)
        except HTTPException:continue
        result.append({k:r[k] for k in ('id','username','role')})
    return result


@router.get('/team/engineers')
def engineers(asset_id:str|None=None,user=Depends(principal)):
    with transaction() as conn:
        role(conn,user,['administrator','maintenance_manager'])
        return eligible(conn,user,asset_id)


@router.get('/issues')
def issues(asset_id:str|None=None,mine:bool=False,user=Depends(principal)):
    with transaction() as conn:
        if asset_id:authorize_asset(conn,user,asset_id)
        result=[]
        for row in conn.execute('SELECT id FROM issues WHERE plant_id=? AND (? IS NULL OR asset_id=?) AND (?=0 OR assignee_id=?) ORDER BY updated_at DESC LIMIT 500',(user['plant_id'],asset_id,asset_id,int(mine),user['id'])):
            try:result.append(view(conn,user,row['id']))
            except HTTPException:continue
        return result


@router.post('/issues')
def report(body:Report,user=Depends(principal)):
    if len(body.title.strip())<5 or len(body.description.strip())<15:raise HTTPException(422,'Describe the observed issue using meaningful text')
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE')
        role(conn,user,['operator','engineer','reliability_engineer','maintenance_manager','administrator'])
        authorize_asset(conn,user,body.asset_id)
        prior=conn.execute('SELECT * FROM issues WHERE created_by=? AND request_key=?',(user['id'],body.request_key)).fetchone()
        if prior:
            if any(prior[key]!=getattr(body,key) for key in ('asset_id','title','description','reported_severity')):raise HTTPException(409,'This request key was already used for a different report')
            return view(conn,user,prior['id'])
        iid=identifier();now=iso()
        conn.execute('INSERT INTO issues VALUES(?,?,?,?,?,?,?,?,?,NULL,1,?,?)',(iid,user['plant_id'],body.asset_id,user['id'],now,body.title,body.description,body.reported_severity,'reported',now,body.request_key))
        conn.execute('INSERT INTO issue_events VALUES(?,?,?,?,?,?,1)',(identifier(),iid,user['id'],now,'reported',body.description))
        audit(conn,user,'report_issue',iid,'Human observation; reported severity is not a validated risk score')
        return view(conn,user,iid)


@router.post('/issues/{iid}/updates')
def update(iid:str,body:Change,user=Depends(principal)):
    if len(body.notes.strip())<15:raise HTTPException(422,'Record a meaningful update')
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');current=view(conn,user,iid)
        if current['version']!=body.expected_version:raise HTTPException(409,'Issue changed. Refresh before updating it.')
        status=current['status'];assignee=current['assignee_id']
        if body.action in ['assign','close','reopen']:
            role(conn,user,['maintenance_manager'])
            if body.action=='assign':
                if status=='closed':raise HTTPException(409,'Reopen the issue before assigning it')
                if body.assignee_id not in {r['id'] for r in eligible(conn,user,current['asset_id'])}:raise HTTPException(422,'Choose an enabled engineer with access to this machine')
                assignee=body.assignee_id;status='assigned'
            elif body.action=='close':
                if status=='closed':raise HTTPException(409,'Issue is already closed')
                status='closed'
            else:
                if status!='closed':raise HTTPException(409,'Only closed issues can be reopened')
                status='reported';assignee=None
        else:
            role(conn,user,['engineer','reliability_engineer','maintenance_manager'])
            if status=='closed':raise HTTPException(409,'Closed issues require manager reopening')
            if user['role']!='maintenance_manager' and assignee!=user['id']:raise HTTPException(403,'Only the assigned engineer or manager may update this investigation')
            if body.action=='start':
                if assignee!=user['id'] or status!='assigned':raise HTTPException(409,'Only the assigned engineer may start an assigned investigation')
                status='investigating'
        version=current['version']+1;now=iso()
        conn.execute('UPDATE issues SET status=?,assignee_id=?,version=?,updated_at=? WHERE id=?',(status,assignee,version,now,iid))
        conn.execute('INSERT INTO issue_events VALUES(?,?,?,?,?,?,?)',(identifier(),iid,user['id'],now,body.action,body.notes,version))
        audit(conn,user,'issue_'+body.action,iid,'Investigation record only; equipment/work status unchanged')
        return view(conn,user,iid)


@router.post('/issues/{iid}/photos')
def add_photo(iid:str,file:UploadFile=File(...),user=Depends(principal)):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE');current=view(conn,user,iid)
        role(conn,user,['operator','engineer','reliability_engineer','maintenance_manager','administrator'])
        if current['status']=='closed':raise HTTPException(409,'Closed issue evidence cannot be changed')
        if user['id'] not in [current['created_by'],current['assignee_id']] and user['role']!='maintenance_manager':raise HTTPException(403,'Only the reporter, assignee or manager may attach issue evidence')
        if len(current['photos'])>=5:raise HTTPException(422,'At most five photos per issue')
        content=file.file.read(5*1024*1024+1)
        if len(content)>5*1024*1024:raise HTTPException(413,'Photo must be under 5 MiB')
        try:
            with Image.open(io.BytesIO(content)) as img:
                if img.format not in ['PNG','JPEG'] or img.width*img.height>12_000_000:raise ValueError()
                normalized=ImageOps.exif_transpose(img).convert('RGB');normalized.thumbnail((2000,2000))
                output=io.BytesIO();normalized.save(output,format='JPEG',quality=85)
        except (UnidentifiedImageError,ValueError,OSError,Image.DecompressionBombError) as exc:raise HTTPException(422,'Use a valid JPEG or PNG photo of at most 12 megapixels') from exc
        pid=identifier();now=iso()
        conn.execute('INSERT INTO issue_photos VALUES(?,?,?,?,?,?)',(pid,iid,user['id'],now,output.getvalue(),'image/jpeg'))
        audit(conn,user,'attach_issue_photo',iid,'Normalized photo; embedded metadata removed')
        return {'id':pid}


@router.get('/issues/{iid}/photos/{pid}')
def photo(iid:str,pid:str,user=Depends(principal)):
    with transaction() as conn:
        view(conn,user,iid)
        row=conn.execute('SELECT content,mime_type FROM issue_photos WHERE id=? AND issue_id=?',(pid,iid)).fetchone()
        if not row:raise HTTPException(404,'Photo unavailable')
        return Response(row['content'],media_type=row['mime_type'],headers={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'})


@router.get('/my-work')
def my_work(user=Depends(principal)):
    with transaction() as conn:
        result=[]
        for row in conn.execute('SELECT DISTINCT work_order_id FROM work_user_assignments WHERE user_id=?',(user['id'],)):
            try:
                item=order(conn,user,row[0])
                if item['status']!='completed':result.append(item)
            except HTTPException:continue
    return {'issues':[i for i in issues(mine=True,user=user) if i['status']!='closed'],'work_orders':result}
