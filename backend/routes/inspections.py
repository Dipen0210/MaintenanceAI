from fastapi import APIRouter,Depends,HTTPException
from backend.core.database import transaction
from backend.core.security import principal
from backend.inspections import service

router=APIRouter(prefix='/api/inspections',tags=['inspection rounds'])

@router.get('/templates')
def templates(user=Depends(principal)):
    with transaction() as conn:
        result=[]
        for row in conn.execute('SELECT id FROM inspection_templates WHERE plant_id=? ORDER BY updated_at DESC LIMIT 500',(user['plant_id'],)):
            try:result.append(service.template(conn,user,row['id']))
            except HTTPException:continue
        return result

@router.post('/templates')
def create(body:service.TemplateInput,user=Depends(principal)):return service.save(body,user)

@router.put('/templates/{tid}')
def revise(tid:str,body:service.TemplateRevision,user=Depends(principal)):return service.save(body,user,tid)

@router.post('/templates/{tid}/assign')
def assign(tid:str,body:service.Assignment,user=Depends(principal)):return service.assign(tid,body,user)

@router.get('/runs')
def runs(mine:bool=False,user=Depends(principal)):
    with transaction() as conn:
        result=[]
        for row in conn.execute('SELECT id FROM inspection_runs WHERE plant_id=? AND (?=0 OR assignee_id=?) ORDER BY due_at DESC LIMIT 500',(user['plant_id'],int(mine),user['id'])):
            try:result.append(service.run(conn,user,row['id']))
            except HTTPException:continue
        return result

@router.post('/runs/{rid}/updates')
def update(rid:str,body:service.Update,user=Depends(principal)):return service.update(rid,body,user)
