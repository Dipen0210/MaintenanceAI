from fastapi import APIRouter,Depends,HTTPException
from backend.core.database import transaction
from backend.core.security import principal
from backend.preventive.contracts import PlanInput,PlanRevision,PlanState
from backend.preventive import service

router=APIRouter(prefix='/api/preventive-plans',tags=['preventive maintenance'])

@router.get('')
def plans(user=Depends(principal)):
    with transaction() as conn:
        result=[]
        for row in conn.execute('SELECT id FROM preventive_plans WHERE plant_id=? ORDER BY updated_at DESC LIMIT 500',(user['plant_id'],)):
            try:result.append(service.view(conn,user,row['id']))
            except HTTPException:continue
        return result

@router.post('')
def create(body:PlanInput,user=Depends(principal)):return service.save(body,user)

@router.put('/{pid}')
def revise(pid:str,body:PlanRevision,user=Depends(principal)):return service.save(body,user,pid)

@router.post('/{pid}/state')
def state(pid:str,body:PlanState,user=Depends(principal)):return service.set_state(pid,body,user)

@router.post('/{pid}/generate')
def generate(pid:str,user=Depends(principal)):return service.generate(pid,user)
