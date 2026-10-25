from fastapi import APIRouter,Depends,HTTPException
from backend.core.database import transaction
from backend.core.security import principal,authorize_asset
from backend.preventive import meters
from backend.preventive.contracts import PlanState

router=APIRouter(prefix='/api',tags=['meter maintenance'])

@router.get('/assets/{aid}/meters')
def mappings(aid:str,user=Depends(principal)):
    with transaction() as conn:
        authorize_asset(conn,user,aid);result=[]
        for row in conn.execute("SELECT id FROM sensors WHERE asset_id=? AND unit IN ('h','cycles') ORDER BY tag_id",(aid,)):
            try:result.append(meters.sensor(conn,user,row['id']))
            except HTTPException:continue
        return result

@router.post('/assets/{aid}/meters')
def register(aid:str,body:meters.MeterRegistration,user=Depends(principal)):return meters.register(aid,body,user)

@router.get('/meter-plans')
def plans(user=Depends(principal)):
    with transaction() as conn:
        result=[]
        for row in conn.execute('SELECT id FROM meter_plans WHERE plant_id=? ORDER BY updated_at DESC LIMIT 500',(user['plant_id'],)):
            try:result.append(meters.view(conn,user,row['id']))
            except HTTPException:continue
        return result

@router.post('/meter-plans')
def create(body:meters.MeterInput,user=Depends(principal)):return meters.save(body,user)

@router.put('/meter-plans/{pid}')
def revise(pid:str,body:meters.MeterRevision,user=Depends(principal)):return meters.save(body,user,pid)

@router.post('/meter-plans/{pid}/state')
def state(pid:str,body:PlanState,user=Depends(principal)):return meters.state(pid,body,user)

@router.post('/meter-plans/{pid}/generate')
def generate(pid:str,user=Depends(principal)):return meters.generate(pid,user)

@router.post('/meter-plans/{pid}/reconcile')
def reconcile(pid:str,body:meters.Reconcile,user=Depends(principal)):return meters.reconcile(pid,body,user)
