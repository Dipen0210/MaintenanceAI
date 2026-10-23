from fastapi import APIRouter,Depends
from backend.core.database import transaction
from backend.core.security import principal,require_roles
from backend.scheduling.contracts import DraftRequest,Review,ProposalRequest,PlanningConfig,WorkOrderResponse,ProposalResponse,ConfigResponse,FieldUpdate
from backend.scheduling import service

router=APIRouter(prefix='/api',tags=['maintenance planning'])

@router.get('/planning/config',response_model=ConfigResponse)
def configuration(user=Depends(principal)):return service.read_config(user)

@router.put('/planning/config',response_model=ConfigResponse)
def configure(body:PlanningConfig,user=Depends(require_roles('administrator','maintenance_manager'))):return service.save_config(body,user)

@router.get('/assets/{asset_id}/procedures')
def procedures(asset_id:str,user=Depends(principal)):
    from backend.core.security import authorize_asset
    from fastapi import HTTPException
    with transaction() as conn:
        authorize_asset(conn,user,asset_id);result=[]
        for r in conn.execute("SELECT id FROM document_chunks WHERE kind='check' AND reviewed=1").fetchall():
            try:result.extend(service.procedures(conn,user,asset_id,[r['id']]))
            except HTTPException:continue
        return result

@router.get('/work-orders',response_model=list[WorkOrderResponse])
def work_orders(asset_id:str|None=None,user=Depends(principal)):
    with transaction() as conn:return service.list_orders(conn,user,asset_id)

@router.post('/work-orders/draft',response_model=WorkOrderResponse)
@router.post('/work-orders/drafts',response_model=WorkOrderResponse)
def draft(body:DraftRequest,user=Depends(principal)):return service.create_draft(body,user)

@router.post('/work-orders/{wid}/review',response_model=WorkOrderResponse)
def review_work(wid:str,body:Review,user=Depends(require_roles('maintenance_manager'))):return service.review_order(wid,body,user)

@router.post('/work-orders/{wid}/field-update',response_model=WorkOrderResponse)
def field_update(wid:str,body:FieldUpdate,user=Depends(principal)):
    from backend.scheduling.execution import update
    return update(wid,body,user)

@router.get('/schedules',response_model=list[ProposalResponse])
def schedules(user=Depends(principal)):return service.list_proposals(user)

@router.post('/schedules/propose',response_model=ProposalResponse)
def propose(body:ProposalRequest,user=Depends(principal)):return service.propose(body,user)

@router.post('/schedules/{pid}/approve',response_model=ProposalResponse)
def approve(pid:str,body:Review,user=Depends(require_roles('maintenance_manager'))):return service.review_proposal(pid,body,user)
