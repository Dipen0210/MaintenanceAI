"""Attributed operations approval and handover; never equipment control."""
from typing import Literal
from fastapi import APIRouter,Depends
from pydantic import Field,field_validator
from backend.core.database import transaction
from backend.core.security import principal,authorize_asset
from backend.scheduling.contracts import Strict,WorkOrderResponse
from backend.scheduling import coordination as service

router=APIRouter(prefix='/api',tags=['operations coordination'])

class Reason(Strict):
    expected_version:int=Field(ge=0)
    reason:str=Field(min_length=15,max_length=1000)

    @field_validator('reason')
    @classmethod
    def useful(cls,value):
        if len(value.strip())<15:raise ValueError('Describe the decision in at least 15 characters')
        return value.strip()

class Policy(Reason):
    required:bool

class Decision(Reason):
    decision:Literal['approved','declined']

class Handover(Reason):
    action:Literal['release','hold','confirm_return','decline_return']

@router.get('/assets/{aid}/operations-policy')
def policy(aid:str,user=Depends(principal)):
    with transaction() as conn:
        authorize_asset(conn,user,aid)
        return {**service.policy(conn,aid),'history':[dict(r) for r in conn.execute('SELECT * FROM operations_policy_history WHERE asset_id=? ORDER BY version DESC',(aid,))]}

@router.put('/assets/{aid}/operations-policy')
def configure(aid:str,body:Policy,user=Depends(principal)):return service.configure(aid,body,user)

@router.get('/operations/requests')
def requests(user=Depends(principal)):return service.requests(user)

@router.post('/operations/requests/{rid}/decision')
def decide(rid:str,body:Decision,user=Depends(principal)):return service.decide(rid,body,user)

@router.post('/work-orders/{wid}/operations-handover',response_model=WorkOrderResponse)
def handover(wid:str,body:Handover,user=Depends(principal)):return service.handover(wid,body,user)
