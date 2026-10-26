"""Personal inbox actions and administrator-owned escalation policy."""
from typing import Literal
from fastapi import APIRouter,Depends
from pydantic import Field,field_validator
from backend.core.database import transaction
from backend.core.security import principal,require_roles
from backend.scheduling.contracts import Strict
from backend.notifications import service

router=APIRouter(prefix='/api',tags=['notifications'])

class Update(Strict):
    action:Literal['read','acknowledge','retry']
    reason:str|None=Field(default=None,max_length=500)

class Policy(Strict):
    expected_version:int=Field(ge=0)
    escalation_minutes:int=Field(ge=1,le=1440)
    escalation_enabled:bool
    reason:str=Field(min_length=15,max_length=500)

    @field_validator('reason')
    @classmethod
    def meaningful(cls,v):
        if len(v.strip())<15:raise ValueError('Record a meaningful policy reason')
        return v.strip()

@router.get('/notifications')
def inbox(history:bool=False,user=Depends(principal)):return service.inbox(user,history)

@router.post('/notifications/{did}/updates')
def update(did:str,body:Update,user=Depends(principal)):return service.update(did,body,user)

@router.get('/notification-policy')
def policy(user=Depends(principal)):
    with transaction() as conn:return service.policy(conn,user['plant_id'])

@router.put('/notification-policy')
def configure(body:Policy,user=Depends(require_roles('administrator'))):return service.configure(body,user)
