from typing import Literal
from fastapi import APIRouter,Depends
from pydantic import Field,field_validator,model_validator
from backend.core.security import principal
from backend.scheduling.contracts import Strict
from backend.inventory import service

router=APIRouter(prefix='/api',tags=['counted inventory'])
class Activation(Strict):
    counting_confirmed:bool=False
    reason:str=Field(min_length=15,max_length=500)
    @field_validator('reason')
    @classmethod
    def useful(cls,v):
        from backend.copilot.safety import quarantine_reason
        if quarantine_reason(v):raise ValueError('Unsafe instructions cannot be recorded as stock guidance')
        if len(v.strip())<15:raise ValueError('Record a meaningful stocktake reason')
        return v.strip()

class Movement(Activation):
    action:Literal['receipt','adjust','issue','consume','return']
    quantity:int=Field(strict=True,ge=-1000000,le=1000000)
    expected_version:int=Field(strict=True,ge=0)
    request_key:str=Field(min_length=16,max_length=100)
    @model_validator(mode='after')
    def quantities(self):
        if self.quantity==0 or self.action!='adjust' and self.quantity<0:raise ValueError('Quantity must be positive; only adjustments may be negative')
        return self

@router.get('/inventory/status')
def status(user=Depends(principal)):return service.status(user)
@router.get('/inventory')
def catalog(user=Depends(principal)):return service.catalog(user)
@router.post('/inventory/activate')
def activate(body:Activation,user=Depends(principal)):return service.activate(body,user)
@router.post('/inventory/parts/{pid}/movements')
def move(pid:str,body:Movement,user=Depends(principal)):return service.apply(pid,body,user)
@router.post('/work-orders/{wid}/parts/{pid}/movements')
def work_move(wid:str,pid:str,body:Movement,user=Depends(principal)):return service.apply(pid,body,user,wid)
