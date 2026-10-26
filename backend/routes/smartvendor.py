"""Maintenance procurement; all purchasing decisions remain human approved."""
from fastapi import APIRouter,Depends
from pydantic import Field,field_validator,model_validator
from backend.core.security import principal
from backend.scheduling.contracts import Strict
from backend.inventory import catalog

router=APIRouter(prefix='/api/inventory',tags=['SmartVendor procurement'])
class Change(Strict):
    reason:str=Field(min_length=15,max_length=500)
    @field_validator('reason')
    @classmethod
    def reason_valid(cls,v):
        from backend.copilot.safety import quarantine_reason
        if len(v.strip())<15 or quarantine_reason(v):raise ValueError('Record a meaningful, safe review reason')
        return v.strip()

class PartDetails(Change):
    expected_version:int=Field(strict=True,ge=0)
    manufacturer:str=Field(default='',max_length=120)
    manufacturer_part_number:str=Field(default='',max_length=120)
    unit:str=Field(default='units',min_length=1,max_length=30)
    location:str=Field(default='',max_length=120)
    minimum_stock:int=Field(default=0,strict=True,ge=0,le=100000)
    target_stock:int=Field(default=0,strict=True,ge=0,le=100000)
    lead_time_days:int=Field(default=0,strict=True,ge=0,le=3650)
    compatible_assets:list[str]=Field(default_factory=list,max_length=100)
    substitute_ids:list[str]=Field(default_factory=list,max_length=100)
    @model_validator(mode='after')
    def coherent(self):
        if self.target_stock<self.minimum_stock:raise ValueError('Target stock must be at least minimum stock')
        if not self.unit.strip():raise ValueError('Unit is required')
        for ids in [self.compatible_assets,self.substitute_ids]:
            if len(set(ids))!=len(ids):raise ValueError('Repeated identifiers are not allowed')
        self.unit=self.unit.strip()
        return self

class NewPart(PartDetails):
    part_id:str=Field(min_length=1,max_length=80,pattern=r'^[A-Za-z0-9_.-]+$')
    name:str=Field(min_length=2,max_length=120)

@router.get('/catalog')
def list_catalog(user=Depends(principal)):return catalog.listing(user)
@router.post('/catalog')
def create_part(body:NewPart,user=Depends(principal)):return catalog.save(body.part_id,body,user,True)
@router.put('/catalog/{pid}')
def edit_part(pid:str,body:PartDetails,user=Depends(principal)):return catalog.save(pid,body,user)
