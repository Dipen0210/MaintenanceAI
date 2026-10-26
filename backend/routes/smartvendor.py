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

from typing import Literal
from datetime import datetime,timezone
from backend.inventory import procurement

class Command(Change):
    request_key:str=Field(min_length=16,max_length=100)
class Supplier(Command):
    name:str=Field(min_length=2,max_length=120)
    contact:str=Field(min_length=3,max_length=200)
    reference:str=Field(min_length=3,max_length=250)
    notes:str=Field(default='',max_length=1000)
class SupplierReview(Change):
    expected_version:int=Field(strict=True,ge=1)
    action:Literal['approve','suspend']
    verified:bool=False
class Quotation(Command):
    supplier_id:str
    part_id:str
    reference:str=Field(min_length=3,max_length=250)
    currency:Literal['USD','EUR','GBP','INR','CAD','AUD']
    unit_price_minor:int=Field(strict=True,ge=0,le=1000000000)
    shipping_minor:int=Field(default=0,strict=True,ge=0,le=1000000000)
    tax_minor:int=Field(default=0,strict=True,ge=0,le=1000000000)
    minimum_quantity:int=Field(default=1,strict=True,ge=1,le=100000)
    maximum_quantity:int=Field(default=100000,strict=True,ge=1,le=100000)
    lead_time_days:int=Field(strict=True,ge=0,le=3650)
    valid_until:datetime
    @field_validator('valid_until')
    @classmethod
    def aware(cls,v):
        if v.tzinfo is None:raise ValueError('Quotation expiry needs a timezone')
        return v.astimezone(timezone.utc)
    @model_validator(mode='after')
    def quantities(self):
        if self.minimum_quantity>self.maximum_quantity:raise ValueError('Minimum order quantity exceeds maximum')
        return self
class QuoteReview(Change):
    expected_version:int=Field(strict=True,ge=1)
    action:Literal['verify','withdraw']
    verified:bool=False

@router.get('/suppliers')
def suppliers(user=Depends(principal)):return procurement.supplier_list(user)
@router.post('/suppliers')
def add_supplier(body:Supplier,user=Depends(principal)):return procurement.supplier_create(body,user)
@router.post('/suppliers/{sid}/review')
def review_supplier(sid:str,body:SupplierReview,user=Depends(principal)):return procurement.supplier_review(sid,body,user)
@router.post('/quotes')
def add_quote(body:Quotation,user=Depends(principal)):return procurement.quote_create(body,user)
@router.post('/quotes/{qid}/review')
def review_quote(qid:str,body:QuoteReview,user=Depends(principal)):return procurement.quote_review(qid,body,user)
@router.get('/quotes/compare')
def compare_quotes(part_id:str,quantity:int,user=Depends(principal)):return procurement.compare(user,part_id,quantity)
