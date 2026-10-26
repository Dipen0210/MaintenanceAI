from datetime import datetime,timezone
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,field_validator,model_validator


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')


class Period(Strict):
    start_utc:datetime
    end_utc:datetime

    @field_validator('start_utc','end_utc')
    @classmethod
    def aware(cls,value):
        if value.tzinfo is None:raise ValueError('Timezone-aware UTC timestamps required')
        return value.astimezone(timezone.utc)

    @model_validator(mode='after')
    def ordered(self):
        if self.start_utc>=self.end_utc:raise ValueError('Period end must be after its start')
        return self


class Technician(Strict):
    id:str=Field(min_length=1,max_length=80)
    name:str=Field(min_length=1,max_length=120)
    user_id:str|None=Field(default=None,min_length=1,max_length=100)
    skills:list[str]=Field(min_length=1,max_length=20)
    availability:list[Period]=Field(min_length=1,max_length=20)

    @field_validator('skills')
    @classmethod
    def skills_valid(cls,value):
        value=[v.strip() for v in value]
        if any(not v or len(v)>80 for v in value) or len(set(value))!=len(value):raise ValueError('Skills must be nonempty, unique and at most 80 characters')
        return value


class Part(Strict):
    id:str=Field(min_length=1,max_length=80)
    name:str=Field(min_length=1,max_length=120)
    quantity:int=Field(ge=0,le=100000)
    available_at:datetime

    @field_validator('available_at')
    @classmethod
    def aware(cls,value):
        if value.tzinfo is None:raise ValueError('Parts availability needs a timezone-aware timestamp')
        return value.astimezone(timezone.utc)


class EquipmentWindow(Period):
    asset_id:str
    id:str=Field(min_length=1,max_length=80)
    disruption_cost_per_minute:int=Field(ge=0,le=1000)


class Blackout(Period):
    asset_id:str
    reason:str=Field(min_length=5,max_length=300)


class PlanningConfig(Strict):
    technicians:list[Technician]=Field(min_length=1,max_length=12)
    parts:list[Part]=Field(default_factory=list,max_length=100)
    equipment_windows:list[EquipmentWindow]=Field(min_length=1,max_length=100)
    blackouts:list[Blackout]=Field(default_factory=list,max_length=100)
    reason:str=Field(min_length=10,max_length=500)
    simulation_only:bool=True

    @model_validator(mode='after')
    def unique(self):
        for items in [self.technicians,self.parts,self.equipment_windows]:
            if len({x.id for x in items})!=len(items):raise ValueError('Resource and window IDs must be unique')
        return self


class DraftRequest(Strict):
    asset_id:str
    title:str=Field(min_length=5,max_length=200)
    task:str=Field(min_length=15,max_length=2000)
    duration_minutes:int=Field(ge=5,le=480)
    required_skills:list[str]=Field(min_length=1,max_length=10)
    team_size:int=Field(default=1,ge=1,le=3)
    parts:dict[str,int]=Field(default_factory=dict,max_length=30)
    dependencies:list[str]=Field(default_factory=list,max_length=20)
    procedure_ids:list[str]=Field(min_length=1,max_length=20)
    risk_id:str
    investigation_id:str|None=None
    due_at:datetime

    @field_validator('due_at')
    @classmethod
    def aware(cls,value):
        if value.tzinfo is None:raise ValueError('Due time must include timezone information')
        return value.astimezone(timezone.utc)

    @model_validator(mode='after')
    def valid_requirements(self):
        self.required_skills=[v.strip() for v in self.required_skills]
        if any(len(v)>80 for v in self.required_skills):raise ValueError('Skill names must be at most 80 characters')
        if any(not key.strip() or len(key)>80 for key in self.parts):raise ValueError('Invalid part identifier')
        if any(q<=0 or q>10000 for q in self.parts.values()):raise ValueError('Part quantities must be 1–10000')
        for items in [self.required_skills,self.dependencies,self.procedure_ids]:
            if len(items)!=len(set(items)) or any(not x.strip() for x in items):raise ValueError('Requirements must be nonempty and unique')
        return self


class Review(Strict):
    decision:Literal['approved','rejected']
    reason:str=Field(min_length=10,max_length=1000)


class FieldUpdate(Strict):
    action:Literal['start','note','report_fixed','verify','return_to_work']
    expected_version:int=Field(ge=0)
    notes:str=Field(min_length=15,max_length=3000)
    site_clearance_confirmed:bool=False

    @field_validator('notes')
    @classmethod
    def useful_notes(cls,value):
        if len(value.strip())<15:raise ValueError('Describe the work or handover in at least 15 characters')
        return value.strip()


class ProposalRequest(Period):
    work_order_ids:list[str]=Field(min_length=1,max_length=20)

    @model_validator(mode='after')
    def bounded(self):
        if (self.end_utc-self.start_utc).total_seconds()>14*86400:raise ValueError('Planning horizon is limited to 14 days')
        if len(self.work_order_ids)!=len(set(self.work_order_ids)):raise ValueError('Duplicate work order')
        return self


class WorkOrderResponse(Strict):
    id:str
    asset_id:str
    created_by:str
    created_at:str
    status:str
    content:dict
    reviewed_by:str|None
    reviewed_at:str|None
    review_reason:str|None
    execution:dict
    operations:dict|None


class ProposalResponse(Strict):
    operations_requests:list[dict]
    id:str
    created_at:str
    config_id:str
    status:str
    result:dict
    reviewed_by:str|None
    reviewed_at:str|None
    review_reason:str|None


class ConfigResponse(Strict):
    id:str
    created_at:str
    config:dict
