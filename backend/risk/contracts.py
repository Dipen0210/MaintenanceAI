from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,FiniteFloat,model_validator


class Limit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    unit:str=Field(min_length=1,max_length=30)
    warning:FiniteFloat
    critical:FiniteFloat
    @model_validator(mode='after')
    def ordered(self):
        if self.warning>=self.critical:raise ValueError('Warning limit must be below critical limit')
        return self


class Criticality(BaseModel):
    model_config=ConfigDict(extra='forbid')
    safety:int=Field(ge=1,le=5)
    production:int=Field(ge=1,le=5)
    maintenance:int=Field(ge=1,le=5)
    damage:int=Field(ge=1,le=5)
    dependencies:list[str]=Field(default_factory=list,max_length=30)
    limits:dict[str,Limit]=Field(min_length=1,max_length=20)
    required_sensors:list[str]=Field(min_length=1,max_length=20)
    operating_states:list[Literal['running','stopped','maintenance']]=Field(default=['running'],min_length=1,max_length=3)
    escalation:str=Field(min_length=10,max_length=500)
    reason:str=Field(min_length=5,max_length=500)

    @model_validator(mode='after')
    def complete(self):
        if set(self.required_sensors)-set(self.limits):raise ValueError('Every required channel needs an approved limit')
        if len(set(self.dependencies))!=len(self.dependencies):raise ValueError('Duplicate dependency')
        return self


class RiskPolicy(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str=Field(min_length=1,max_length=100)
    safety_weight:float=Field(ge=0,le=1,allow_inf_nan=False)
    production_weight:float=Field(ge=0,le=1,allow_inf_nan=False)
    maintenance_weight:float=Field(ge=0,le=1,allow_inf_nan=False)
    damage_weight:float=Field(ge=0,le=1,allow_inf_nan=False)
    critical_score:float=Field(ge=1,le=100,allow_inf_nan=False)
    high_score:float=Field(ge=1,le=100,allow_inf_nan=False)
    planned_score:float=Field(ge=1,le=100,allow_inf_nan=False)
    prediction_max_age_seconds:int=Field(ge=1,le=3600)
    reason:str=Field(min_length=5,max_length=500)

    @model_validator(mode='after')
    def consistent(self):
        if abs(self.safety_weight+self.production_weight+self.maintenance_weight+self.damage_weight-1)>1e-6:raise ValueError('Consequence weights must sum to one')
        if not self.planned_score<self.high_score<self.critical_score:raise ValueError('Priority thresholds must be ordered')
        return self


class EvaluateRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    asset_ids:list[str]=Field(min_length=1,max_length=100)


class ConfigResponse(BaseModel):
    id:str
    config:dict
    timestamp_utc:str


class RiskResponse(BaseModel):
    id:str
    asset_id:str
    policy_id:str
    profile_id:str
    timestamp_utc:str
    priority:Literal['Critical review required','High priority','Planned maintenance','Monitor','Insufficient evidence']
    stale:bool=False
    expires_at:str
    rank:int
    score:FiniteFloat|None
    suspected_issue:str
    severity:str
    components:dict
    evidence:list[dict]
    model_output:dict|None
    criticality:dict
    explanation:list[str]
    missing_evidence:list[str]
    confidence:str
    required_verification:list[str]
    suggested_action:str
    human_review_required:Literal[True]
    escalation:str|None
    source_ids:list[str]
