from datetime import datetime,timezone
from typing import Literal
from pydantic import BaseModel,Field,FiniteFloat,ConfigDict,field_validator,model_validator


class SignalInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    source_id:str=Field(max_length=100)
    kind:Literal['vibration','sequence']
    timestamp_utc:datetime
    sample_rate:int|None=Field(default=None,ge=1,le=192000)
    sensor_position:str=Field(min_length=1,max_length=100)
    dataset_id:str=Field(min_length=1,max_length=100)
    quality:Literal['good','bad','uncertain']='good'
    operating_state:Literal['running','stopped','maintenance','unknown']='running'
    historical:bool=False
    feature_names:list[str]|None=Field(default=None,max_length=17)
    values:list[FiniteFloat]|list[list[FiniteFloat]]=Field(min_length=1,max_length=65536)

    @field_validator('timestamp_utc')
    @classmethod
    def aware(cls,value):
        if value.tzinfo is None:raise ValueError('UTC-aware timestamp required')
        return value.astimezone(timezone.utc)

    @model_validator(mode='after')
    def dimensions(self):
        if self.kind=='sequence':
            if len(self.values)>10000 or any(not isinstance(r,list) or len(r)!=17 for r in self.values):raise ValueError('Expected at most 10000 rows of 17 features')
            expected=['op1','op2','op3']+[f'sensor_{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
            if self.feature_names!=expected:raise ValueError('C-MAPSS feature names and order must match the published 17-feature contract')
        elif any(isinstance(r,list) for r in self.values) or not self.sample_rate:raise ValueError('Vibration requires scalar samples and sampling rate')
        return self


class PredictionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    family:str=Field(min_length=1,max_length=100)
    input_id:str|None=None
    model_id:str|None=None


class ActivationRequest(BaseModel):
    reason:str=Field(min_length=5,max_length=500)


class PredictionResponse(BaseModel):
    id:str
    asset_id:str
    model_id:str
    model_version:str
    family:str
    dataset_id:str
    preprocessing_version:str
    status:Literal['SUCCEEDED','UNSUPPORTED_ASSET','UNSUPPORTED_STATE','MISSING_INPUT','UNSUPPORTED_INPUT','INVALID_INPUT','STALE_INPUT','MODEL_ERROR']
    reason:str|None
    timestamp_utc:datetime
    input_id:str|None
    input_quality:str
    historical:bool
    signal_timestamp:datetime|None
    output:dict
    evidence:dict
    drift:dict|None
    uncertainty_interval:None
    uncertainty:str
    limitations:str
    data_mode:str
    latency_ms:FiniteFloat


class SignalResponse(BaseModel):
    id:str
    asset_id:str
    source_id:str
    kind:Literal['vibration','sequence','audio']
    sha256:str
    metadata:dict


class ModelResponse(BaseModel):
    id:str
    family:str
    version:str
    kind:Literal['anomaly','fault','rul','condition']
    filename:str
    sha256:str
    operating_states:list[str]
    mode:Literal['reference','simulation']
    uncertainty:str
    limitations:str
    dataset_id:str
    preprocessing_version:str
    equipment_types:list[str]
    input_kind:Literal['audio','vibration','sequence']
    sample_rate:int|None
    active:bool
    available:bool
    metrics:dict


class ActivationResponse(BaseModel):
    activation_id:str
    previous_id:str|None
    active_id:str


class ActivationEvent(BaseModel):
    id:str
    family:str
    previous_id:str|None
    next_id:str
    actor_id:str
    timestamp_utc:datetime
    reason:str


class Binding(BaseModel):
    asset_id:str
    family:str
    dataset_id:str


class BindingsResponse(BaseModel):
    asset_id:str
    equipment_type:str
    bindings:list[Binding]
    rul_notice:str


class MonitoringResponse(BaseModel):
    observations:int
    failures:int
    distribution_warnings:int
    latency_p95_ms:FiniteFloat|None
    scope:str
    calibration:str
