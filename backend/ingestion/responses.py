"""Typed public API responses for the data foundation."""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel,FiniteFloat


class UserResponse(BaseModel):
    id:str
    username:str
    role:Literal['administrator','engineer','maintenance_manager','reliability_engineer','operator','operations_coordinator']
    plant_id:str


class LoginResponse(BaseModel):
    user:UserResponse
    access_token:str
    token_type:Literal['bearer']


class LogoutResponse(BaseModel):
    status:Literal['logged_out']


class AssetResponse(BaseModel):
    id:str
    plant_id:str
    name:str
    equipment_type:str
    area:str
    operating_state:str
    safety_criticality:int
    production_criticality:int
    data_quality:str|None=None
    data_mode:str|None=None
    condition:Literal['normal','warning','critical','unknown']='unknown'
    condition_label:str='Awaiting readings'
    condition_detail:str='No current condition evidence'
    condition_tags:list[str]=[]
    last_observed:datetime|None=None


class SourceResponse(BaseModel):
    id:str
    plant_id:str
    name:str
    kind:str
    read_only:Literal[1]
    status:str
    last_received:datetime|None


class SensorResponse(BaseModel):
    id:str
    asset_id:str
    tag_id:str
    unit:str
    lower_bound:FiniteFloat
    upper_bound:FiniteFloat
    stale_after_seconds:int
    source_id:str


class ObservationResponse(BaseModel):
    id:str
    asset_id:str
    tag_id:str
    timestamp_utc:datetime
    value:FiniteFloat
    unit:str
    quality:Literal['good','uncertain','bad']
    source:str
    operating_state:Literal['running','stopped','maintenance','unknown']
    ingestion_timestamp:datetime


class IngestionStatus(BaseModel):
    status:Literal['accepted','flagged','rejected']
    reason:str


class ImportResult(IngestionStatus):
    asset_id:str
    tag_id:str|None


class ImportResponse(BaseModel):
    results:list[ImportResult]


class LatestReading(BaseModel):
    sensor:SensorResponse
    observation:ObservationResponse|None
    effective_quality:Literal['missing','good','uncertain','bad','stale']
    last_ingestion:IngestionStatus|None


class TelemetryResponse(BaseModel):
    asset_id:str
    latest:list[LatestReading]
    history:list[ObservationResponse]
    server_time:datetime


class AlarmResponse(BaseModel):
    id:str
    asset_id:str
    severity:str
    source:str
    state:str
    timestamp_utc:datetime
    message:str
