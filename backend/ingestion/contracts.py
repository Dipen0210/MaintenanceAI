"""Normalized scalar telemetry; waveforms use separate recording references."""
from datetime import datetime,timezone
from typing import Literal
from pydantic import BaseModel,Field,FiniteFloat,field_validator,ConfigDict


class TelemetryRecord(BaseModel):
    model_config=ConfigDict(extra='forbid')
    asset_id:str=Field(min_length=1,max_length=100)
    tag_id:str=Field(min_length=1,max_length=100)
    timestamp_utc:datetime
    value:FiniteFloat
    unit:str=Field(min_length=1,max_length=32)
    quality:Literal['good','uncertain','bad']
    source:str=Field(min_length=1,max_length=100)
    operating_state:Literal['running','stopped','maintenance','unknown']
    ingestion_timestamp:datetime | None=None

    @field_validator('timestamp_utc','ingestion_timestamp')
    @classmethod
    def utc_timestamp(cls,value):
        if value is None:return value
        if value.tzinfo is None or value.utcoffset() is None:raise ValueError('Timezone-aware timestamp required')
        return value.astimezone(timezone.utc)
