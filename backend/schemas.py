"""Validated legacy machine contracts, shared by HTTP and in-process updates."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Cycles = Annotated[float, Field(ge=0, le=1000000, allow_inf_nan=False)]


class MachineInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    machine_id: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_-]+$')
    machine_type: Literal['fan', 'pump', 'valve', 'bearing']
    area: str = Field(default='unknown', min_length=1, max_length=100)
    anomaly_score: Score = 0
    fault_type: Literal['Normal', 'Ball_007', 'Ball_014', 'Ball_021', 'IR_007', 'IR_014', 'IR_021', 'OR_007', 'OR_014', 'OR_021'] | None = None
    fault_confidence: Score = 0
    rul_cycles: Cycles | None = None
