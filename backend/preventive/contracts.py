from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo,ZoneInfoNotFoundError
from pydantic import Field,field_validator,model_validator
from backend.scheduling.contracts import Strict,DraftRequest


class PlanInput(Strict):
    asset_id:str
    title:str=Field(min_length=5,max_length=200)
    task:str=Field(min_length=15,max_length=2000)
    procedure_ids:list[str]=Field(min_length=1,max_length=20)
    duration_minutes:int=Field(ge=5,le=480)
    required_skills:list[str]=Field(min_length=1,max_length=10)
    team_size:int=Field(default=1,ge=1,le=3)
    parts:dict[str,int]=Field(default_factory=dict,max_length=30)
    start_local:datetime
    timezone:str=Field(max_length=100)
    interval_unit:Literal['days','weeks','months']
    interval_count:int=Field(ge=1,le=365)
    grace_hours:int=Field(default=24,ge=0,le=720)
    reason:str=Field(min_length=10,max_length=500)

    @field_validator('start_local')
    @classmethod
    def wall_time(cls,value):
        if value.tzinfo is not None:raise ValueError('Enter local calendar time without a UTC offset and select its timezone')
        if not 2020<=value.year<=2100:raise ValueError('Start year must be 2020–2100')
        return value.replace(second=0,microsecond=0)

    @field_validator('timezone')
    @classmethod
    def timezone_valid(cls,value):
        try:ZoneInfo(value)
        except (ZoneInfoNotFoundError,ValueError) as exc:raise ValueError('Use a valid IANA timezone') from exc
        return value

    @model_validator(mode='after')
    def scope_valid(self):
        from datetime import timezone
        # Reuse maintenance-scope validation without inventing an assessment.
        checked=DraftRequest(asset_id=self.asset_id,title=self.title,task=self.task,procedure_ids=self.procedure_ids,
            duration_minutes=self.duration_minutes,required_skills=self.required_skills,team_size=self.team_size,
            parts=self.parts,risk_id='validation-only',due_at=self.start_local.replace(tzinfo=timezone.utc))
        self.required_skills=checked.required_skills
        if len(self.title.strip())<5 or len(self.task.strip())<15 or len(self.reason.strip())<10:raise ValueError('Use meaningful scope and change notes')
        return self


class PlanRevision(PlanInput):
    expected_version:int=Field(ge=1)


class PlanState(Strict):
    expected_version:int=Field(ge=1)
    enabled:bool
    reason:str=Field(min_length=10,max_length=500)
