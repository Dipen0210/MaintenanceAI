from datetime import datetime,timezone
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,field_validator,model_validator


class DocumentMetadata(BaseModel):
    model_config=ConfigDict(extra='forbid')
    title:str=Field(min_length=1,max_length=200)
    revision:str=Field(min_length=1,max_length=60)
    equipment_types:list[str]=Field(min_length=1,max_length=30)
    asset_ids:list[str]=Field(default_factory=list,max_length=100)
    reader_ids:list[str]=Field(min_length=1,max_length=100)
    simulation_only:bool=True


class ChunkReview(BaseModel):
    id:str
    kind:Literal['reference','hypothesis','alternative','check','safety']
    fault_classes:list[str]=Field(default_factory=list,max_length=20)

    @model_validator(mode='after')
    def hypothesis_scope(self):
        if self.kind=='hypothesis' and not self.fault_classes:raise ValueError('Hypotheses require explicit supporting model classes')
        return self


class DocumentApproval(BaseModel):
    model_config=ConfigDict(extra='forbid')
    chunks:list[ChunkReview]=Field(min_length=1,max_length=1000)
    reason:str=Field(min_length=10,max_length=500)


class RecordInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    asset_id:str=Field(min_length=1,max_length=100)
    external_id:str=Field(min_length=1,max_length=150)
    record_type:Literal['fault','inspection','work_order','schedule']
    timestamp_utc:datetime
    summary:str=Field(min_length=1,max_length=3000)
    details:dict=Field(default_factory=dict)

    @field_validator('timestamp_utc')
    @classmethod
    def aware(cls,value):
        if value.tzinfo is None:raise ValueError('UTC-aware timestamp required')
        return value.astimezone(timezone.utc)


class ToolArguments(BaseModel):
    model_config=ConfigDict(extra='forbid')
    asset_id:str=Field(min_length=1,max_length=100)
    query:str=Field(default='',max_length=2000)
    limit:int=Field(default=10,ge=1,le=100)


class ChatRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    asset_id:str=Field(min_length=1,max_length=100)
    message:str=Field(min_length=1,max_length=4000)
    session_id:str|None=None


class Selection(BaseModel):
    """LLM selects grounded evidence; it cannot author measurements or maintenance commands."""
    model_config=ConfigDict(extra='forbid')
    evidence_ids:list[str]=Field(default_factory=list,max_length=24)
    hypothesis_ids:list[str]=Field(default_factory=list,max_length=4)
    alternative_ids:list[str]=Field(default_factory=list,max_length=4)
    check_ids:list[str]=Field(default_factory=list,max_length=4)


class Claim(BaseModel):
    text:str
    citations:list[str]


class Citation(BaseModel):
    id:str
    asset_id:str
    kind:str
    title:str
    text:str
    timestamp_utc:str|None=None
    document_id:str|None=None
    revision:str|None=None
    page:int|None=None
    section:str|None=None
    model_version:str|None=None


class ChatResponse(BaseModel):
    session_id:str
    message_id:str
    trace_id:str
    asset_id:str
    status:Literal['answered','abstained','refused','unavailable']
    answer:str
    observations:list[Claim]
    predictions:list[Claim]
    hypotheses:list[Claim]
    alternatives:list[Claim]
    approved_checks:list[Claim]
    maintenance_history:list[Claim]
    manual_references:list[Claim]
    missing_information:list[str]
    citations:list[Citation]
    human_review_required:Literal[True]
    uncertainty:str
    model_id:str|None
    as_of:str


class DocumentImportResponse(BaseModel):
    id:str
    title:str
    revision:str
    status:Literal['pending','quarantined']
    chunk_count:int
    quarantine_reason:str|None


class ApprovalResponse(BaseModel):
    id:str
    status:Literal['approved']
    reviewed_chunks:int


class DocumentResponse(BaseModel):
    id:str
    title:str
    revision:str
    status:str
    equipment_types:list[str]
    asset_ids:list[str]
    created_at:str
    approved_at:str|None
    approved_by:str|None
    metadata:dict
    chunks:list[dict]=Field(default_factory=list)


class CMMSResponse(BaseModel):
    imported:int
    ids:list[str]
    source_id:str


class ToolRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str=Field(min_length=1,max_length=100)
    arguments:ToolArguments


class ToolResponse(BaseModel):
    name:str
    result:dict|list|None


class SessionMessage(BaseModel):
    id:str
    role:Literal['user','assistant']
    content:str|dict
    timestamp_utc:str
    trace_id:str|None


class SessionResponse(BaseModel):
    id:str
    asset_id:str
    messages:list[SessionMessage]


class FeedbackRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    message_id:str
    rating:Literal[-1,1]
    comment:str=Field(default='',max_length=1000)


class InvestigationRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    asset_id:str
    session_id:str
    notes:str=Field(default='',max_length=5000)


class IDResponse(BaseModel):
    id:str
    status:str


class RevokeRequest(BaseModel):
    reason:str=Field(min_length=10,max_length=500)


class SessionSummary(BaseModel):
    id:str
    asset_id:str
    created_at:str


class UserSummary(BaseModel):
    id:str
    username:str
    role:str
    enabled:int
