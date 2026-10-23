from fastapi import APIRouter,Depends
from backend.core.database import transaction
from backend.core.security import principal,require_roles,authorize_asset,allowed_assets
from backend.risk.contracts import Criticality,RiskPolicy,EvaluateRequest,ConfigResponse,RiskResponse
from backend.risk.service import config_row,save_profile,save_policy,evaluate,latest

router=APIRouter(prefix='/api',tags=['maintenance risk'])

@router.get('/assets/{asset_id}/criticality',response_model=ConfigResponse)
def profile(asset_id:str,user=Depends(principal)):
    with transaction() as conn:
        authorize_asset(conn,user,asset_id)
        return config_row(conn,'asset_profiles','asset_id',asset_id)

@router.put('/assets/{asset_id}/criticality',response_model=ConfigResponse)
def update_profile(asset_id:str,body:Criticality,user=Depends(require_roles('administrator','reliability_engineer','maintenance_manager'))):
    return save_profile(asset_id,body,user)

@router.get('/risks/policy',response_model=ConfigResponse)
def policy(user=Depends(principal)):
    with transaction() as conn:return config_row(conn,'risk_policies','plant_id',user['plant_id'])

@router.put('/risks/policy',response_model=ConfigResponse)
def update_policy(body:RiskPolicy,user=Depends(require_roles('administrator','reliability_engineer','maintenance_manager'))):
    return save_policy(body,user)

@router.post('/risks/evaluate',response_model=list[RiskResponse])
def evaluations(body:EvaluateRequest,user=Depends(require_roles('administrator','engineer','reliability_engineer','maintenance_manager'))):
    return evaluate(body.asset_ids,user)

@router.get('/assets/{asset_id}/risk',response_model=RiskResponse|None)
def asset_risk(asset_id:str,user=Depends(principal)):
    with transaction() as conn:return latest(conn,user,asset_id)

@router.get('/risks',response_model=list[RiskResponse])
def risks(user=Depends(principal)):
    with transaction() as conn:
        rows=[r for a in allowed_assets(conn,user) if (r:=latest(conn,user,a['id'])) is not None]
        return sorted(rows,key=lambda r:(r['rank'],-(r['score'] or 0),r['asset_id']))
