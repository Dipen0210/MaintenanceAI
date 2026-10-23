"""Persisted demo control; never starts or stops physical equipment."""
import os
from typing import Literal
from fastapi import APIRouter,Depends,HTTPException
from pydantic import StrictBool
from backend.core.database import transaction,iso,audit
from backend.core.security import principal,authorize_source,allowed_assets
from backend.scheduling.contracts import Strict

router=APIRouter(prefix='/api/simulation',tags=['demo simulation'])


def enabled(conn):
    row=conn.execute("SELECT enabled FROM simulation_controls WHERE plant_id='demo-plant'").fetchone()
    return bool(row['enabled']) if row else os.getenv('MAINTENANCEAI_SIMULATION_ENABLED','1')=='1'


def scenario(conn):
    row=conn.execute("SELECT scenario FROM simulation_controls WHERE plant_id='demo-plant'").fetchone()
    return row['scenario'] if row else 'mixed'


def status(conn,user):
    available=os.getenv('MAINTENANCEAI_MODE','demo')=='demo' and user['plant_id']=='demo-plant'
    can_control=False
    if available:
        source=conn.execute("SELECT 1 FROM source_grants WHERE source_id='demo-simulator' AND user_id=?",(user['id'],)).fetchone()
        targets={r[0] for r in conn.execute("SELECT DISTINCT s.asset_id FROM sensors s JOIN assets a ON s.asset_id=a.id WHERE s.source_id='demo-simulator' AND a.plant_id=?",(user['plant_id'],))}
        can_control=bool(source and targets and targets<={a['id'] for a in allowed_assets(conn,user)})
    return dict(available=available,enabled=available and enabled(conn),can_control=can_control,interval_seconds=10,scenario=scenario(conn) if available else None)


class Toggle(Strict):
    enabled:StrictBool
    scenario:Literal['mixed','normal','warning','critical']|None=None


@router.get('')
def current(user=Depends(principal)):
    with transaction() as conn:return status(conn,user)


@router.put('')
def toggle(body:Toggle,user=Depends(principal)):
    with transaction() as conn:
        conn.execute('BEGIN IMMEDIATE')
        state=status(conn,user)
        if not state['available']:raise HTTPException(403,'Simulation controls are available only in the local demo facility')
        if not state['can_control']:raise HTTPException(403,'Access to all simulated equipment and its source is required')
        authorize_source(conn,user,'demo-simulator')
        selected=body.scenario or scenario(conn)
        conn.execute('INSERT INTO simulation_controls(plant_id,enabled,updated_at,actor_id,scenario) VALUES(?,?,?,?,?) ON CONFLICT(plant_id) DO UPDATE SET enabled=excluded.enabled,updated_at=excluded.updated_at,actor_id=excluded.actor_id,scenario=excluded.scenario',(user['plant_id'],int(body.enabled),iso(),user['id'],selected))
        conn.execute("UPDATE sources SET status=? WHERE id='demo-simulator'",('starting' if body.enabled else 'paused',))
        audit(conn,user,'set_demo_simulation','demo-simulator',('running' if body.enabled else 'paused')+'; scenario='+selected)
    if body.enabled:
        from backend.core.demo import simulate_tick
        simulate_tick()
    with transaction() as conn:return status(conn,user)
