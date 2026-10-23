"""Explicitly simulated planning resources, never automatically approved work."""
import os
from datetime import timedelta
from backend.core.database import transaction,utcnow,iso
from backend.scheduling.contracts import PlanningConfig
from backend.scheduling.service import save_config


def seed_planning():
    if os.getenv('MAINTENANCEAI_MODE','demo')!='demo':return
    with transaction() as conn:
        if not conn.execute("SELECT 1 FROM users WHERE id='admin'").fetchone():return
        if conn.execute("SELECT 1 FROM planning_configs WHERE plant_id='demo-plant'").fetchone():return
        assets=[r['id'] for r in conn.execute("SELECT id FROM assets WHERE plant_id='demo-plant'")]
    start=(utcnow()+timedelta(days=1)).replace(hour=8,minute=0,second=0,microsecond=0);end=start+timedelta(hours=8)
    period=dict(start_utc=iso(start),end_utc=iso(end))
    config=dict(technicians=[dict(id='demo-mechanical',name='Simulated mechanical technician',skills=['mechanical'],availability=[period]),
                             dict(id='demo-electrical',name='Simulated electrical technician',skills=['electrical','mechanical'],availability=[period])],
                parts=[dict(id='demo-inspection-kit',name='Simulated inspection consumables',quantity=10,available_at=iso(start))],
                equipment_windows=[dict(**period,id='demo-window-'+a,asset_id=a,disruption_cost_per_minute=1) for a in assets],
                blackouts=[dict(asset_id='DEMO-CONVEYOR-001',start_utc=iso(start+timedelta(hours=3)),end_utc=iso(start+timedelta(hours=4)),reason='Simulated production restriction')],
                reason='Initial simulated resources; verify real calendars and parts separately.',simulation_only=True)
    save_config(PlanningConfig.model_validate(config),dict(id='admin',plant_id='demo-plant',role='administrator'))
