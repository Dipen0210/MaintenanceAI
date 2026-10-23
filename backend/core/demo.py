"""Explicitly simulated local facility, identities and source records."""
import os
from datetime import timedelta
from backend.core.database import transaction,iso,utcnow
from backend.core.security import hash_password
from backend.services.sample_data import SAMPLE_MACHINES
from backend.ingestion.adapters import SimulatedAdapter
from backend.ingestion.service import ingest


def seed_demo():
    if os.getenv('MAINTENANCEAI_MODE','demo')!='demo':return
    with transaction() as conn:
        if conn.execute('SELECT 1 FROM plants WHERE id=?',('demo-plant',)).fetchone():return
        conn.execute('INSERT INTO plants VALUES(?,?,?)',('demo-plant','Simulated rotating-equipment facility','simulation'))
        conn.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?)',('demo-simulator','demo-plant','Simulated telemetry','simulation',1,'configured',None))
        conn.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?)',('demo-import','demo-plant','Historical file import','file',1,'configured',None))
        assets=[dict(id=m['machine_id'],name=m['machine_id'],equipment_type=m['machine_type'],area=m['area']) for m in SAMPLE_MACHINES]
        assets.append(dict(id='DEMO-CONVEYOR-001',name='Demo conveyor motor',equipment_type='conveyor_motor',area='production_line_a'))
        for asset in assets:
            conn.execute('INSERT INTO assets VALUES(?,?,?,?,?,?,?,?)',(asset['id'],'demo-plant',asset['name'],asset['equipment_type'],asset['area'],'running',2,3))
            for tag,unit,lower,upper in [('temperature','degC',-40,180),('vibration_rms','mm/s',0,100),('current','A',0,500)]:
                conn.execute('INSERT INTO sensors VALUES(?,?,?,?,?,?,?,?)',(asset['id']+':'+tag,asset['id'],tag,unit,lower,upper,30,'demo-simulator'))
        for username,role in [('admin','administrator'),('engineer','engineer'),('manager','maintenance_manager'),('reliability','reliability_engineer')]:
            conn.execute('INSERT INTO users VALUES(?,?,?,?,?,?)',(username,username,hash_password(username+'-demo-password'),role,'demo-plant',1))
            for asset in assets:conn.execute('INSERT INTO asset_grants VALUES(?,?)',(username,asset['id']))
            for source in ['demo-simulator','demo-import']:conn.execute('INSERT INTO source_grants VALUES(?,?)',(username,source))
    user=dict(id='admin',plant_id='demo-plant',role='administrator')
    now=utcnow()
    # Initial trend history uses explicit synthetic observations and timestamps.
    for i in range(12):
        rows=list(SimulatedAdapter(assets,seed=42+i,now=now-timedelta(seconds=(11-i)*5)).records())
        ingest(rows,user,'demo-simulator',now=now)


def simulate_tick():
    if os.getenv('MAINTENANCEAI_MODE','demo')!='demo':return []
    from backend.routes.simulation import enabled,scenario
    with transaction() as conn:
        # Serialize against pause so no tick commits after a pause response.
        conn.execute('BEGIN IMMEDIATE')
        if not enabled(conn):return []
        assets=[dict(r) for r in conn.execute("SELECT * FROM assets WHERE plant_id=? AND id IN (SELECT asset_id FROM sensors WHERE source_id='demo-simulator')",('demo-plant',))]
        if not assets:return []
        return ingest(list(SimulatedAdapter(assets,seed=int(utcnow().timestamp())//5,scenario=scenario(conn)).records()),
                      dict(id='admin',plant_id='demo-plant',role='administrator'),'demo-simulator',connection=conn)
