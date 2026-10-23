import json,os
from datetime import datetime,timedelta
from fastapi import HTTPException
from backend.core.database import transaction,identifier,iso,utcnow,audit
from backend.core.security import authorize_asset,authorize_source,allowed_assets
from backend.ingestion.service import latest_telemetry
from backend.ml.service import history
from backend.risk.contracts import Criticality,RiskPolicy
from backend.risk.engine import assess

DEFAULT_POLICY=dict(name='Demo risk policy v1',safety_weight=.4,production_weight=.3,maintenance_weight=.15,damage_weight=.15,
                    critical_score=85,high_score=50,planned_score=25,prediction_max_age_seconds=60,reason='Initial explicitly simulated site policy')
DEFAULT_PROFILE=dict(safety=2,production=3,maintenance=2,damage=2,dependencies=[],
                     limits={'temperature':dict(unit='degC',warning=70,critical=95),'vibration_rms':dict(unit='mm/s',warning=4.5,critical=9),
                             'current':dict(unit='A',warning=25,critical=40)},required_sensors=['temperature','vibration_rms','current'],
                     escalation='Contact the demo maintenance manager for review under the site escalation process. No equipment operation is authorized.',
                     reason='Initial explicitly simulated operating limits; not OEM specifications')


def seed_risk():
    if os.getenv('MAINTENANCEAI_MODE','demo')!='demo':return
    with transaction() as conn:
        if not conn.execute("SELECT 1 FROM users WHERE id='admin'").fetchone():return
        if not conn.execute("SELECT 1 FROM risk_policies WHERE plant_id='demo-plant'").fetchone():
            conn.execute('INSERT INTO risk_policies VALUES(?,?,?,?,?,1)',(identifier(),'demo-plant',json.dumps(DEFAULT_POLICY),'admin',iso()))
        for a in conn.execute("SELECT * FROM assets WHERE plant_id='demo-plant'").fetchall():
            if conn.execute('SELECT 1 FROM asset_profiles WHERE asset_id=?',(a['id'],)).fetchone():continue
            profile=dict(DEFAULT_PROFILE)
            if a['id']=='DEMO-CONVEYOR-001':profile.update(safety=3,production=5,maintenance=4,damage=3,dependencies=['PLA-PUMP-001'])
            conn.execute('INSERT INTO asset_profiles VALUES(?,?,?,?,?,1)',(identifier(),a['id'],json.dumps(profile),'admin',iso()))
            conn.execute('UPDATE assets SET safety_criticality=?,production_criticality=? WHERE id=?',(profile['safety'],profile['production'],a['id']))


def config_row(conn,table,key,value):
    row=conn.execute(f'SELECT id,config,timestamp_utc FROM {table} WHERE {key}=? AND active=1',(value,)).fetchone()
    if not row:raise HTTPException(409,'Risk policy or asset criticality has not been configured')
    return dict(id=row['id'],config=json.loads(row['config']),timestamp_utc=row['timestamp_utc'])


def save_profile(asset_id,body,user):
    with transaction() as conn:
        authorize_asset(conn,user,asset_id)
        sensors={r['tag_id']:dict(r) for r in conn.execute('SELECT * FROM sensors WHERE asset_id=?',(asset_id,))}
        for tag,limit in body.limits.items():
            if tag not in sensors or limit.unit!=sensors[tag]['unit'] or not sensors[tag]['lower_bound']<=limit.warning<limit.critical<=sensors[tag]['upper_bound']:
                raise HTTPException(422,'Operating limits must match mapped sensor units and physical boundaries')
        for dep in body.dependencies:
            if dep==asset_id:raise HTTPException(422,'Asset cannot depend on itself')
            authorize_asset(conn,user,dep)
        # Detect cycles against existing active dependency graph.
        graph={r['asset_id']:json.loads(r['config'])['dependencies'] for r in conn.execute('SELECT asset_id,config FROM asset_profiles WHERE active=1')}
        graph[asset_id]=body.dependencies
        def visit(node,path):
            if node in path:raise HTTPException(422,'Cyclic production dependency')
            for target in graph.get(node,[]):visit(target,path|{node})
        visit(asset_id,set())
        rid=identifier();stamp=iso();config=body.model_dump()
        conn.execute('UPDATE asset_profiles SET active=0 WHERE asset_id=?',(asset_id,))
        conn.execute('INSERT INTO asset_profiles VALUES(?,?,?,?,?,1)',(rid,asset_id,json.dumps(config),user['id'],stamp))
        conn.execute('UPDATE assets SET safety_criticality=?,production_criticality=? WHERE id=?',(body.safety,body.production,asset_id))
        audit(conn,user,'update_criticality',asset_id,body.reason)
        return dict(id=rid,config=config,timestamp_utc=stamp)


def save_policy(body,user):
    with transaction() as conn:
        rid=identifier();stamp=iso();config=body.model_dump()
        conn.execute('UPDATE risk_policies SET active=0 WHERE plant_id=?',(user['plant_id'],))
        conn.execute('INSERT INTO risk_policies VALUES(?,?,?,?,?,1)',(rid,user['plant_id'],json.dumps(config),user['id'],stamp))
        audit(conn,user,'update_risk_policy',rid,body.reason)
        return dict(id=rid,config=config,timestamp_utc=stamp)


def evaluate(asset_ids,user):
    now=utcnow();results=[]
    with transaction() as conn:
        policy=config_row(conn,'risk_policies','plant_id',user['plant_id'])
        for aid in dict.fromkeys(asset_ids):
            authorize_asset(conn,user,aid);profile=config_row(conn,'asset_profiles','asset_id',aid)
            predictions=history(conn,user,aid,50)
            bindings={(r['family'],r['dataset_id']) for r in conn.execute('SELECT family,dataset_id FROM model_bindings WHERE asset_id=?',(aid,))}
            predictions=[p for p in predictions if (p['family'],p['dataset_id']) in bindings]
            for p in predictions:
                if p['input_id']:
                    row=conn.execute('SELECT source_id FROM signal_inputs WHERE id=?',(p['input_id'],)).fetchone()
                    if row:p['source_id']=row['source_id']
            readings=latest_telemetry(conn,user,aid,now)
            result=assess(profile['config'],policy['config'],readings,predictions,now)
            deadlines=[now+timedelta(seconds=30)]
            deadlines.extend(datetime.fromisoformat(r['observation']['timestamp_utc'])+timedelta(seconds=r['sensor']['stale_after_seconds']) for r in readings if r['observation'] and r['effective_quality']=='good')
            for p in predictions:
                if any(e['id']==p['id'] for e in result['evidence']):deadlines.append(datetime.fromisoformat(p['signal_timestamp'])+timedelta(seconds=policy['config']['prediction_max_age_seconds']))
            result.update(id=identifier(),asset_id=aid,policy_id=policy['id'],profile_id=profile['id'],timestamp_utc=iso(now),stale=False,expires_at=iso(min(deadlines)))
            conn.execute('INSERT INTO risk_assessments VALUES(?,?,?,?,?,?,?)',(result['id'],aid,policy['id'],profile['id'],user['id'],iso(now),json.dumps(result,allow_nan=False)))
            audit(conn,user,'evaluate_risk',result['id'],result['priority']);results.append(result)
    return sorted(results,key=lambda r:(r['rank'],-(r['score'] or 0),r['asset_id']))


def latest(conn,user,asset_id):
    authorize_asset(conn,user,asset_id)
    allowed={r['source_id'] for r in conn.execute('SELECT source_id FROM source_grants WHERE user_id=?',(user['id'],))}
    for row in conn.execute('SELECT result FROM risk_assessments WHERE asset_id=? ORDER BY timestamp_utc DESC LIMIT 100',(asset_id,)):
        result=json.loads(row['result'])
        if set(result['source_ids'])<=allowed:
            policy=config_row(conn,'risk_policies','plant_id',user['plant_id'])
            profile=config_row(conn,'asset_profiles','asset_id',asset_id)
            result['stale']=utcnow()>datetime.fromisoformat(result['expires_at']) or result['policy_id']!=policy['id'] or result['profile_id']!=profile['id']
            return result
    return None
