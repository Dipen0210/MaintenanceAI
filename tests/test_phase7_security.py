"""Local registry authorization and adversarial approval boundaries."""
from datetime import datetime,timedelta
from copy import deepcopy
import json
from tests.test_phase1_foundation import foundation,record
from tests.test_phase6_scheduling import scenario
from tests.test_phase2_ml import ml,store,prediction
from backend.core.database import transaction
from backend.core.security import hash_password
from backend.risk.service import DEFAULT_PROFILE
from backend.scheduling.solver import solve,validate


def account(**changes):
    return dict(username='scoped-engineer',password='test-only-password-9876',role='engineer',enabled=True,asset_ids=['DEMO-CONVEYOR-001'],source_ids=['demo-import'],document_ids=[],reason='Create an explicitly scoped regression-test identity.',**changes)


def login(c,name,password):
    r=c.post('/api/auth/login',json={'username':name,'password':password});assert r.status_code==200,r.text
    return {'Authorization':'Bearer '+r.json()['access_token']}


def test_identity_grants_login_and_revocation(foundation):
    c=foundation;request=account();r=c.post('/api/admin/identities',json=request);assert r.status_code==200,r.text
    identity=r.json();assert 'password' not in r.text and 'password_hash' not in r.text
    auth=login(c,request['username'],request['password'])
    assert [a['id'] for a in c.get('/api/assets',headers=auth).json()]==['DEMO-CONVEYOR-001']
    assert c.get('/api/assets/PLA-PUMP-001/telemetry',headers=auth).status_code==404
    assert c.get('/api/admin/identities',headers=auth).status_code==403
    assert c.post('/api/admin/sources',json={'name':'Denied source','reason':'No administrator authorization'},headers=auth).status_code==403
    body={k:request[k] for k in ['role','enabled','asset_ids','source_ids','document_ids','reason']};body.update(enabled=False,reason='Disable test identity and revoke all existing sessions.')
    r=c.put('/api/admin/identities/'+identity['id']+'/access',json=body);assert r.status_code==200,r.text
    assert c.get('/api/assets',headers=auth).status_code==401
    assert c.post('/api/auth/login',json={'username':request['username'],'password':request['password']}).status_code==401
    with transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM sessions WHERE user_id=?',(identity['id'],)).fetchone()[0]==0
        audit=' '.join(r[0] for r in conn.execute('SELECT outcome FROM audit_events'))
        assert request['password'] not in audit and 'sessions revoked' in audit


def test_admin_cannot_change_self_or_grant_foreign_resources(foundation):
    c=foundation;body={k:v for k,v in account().items() if k not in ['username','password']}
    assert c.put('/api/admin/identities/admin/access',json=body).status_code==409
    with transaction() as conn:
        conn.execute("INSERT INTO plants VALUES('foreign','Foreign plant','simulation')")
        conn.execute("INSERT INTO assets VALUES('foreign-asset','foreign','Foreign asset','pump','area','unknown',1,1)")
        conn.execute("INSERT INTO sources VALUES('foreign-source','foreign','Foreign source','file',1,'configured',NULL)")
        conn.execute('INSERT INTO users VALUES(?,?,?,?,?,1)',('foreign-user','foreign-user',hash_password('foreign-test-password'),'engineer','foreign'))
        # Deliberately malformed grants must not defeat plant isolation.
        conn.execute("INSERT INTO asset_grants VALUES('admin','foreign-asset')")
        conn.execute("INSERT INTO source_grants VALUES('admin','foreign-source')")
    assert c.put('/api/admin/identities/foreign-user/access',json=body).status_code==404
    assert c.post('/api/admin/identities',json={**account(),'asset_ids':['foreign-asset']}).status_code==404
    assert c.post('/api/admin/identities',json={**account(),'source_ids':['foreign-source']}).status_code==404
    assert 'foreign-user' not in c.get('/api/admin/identities').text
    assert 'foreign-asset' not in c.get('/api/admin/identities').text
    assert 'foreign-source' not in c.get('/api/admin/identities').text
    assert 'foreign-asset' not in c.get('/api/admin/mappings').text
    with transaction() as conn:conn.execute("DELETE FROM asset_grants WHERE user_id='admin' AND asset_id='PLA-PUMP-001'")
    assert c.get('/api/admin/identities').status_code==403
    assert c.post('/api/admin/identities',json=account()).status_code==403


def registration(source):
    return dict(id='REGISTERED-PUMP',name='Registered test pump',equipment_type='pump',area='Test area',reason='Explicit sensor contract and site-limit configuration for the test.',
        sensors=[dict(tag_id=tag,unit=unit,lower_bound=lo,upper_bound=hi,stale_after_seconds=30,source_id=source) for tag,unit,lo,hi in [('temperature','degC',-40,180),('vibration_rms','mm/s',0,100),('current','A',0,500)]],criticality=deepcopy(DEFAULT_PROFILE))


def test_registry_creates_no_data_models_or_industrial_connection(foundation):
    c=foundation;body={'name':'New archive import','reason':'Register a read-only historical file import for this test.'}
    r=c.post('/api/admin/sources',json={**body,'kind':'opcua','url':'http://example.com'});assert r.status_code==422
    r=c.post('/api/admin/sources',json=body);assert r.status_code==200,r.text
    sid=r.json()['id'];request=registration(sid);r=c.post('/api/admin/assets',json=request);assert r.status_code==200,r.text
    aid=r.json()['id'];assert c.get(f'/api/assets/{aid}').json()['operating_state']=='unknown'
    with transaction() as conn:
        source=conn.execute('SELECT * FROM sources WHERE id=?',(sid,)).fetchone();assert source['kind']=='file' and source['read_only']==1
        for table in ['telemetry','model_bindings','predictions']:assert conn.execute(f'SELECT COUNT(*) FROM {table} WHERE asset_id=?',(aid,)).fetchone()[0]==0
        assert json.loads(conn.execute('SELECT config FROM asset_profiles WHERE asset_id=?',(aid,)).fetchone()[0])['limits']['temperature']['critical']==95
    telemetry=record(asset_id=aid,source=sid)
    r=c.post(f'/api/ingestion/{sid}/records',json={'records':[telemetry]});assert r.status_code==200 and r.json()['results'][0]['status']=='accepted',r.text
    assert c.get(f'/api/assets/{aid}/telemetry').json()['latest'][1]['observation']['source']==sid
    assert c.post('/api/admin/assets',json=request).status_code==409
    assert c.put(f'/api/admin/assets/{aid}',json={'name':'Renamed test pump','area':'Revised test area','reason':'Documented metadata correction; equipment state is unchanged.'}).status_code==200
    assert c.get(f'/api/assets/{aid}').json()['operating_state']=='running'
    assert any(e['action']=='register_asset' for e in c.get('/api/admin/audit').json())


def test_invalid_registry_changes_are_atomic(foundation):
    c=foundation;body=registration('demo-import');body['criticality']['limits']['temperature']['unit']='K'
    assert c.post('/api/admin/assets',json=body).status_code==422
    assert c.get('/api/assets/REGISTERED-PUMP').status_code==404
    r=c.post('/api/admin/identities',json={**account(),'password':'pwdsecret9'});assert r.status_code==422 and 'pwdsecret9' not in r.text
    body=registration('demo-import');body['sensors'][1]['tag_id']='temperature'
    assert c.post('/api/admin/assets',json=body).status_code==422


def test_site_metadata_configuration_does_not_change_industrial_mode(foundation):
    c=foundation;body=dict(name='Documented demo facility',location='Offline simulated facility',display_timezone='America/Chicago',reason='Record the explicit demonstration location and display timezone.')
    r=c.put('/api/admin/plant',json=body);assert r.status_code==200,r.text
    assert r.json()['mode']=='simulation' and r.json()['configuration']['display_timezone']=='America/Chicago'
    assert c.put('/api/admin/plant',json={**body,'mode':'physical'}).status_code==422
    assert c.put('/api/admin/plant',json={**body,'display_timezone':'Invalid/Zone'}).status_code==422
    assert c.get('/api/admin/plant').json()['location']=='Offline simulated facility'


def test_simulator_does_not_overwrite_file_only_registered_assets(foundation):
    from backend.core.demo import simulate_tick
    c=foundation;r=c.post('/api/admin/assets',json=registration('demo-import'));assert r.status_code==200,r.text
    simulate_tick()
    with transaction() as conn:assert conn.execute("SELECT COUNT(*) FROM telemetry WHERE asset_id='REGISTERED-PUMP'").fetchone()[0]==0


def test_independent_validator_rejects_subminute_assignment_tampering():
    s=scenario();s['orders']=s['orders'][:1];r=solve(s);a=r['assignments'][0]
    # Previously rounding could hide 61 actual minutes behind a 60-minute task.
    a['start_utc']=(datetime.fromisoformat(a['start_utc'])+timedelta(seconds=30)).isoformat()
    a['end_utc']=(datetime.fromisoformat(a['end_utc'])+timedelta(seconds=90)).isoformat()
    assert any('one-minute' in issue for issue in validate([a],s))


def test_synthetic_incident_alarm_state_and_source_permissions(ml):
    c=ml;asset='DEMO-CONVEYOR-001'
    r=c.post(f'/api/assets/{asset}/demo-prediction',json={});assert r.status_code==200,r.text
    alarms=c.get(f'/api/assets/{asset}/alarms').json();assert len(alarms)==1 and alarms[0]['state']=='active'
    assert 'Synthetic scenario' in alarms[0]['message'] and 'physical fault' in alarms[0]['message']
    r=c.post(f'/api/assets/{asset}/demo-prediction',json={'state':'normal'});assert r.status_code==200,r.text
    assert all(a['state']=='cleared' for a in c.get(f'/api/assets/{asset}/alarms').json())
    reference=c.get('/api/evidence/alarm:'+alarms[0]['id'],params={'asset_id':asset});assert reference.status_code==200
    assert 'Recorded alarm state: cleared' in reference.json()['text']
    with transaction() as conn:conn.execute("DELETE FROM source_grants WHERE user_id='admin' AND source_id='demo-simulator'")
    assert c.get(f'/api/assets/{asset}/alarms').json()==[]
    assert c.get('/api/evidence/alarm:'+alarms[0]['id'],params={'asset_id':asset}).status_code==404


def test_model_activation_and_history_are_plant_scoped(ml):
    c=ml;versions=[v for v in c.get('/api/ml/models').json() if v['family']=='conveyor-synthetic']
    old=next(v['id'] for v in versions if v['active']);new=next(v['id'] for v in versions if not v['active'])
    with transaction() as conn:
        conn.execute("INSERT INTO plants VALUES('other-model-plant','Other model plant','simulation')")
        conn.execute('INSERT INTO users VALUES(?,?,?,?,?,1)',('other-reliability','other-reliability',hash_password('other-model-test-password'),'reliability_engineer','other-model-plant'))
    auth=login(c,'other-reliability','other-model-test-password')
    r=c.post(f'/api/ml/models/{new}/activate',headers=auth,json={'reason':'Other plant independently evaluates the registered challenger.'});assert r.status_code==200,r.text
    assert c.get('/api/ml/activations').json()==[]
    assert len(c.get('/api/ml/activations',headers=auth).json())==1
    assert prediction(c,store(c))['model_id']==old
    assert next(v['id'] for v in c.get('/api/ml/models',headers=auth).json() if v['family']=='conveyor-synthetic' and v['active'])==new
    assert c.post(f'/api/ml/models/{new}/activate',json={'reason':'This plant explicitly approves its own challenger evaluation.'}).status_code==200
    assert len(c.get('/api/ml/activations').json())==1
    from backend.ml.registry import initialize_registry
    initialize_registry();assert prediction(c,store(c))['model_id']==new
    with transaction() as conn:conn.execute("DELETE FROM asset_grants WHERE user_id='admin' AND asset_id='DEMO-CONVEYOR-001'")
    assert c.post(f'/api/ml/models/{old}/activate',json={'reason':'Cannot change selection for an unauthorized bound asset.'}).status_code==404


def test_existing_model_selection_migration_preserves_choices(tmp_path,monkeypatch):
    import shutil
    from backend.core import database
    from backend.core.demo import seed_demo
    from backend.ml.registry import initialize_registry,active_id
    original=database.ROOT;old_root=tmp_path/'previous';migrations=old_root/'infrastructure/migrations';migrations.mkdir(parents=True)
    for source in (original/'infrastructure/migrations').glob('*.sql'):
        if source.name<'007':shutil.copyfile(source,migrations/source.name)
    monkeypatch.setenv('MAINTENANCEAI_DB',str(tmp_path/'migration.db'));monkeypatch.setenv('MAINTENANCEAI_MODE','demo')
    monkeypatch.setattr(database,'ROOT',old_root);database.migrate();seed_demo();initialize_registry()
    with transaction() as conn:
        next_id=conn.execute("SELECT id FROM model_versions WHERE family='conveyor-synthetic' AND active=0").fetchone()[0]
        conn.execute("UPDATE model_versions SET active=0 WHERE family='conveyor-synthetic'")
        conn.execute('UPDATE model_versions SET active=1 WHERE id=?',(next_id,))
        conn.execute('INSERT INTO model_activations VALUES(?,?,?,?,?,?,?)',('legacy-activation','conveyor-synthetic',None,next_id,'admin',database.iso(),'Prior reviewed choice'))
    monkeypatch.setattr(database,'ROOT',original);database.migrate();database.migrate()
    with transaction() as conn:
        assert active_id(conn,'conveyor-synthetic','demo-plant')==next_id
        assert conn.execute("SELECT plant_id FROM model_activations WHERE id='legacy-activation'").fetchone()[0]=='demo-plant'
