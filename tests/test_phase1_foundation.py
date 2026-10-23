from datetime import timedelta
import asyncio,csv,io,json
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.core.database import migrate,transaction,utcnow,iso
from backend.core.demo import seed_demo
from backend.core.security import hash_password
from backend.ingestion.adapters import OPCUAReadOnlyAdapter,HistorianReadOnlyAdapter,SimulatedAdapter,ParquetAdapter


@pytest.fixture
def foundation(tmp_path,monkeypatch):
    monkeypatch.setenv('MAINTENANCEAI_DB',str(tmp_path/'app.db'))
    monkeypatch.setenv('MAINTENANCEAI_MODE','demo')
    migrate();seed_demo()
    with transaction() as conn:
        conn.execute('DELETE FROM telemetry');conn.execute('DELETE FROM ingestion_events')
    client=TestClient(app,raise_server_exceptions=False)
    token=client.post('/api/auth/login',json={'username':'admin','password':'admin-demo-password'}).json()['access_token']
    client.headers['Authorization']='Bearer '+token
    return client


def record(**changes):
    row=dict(asset_id='DEMO-CONVEYOR-001',tag_id='temperature',timestamp_utc=iso(),value=70,unit='degC',quality='good',source='demo-import',operating_state='running')
    row.update(changes);return row


def ingest(client,*rows):return client.post('/api/ingestion/demo-import/records',json={'records':list(rows)})


def test_authentication_and_revocation(foundation):
    c=foundation
    assert len(c.get('/api/assets').json())==15
    assert c.get('/api/assets/DEMO-CONVEYOR-001').status_code==200
    assert c.post('/api/auth/logout').status_code==200
    assert c.get('/api/assets').status_code==401
    assert c.post('/api/auth/login',json={'username':'admin','password':'wrong'}).status_code==401
    with transaction() as conn:assert conn.execute("SELECT COUNT(*) FROM audit_events WHERE action='login' AND outcome='denied'").fetchone()[0]==1


def test_role_permissions(foundation):
    r=foundation.post('/api/auth/login',json={'username':'engineer','password':'engineer-demo-password'})
    foundation.headers['Authorization']='Bearer '+r.json()['access_token']
    assert foundation.get('/api/assets').status_code==200
    assert ingest(foundation,record()).status_code==403


def test_asset_and_plant_isolation(foundation):
    with transaction() as conn:
        conn.execute("DELETE FROM asset_grants WHERE user_id='admin' AND asset_id='PLA-FAN-001'")
        conn.execute("INSERT INTO plants VALUES('other','Other','simulation')")
        conn.execute("INSERT INTO assets VALUES('foreign','other','Foreign','pump','area','running',1,1)")
        conn.execute("INSERT INTO asset_grants VALUES('admin','foreign')")
    assert foundation.get('/api/assets/PLA-FAN-001/telemetry').status_code==404
    assert foundation.get('/api/assets/foreign/telemetry').status_code==404
    ids=[a['id'] for a in foundation.get('/api/assets').json()]
    assert 'foreign' not in ids and 'PLA-FAN-001' not in ids


def test_source_permissions_apply_to_reads_and_imports(foundation):
    assert ingest(foundation,record()).status_code==200
    with transaction() as conn:conn.execute("DELETE FROM source_grants WHERE user_id='admin' AND source_id='demo-import'")
    assert ingest(foundation,record()).status_code==404
    data=foundation.get('/api/assets/DEMO-CONVEYOR-001/telemetry').json()
    assert data['history']==[] and all(x['observation'] is None for x in data['latest'])


def test_atomic_cross_asset_batch(foundation):
    r=ingest(foundation,record(),record(asset_id='unauthorized'))
    assert r.status_code==404
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0]==0


def test_order_duplicate_bounds_units_and_missing_timestamp(foundation):
    now=utcnow()
    assert ingest(foundation,record(timestamp_utc=iso(now))).json()['results'][0]['status']=='accepted'
    for changes in [dict(timestamp_utc=iso(now)),dict(timestamp_utc=iso(now-timedelta(seconds=1))),dict(value=1000),dict(unit='K'),dict(timestamp_utc='2026-10-20T00:00:00'),dict(timestamp_utc=None),dict(tag_id='unmapped'),dict(value='NaN')]:
        r=ingest(foundation,record(**changes))
        assert r.status_code==200,r.text
        assert r.json()['results'][0]['status']=='rejected',changes
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0]==1


def test_stale_quality_and_operating_state(foundation):
    now=utcnow()
    r=ingest(foundation,record(timestamp_utc=iso(now-timedelta(seconds=60))))
    assert r.json()['results'][0]['reason']=='Stale observation'
    data=foundation.get('/api/assets/DEMO-CONVEYOR-001/telemetry').json()
    assert next(x for x in data['latest'] if x['sensor']['tag_id']=='temperature')['effective_quality']=='stale'
    r=ingest(foundation,record(quality='bad'))
    assert r.json()['results'][0]['status']=='flagged'
    r=ingest(foundation,record(operating_state='stopped'))
    assert 'Operating state changed' in r.json()['results'][0]['reason']


def test_csv_import_and_persistence(foundation):
    row=record();body=io.StringIO();writer=csv.DictWriter(body,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    r=foundation.post('/api/ingestion/demo-import/file',files={'file':('history.csv',body.getvalue(),'text/csv')})
    assert r.status_code==200,r.text
    assert r.json()['results'][0]['status']=='accepted'
    # A new database connection sees the committed record.
    with transaction() as conn:assert conn.execute('SELECT value FROM telemetry').fetchone()[0]==70
    assert foundation.get('/api/assets/DEMO-CONVEYOR-001/telemetry').json()['history'][0]['source']=='demo-import'


def test_migrations_idempotent(foundation):
    with transaction() as conn:before=[tuple(r) for r in conn.execute('SELECT * FROM schema_versions ORDER BY version')]
    migrate();migrate()
    with transaction() as conn:
        assert [tuple(r) for r in conn.execute('SELECT * FROM schema_versions ORDER BY version')]==before
        assert conn.execute('SELECT COUNT(*) FROM assets').fetchone()[0]==15


def test_opcua_read_only_contract_and_approval():
    class Node:
        async def read_data_value(self):return SimpleNamespace(SourceTimestamp=utcnow(),Value=SimpleNamespace(Value=42),StatusCode=SimpleNamespace(is_good=lambda:True))
        async def write_value(self,*args):pytest.fail('Industrial writes must never be called')
    class Gateway:
        def get_node(self,node):assert node=='ns=2;i=1';return Node()
    args=(Gateway(),[dict(node_id='ns=2;i=1',asset_id='A',tag_id='temperature',unit='degC')],'gateway')
    with pytest.raises(ValueError,match='approval'):OPCUAReadOnlyAdapter(*args)
    rows=asyncio.run(OPCUAReadOnlyAdapter(*args,approved=True).records())
    assert rows[0]['value']==42 and rows[0]['quality']=='good'


def test_historian_contract_rejects_unmapped_tags():
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'observations':[{'tag_id':'unapproved'}]}
    class Gateway:
        def get(self,path):assert path=='/telemetry/read';return Response()
    with pytest.raises(ValueError,match='approval'):HistorianReadOnlyAdapter(Gateway(),'/telemetry/read',{},'gateway')
    with pytest.raises(ValueError,match='Unapproved'):HistorianReadOnlyAdapter(Gateway(),'/telemetry/read',{},'gateway',True).records()


def test_simulation_is_reproducible_and_labelled():
    now=utcnow();assets=[{'id':'DEMO-CONVEYOR-001'}]
    a=list(SimulatedAdapter(assets,now=now).records());b=list(SimulatedAdapter(assets,now=now).records())
    assert a==b and all(r['source']=='demo-simulator' for r in a)


def test_parquet_roundtrip(foundation):
    import pandas as pd
    buffer=io.BytesIO();pd.DataFrame([record()]).to_parquet(buffer,index=False)
    r=foundation.post('/api/ingestion/demo-import/file',files={'file':('historical.parquet',buffer.getvalue(),'application/octet-stream')})
    assert r.status_code==200,r.text
    assert r.json()['results'][0]['status']=='accepted'


def test_csrf_cookie_mutation_rejected(foundation):
    foundation.headers.pop('Authorization')
    assert foundation.post('/api/auth/logout',headers={'Origin':'https://untrusted.example'}).status_code==403
    assert foundation.post('/api/auth/logout',headers={'Origin':'http://localhost:3000'}).status_code==200


def test_no_industrial_write_path(foundation):
    assert foundation.post('/api/ingestion/demo-simulator/records',json={'records':[record(source='demo-simulator')]}).status_code==422
    for route in ['/api/plc/write','/api/scada/command','/api/actuators/start']:
        assert foundation.post(route,json={'value':1}).status_code==404


def test_scoped_websocket_rechecks_session(foundation):
    from starlette.websockets import WebSocketDisconnect
    with foundation.websocket_connect('/api/ws/telemetry',headers={'Origin':'http://localhost:3000'}) as ws:
        payload=ws.receive_json()
        assert payload['type']=='telemetry_snapshot'
        assert len(payload['assets'])==15
        with transaction() as conn:conn.execute("DELETE FROM sessions WHERE user_id='admin'")
        with pytest.raises(WebSocketDisconnect) as closed:ws.receive_json()
        assert closed.value.code==1008
    with pytest.raises(WebSocketDisconnect):
        with foundation.websocket_connect('/api/ws/telemetry',headers={'Origin':'https://untrusted.example'}):pass


def test_gateway_failure_is_audited_without_inventing_data(foundation):
    from backend.ingestion.gateway import collect
    with transaction() as conn:
        conn.execute("INSERT INTO sources VALUES('gateway','demo-plant','Test','historian',1,'configured',NULL)")
        conn.execute("INSERT INTO source_grants VALUES('admin','gateway')")
    class Broken:
        def records(self):raise ConnectionError('secret-value-in-url')
    result=asyncio.run(collect(Broken(),dict(id='admin',plant_id='demo-plant'),'gateway'))
    assert result['status']=='unavailable' and 'secret-value' not in str(result)
    with transaction() as conn:
        assert conn.execute("SELECT status FROM sources WHERE id='gateway'").fetchone()[0]=='disconnected'
        assert conn.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0]==0
        assert 'secret-value' not in str([tuple(r) for r in conn.execute('SELECT * FROM audit_events')])
