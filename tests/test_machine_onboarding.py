"""Real ingestion, isolation, simulation controls and sensor-map regressions."""
import asyncio,hashlib,io,json,socket
from datetime import timedelta
import pytest
from PIL import Image
from tests.test_phase1_foundation import foundation
from tests.test_phase7_security import login
from backend.core.database import transaction,iso,utcnow,migrate
from backend.core.demo import simulate_tick
from backend.risk.service import seed_risk


def machine(aid='FACTORY-PUMP-01'):
    return dict(id=aid,name='Factory cooling pump',equipment_type='pump',area='Process line A',
        sensors=[dict(tag_id='temperature',unit='degC',lower_bound=-20,upper_bound=150,stale_after_seconds=30),dict(tag_id='vibration_rms',unit='mm/s',lower_bound=0,upper_bound=50,stale_after_seconds=30)],
        criticality=dict(safety=3,production=4,maintenance=3,damage=3,dependencies=[],limits={'temperature':dict(unit='degC',warning=70,critical=90),'vibration_rms':dict(unit='mm/s',warning=4.5,critical=9)},required_sensors=['temperature','vibration_rms'],operating_states=['running'],escalation='Contact the qualified site maintenance manager.',reason='Test fixture only; synthetic limits, not equipment specifications.'),
        reason='Test-only commissioning: reviewed synthetic sensor mappings and limits.')


def onboard(client,aid='FACTORY-PUMP-01'):
    r=client.post('/api/admin/machines',json=machine(aid));assert r.status_code==200
    assert r.headers['cache-control']=='no-store'
    return r.json()


def rows(config,aid='FACTORY-PUMP-01',temperature=45,vibration=1.2,stamp=None):
    return [dict(asset_id=aid,tag_id=tag,unit=unit,value=value,timestamp_utc=stamp or iso(),quality='good',source=config['source_id'],operating_state='running') for tag,unit,value in [('temperature','degC',temperature),('vibration_rms','mm/s',vibration)]]


def send(c,config,records=None):
    return c.post(config['endpoint'],headers={'Authorization':'Bearer '+config['token']},json={'records':records or rows(config)})


def test_onboard_awaits_actual_data_and_keeps_simulation_separate(foundation):
    c=foundation;config=onboard(c);aid='FACTORY-PUMP-01'
    asset=c.get('/api/assets/'+aid).json()
    assert asset['operating_state']=='unknown' and asset['condition']=='unknown' and asset['data_mode']=='gateway' and asset['last_observed'] is None
    assert next(s for s in c.get('/api/integrations/status').json() if s['id']==config['source_id'])['status']=='awaiting data'
    simulate_tick()
    assert c.get('/api/assets/'+aid+'/telemetry').json()['history']==[]
    with transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM model_bindings WHERE asset_id=?',(aid,)).fetchone()[0]==0
        key=conn.execute('SELECT token_hash FROM gateway_credentials WHERE source_id=?',(config['source_id'],)).fetchone()[0]
        assert key==hashlib.sha256(config['token'].encode()).hexdigest()
    response=send(c,config);assert response.status_code==200 and all(r['status']=='accepted' for r in response.json()['results'])
    asset=c.get('/api/assets/'+aid).json();assert asset['condition']=='normal' and asset['operating_state']=='running' and asset['data_mode']=='gateway'
    assert send(c,config,rows(config,temperature=75)).status_code==200
    assert c.get('/api/assets/'+aid).json()['condition']=='warning'
    assert send(c,config,rows(config,temperature=95)).status_code==200
    assert c.get('/api/assets/'+aid).json()['condition']=='critical'
    assert config['token'] not in c.get('/api/admin/audit').text
    migrate();assert c.get('/api/assets/'+aid).json()['condition']=='critical'


def test_onboarding_is_atomic_and_requires_scoped_admin(foundation):
    c=foundation;engineer=login(c,'engineer','engineer-demo-password')
    assert c.post('/api/admin/machines',json=machine(),headers=engineer).status_code==403
    body=machine();body['sensors'][1]['tag_id']='temperature'
    with transaction() as conn:before=conn.execute('SELECT COUNT(*) FROM sources').fetchone()[0]
    assert c.post('/api/admin/machines',json=body).status_code==422
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM sources').fetchone()[0]==before
    cfg=onboard(c)
    assert c.post('/api/admin/machines',json=machine()).status_code==409
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM sources').fetchone()[0]==before+1
    assert c.get('/api/assets/FACTORY-PUMP-01',headers=engineer).status_code==404
    assert c.get('/api/admin/gateways/'+cfg['source_id'],headers=engineer).status_code==403


def test_gateway_key_cannot_read_ui_or_target_other_machine(foundation):
    c=foundation;cfg=onboard(c);other=onboard(c,'OTHER-PUMP')
    assert c.get('/api/assets',headers={'Authorization':'Bearer '+cfg['token']}).status_code==401
    assert c.post(cfg['endpoint'],json={'records':rows(cfg)}).status_code==401
    assert c.post(other['endpoint'],headers={'Authorization':'Bearer '+cfg['token']},json={'records':rows(other,'OTHER-PUMP')}).status_code==401
    payload=rows(cfg)+rows(cfg,'OTHER-PUMP')
    assert send(c,cfg,payload).status_code==403
    assert c.get('/api/assets/FACTORY-PUMP-01/telemetry').json()['history']==[]
    assert c.post('/api/ingestion/demo-import/records',json={'records':[{**rows(cfg)[0],'source':'demo-import'}]}).json()['results'][0]['status']=='rejected'


@pytest.mark.parametrize('change',['revoke','rotate','expire','disable_owner','remove_grant','change_role'])
def test_gateway_access_can_be_stopped(foundation,change):
    c=foundation;cfg=onboard(c);sid=cfg['source_id']
    if change=='revoke':assert c.delete(f'/api/admin/gateways/{sid}/credential').status_code==200
    elif change=='rotate':
        new=c.post(f'/api/admin/gateways/{sid}/credential');assert new.status_code==200
        assert send(c,new.json()).status_code==200
    else:
        with transaction() as conn:
            if change=='expire':conn.execute('UPDATE gateway_credentials SET expires_at=? WHERE source_id=?',(iso(utcnow()-timedelta(days=1)),sid))
            elif change=='disable_owner':conn.execute("UPDATE users SET enabled=0 WHERE id='admin'")
            elif change=='remove_grant':conn.execute("DELETE FROM source_grants WHERE user_id='admin' AND source_id=?",(sid,))
            elif change=='change_role':conn.execute("UPDATE users SET role='engineer' WHERE id='admin'")
    assert send(c,cfg).status_code in [401,404]


def test_stale_missing_rejected_and_bad_readings_never_show_normal(foundation):
    c=foundation;cfg=onboard(c);url='/api/assets/FACTORY-PUMP-01'
    assert send(c,cfg,rows(cfg)[:1]).status_code==200
    assert c.get(url).json()['condition']=='unknown'
    assert send(c,cfg).status_code==200
    assert c.get(url).json()['condition']=='normal'
    wrong=rows(cfg);wrong[0]['unit']='F'
    assert send(c,cfg,wrong).json()['results'][0]['status']=='rejected'
    assert c.get(url).json()['condition']=='unknown'
    bad=rows(cfg);bad[0]['quality']='bad'
    assert send(c,cfg,bad).json()['results'][0]['status']=='flagged'
    assert c.get(url).json()['condition']=='unknown'
    send(c,cfg)
    with transaction() as conn:
        conn.execute('DELETE FROM telemetry WHERE asset_id=?',('FACTORY-PUMP-01',))
    assert send(c,cfg,rows(cfg,stamp=iso(utcnow()-timedelta(minutes=2)))).status_code==200
    assert c.get(url).json()['condition']=='unknown'
    source=next(s for s in c.get('/api/integrations/status').json() if s['id']==cfg['source_id'])
    # Arrival is fresh but original observation time remains stale.
    assert source['status']=='connected'


def test_simulation_run_pause_persists_and_requires_demo_scope(foundation,monkeypatch):
    c=foundation;seed_risk();engineer=login(c,'engineer','engineer-demo-password')
    assert c.put('/api/simulation',json={'enabled':False},headers=engineer).json()['enabled'] is False
    with transaction() as conn:before=conn.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0]
    assert simulate_tick()==[];migrate()
    assert c.get('/api/simulation').json()['enabled'] is False
    with transaction() as conn:assert conn.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0]==before
    assert c.put('/api/simulation',json={'enabled':True},headers=engineer).json()['enabled'] is True
    assert c.get('/api/assets/DEMO-CONVEYOR-001').json()['condition']=='warning'
    assert c.get('/api/assets/PLA-PUMP-001').json()['condition']=='normal'
    with transaction() as conn:conn.execute("DELETE FROM asset_grants WHERE user_id='engineer' AND asset_id='PLA-PUMP-001'")
    assert c.put('/api/simulation',json={'enabled':False},headers=engineer).status_code==403
    monkeypatch.setenv('MAINTENANCEAI_MODE','production')
    assert c.get('/api/simulation').json()['available'] is False
    assert c.put('/api/simulation',json={'enabled':True}).status_code==403
    assert simulate_tick()==[]


def test_photo_and_marker_authorization_and_validation(foundation):
    c=foundation;onboard(c);aid='FACTORY-PUMP-01';base=f'/api/admin/assets/{aid}';read=f'/api/assets/{aid}'
    initial=c.get(read+'/diagram').json();assert initial['layout_kind']=='illustrative' and not initial['photo_available']
    layout={'markers':[dict(tag_id='temperature',label='Drive-end bearing housing',x=40,y=50),dict(tag_id='vibration_rms',label='Pump bearing housing',x=65,y=48)]}
    assert c.put(base+'/diagram',json={'markers':layout['markers'][:1]}).status_code==422
    assert c.put(base+'/diagram',json=layout).status_code==200
    assert c.get(read+'/diagram').json()['markers']==layout['markers']
    assert c.put(base+'/photo',files={'file':('bad.svg',b'<svg onload="alert(1)"/>','image/svg+xml')}).status_code==422
    image=io.BytesIO();Image.new('RGB',(600,400),'navy').save(image,format='PNG')
    response=c.put(base+'/photo',files={'file':('machine.png',image.getvalue(),'image/png')});assert response.status_code==200
    assert response.json()['photo_available'] and response.json()['layout_kind']=='illustrative'
    photo=c.get(read+'/photo');assert photo.status_code==200 and photo.headers['content-type']=='image/jpeg'
    assert Image.open(io.BytesIO(photo.content)).size==(600,400)
    engineer=login(c,'engineer','engineer-demo-password')
    assert c.get(read+'/photo',headers=engineer).status_code==404
    assert c.put(base+'/diagram',json=layout,headers=engineer).status_code==403


def test_actual_loopback_opcua_read_reaches_machine_monitoring(foundation):
    from asyncua import Server,Client
    from scripts.opcua_gateway import forward_once,validate
    c=foundation;cfg=onboard(c)
    async def run():
        sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
        server=Server();await server.init();endpoint=f'opc.tcp://127.0.0.1:{port}/test/'
        server.set_endpoint(endpoint);ns=await server.register_namespace('maintenanceai:test:gateway')
        temp=await server.nodes.objects.add_variable(ns,'temperature',42.5);vib=await server.nodes.objects.add_variable(ns,'vibration',1.25)
        config={**cfg,'endpoint':'http://testserver'+cfg['endpoint'],'opcua_endpoint':endpoint,'allow_insecure_loopback':True,'ot_owner_approved':True,
                'mappings':[dict(node_id=n.nodeid.to_string(),asset_id='FACTORY-PUMP-01',tag_id=tag,unit=unit,operating_state='running') for n,tag,unit in [(temp,'temperature','degC'),(vib,'vibration_rms','mm/s')]]}
        # TestClient accepts a loopback URL without making external network calls.
        config['endpoint']='http://127.0.0.1:8000'+cfg['endpoint']
        async with server:
            async with Client(endpoint) as client:
                last_seen={}
                report=await forward_once(client,config,c,last_seen)
                assert (await forward_once(client,config,c,last_seen))['results']==[]
                assert all(r['status']=='accepted' for r in report['results'])
                assert await temp.read_value()==42.5 and await vib.read_value()==1.25
        assert c.get('/api/assets/FACTORY-PUMP-01').json()['condition']=='normal'
        with pytest.raises(ValueError):validate({**config,'ot_owner_approved':False})
        with pytest.raises(ValueError):validate({**config,'opcua_endpoint':'opc.tcp://factory-gateway:4840'})
        with pytest.raises(ValueError):validate({**config,'endpoint':'http://factory-api'+cfg['endpoint']})
    asyncio.run(run())
