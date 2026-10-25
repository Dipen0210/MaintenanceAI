"""Human issue intake, named assignments and scoped demo/plant views."""
from concurrent.futures import ThreadPoolExecutor
import io
from uuid import uuid4
import pytest
from PIL import Image
from fastapi import HTTPException
from backend.core.database import transaction,migrate
from backend.routes.issues import Change,update
from tests.test_phase1_foundation import foundation
from tests.test_machine_onboarding import onboard
from tests.test_phase4_copilot import copilot,ASSET
from tests.test_phase6_scheduling import planning,login,draft,proposed,APPROVAL
from tests.test_field_work import change

PLANT='FACTORY-PUMP-01'
DEMO={'X-Maintenance-Workspace':'demo'}
REAL={'X-Maintenance-Workspace':'plant'}


def report(c,asset=ASSET,**extra):
    return c.post('/api/issues',json=dict(asset_id=asset,title='Unusual bearing noise',description='Rattling observed beside the drive end during the morning walk.',reported_severity='high',request_key=str(uuid4()),**extra))


def issue_update(c,i,action,**extra):
    return c.post('/api/issues/'+i['id']+'/updates',json=dict(action=action,expected_version=i['version'],notes='Observed on the morning walk; review this investigation.',**extra))


def test_workspace_lists_direct_reads_sources_admin_and_socket(foundation):
    c=foundation;cfg=onboard(c)
    assert len(c.get('/api/assets',headers=DEMO).json())==15
    assert [a['id'] for a in c.get('/api/assets',headers=REAL).json()]==[PLANT]
    assert c.get('/api/assets/'+PLANT,headers=DEMO).status_code==404
    assert c.get('/api/assets/'+ASSET+'/telemetry',headers=REAL).status_code==404
    assert c.get('/api/assets',headers={'X-Maintenance-Workspace':'invalid'}).status_code==422
    assert {s['id'] for s in c.get('/api/integrations/status',headers=REAL).json()}=={cfg['source_id']}
    assert len(c.get('/api/admin/mappings',headers=REAL).json())==16
    assert len(c.get('/api/admin/sources',headers=REAL).json())==3
    r=c.get('/api/admin/assets/'+PLANT+'/telemetry',headers=DEMO)
    assert r.status_code==200 and r.json()['history']==[],r.text
    assert c.get('/api/simulation',headers=REAL).json()['available'] is False
    assert c.put('/api/simulation',headers=REAL,json={'enabled':True}).status_code==403
    with c.websocket_connect('/api/ws/telemetry?workspace=plant',headers={'Origin':'http://localhost:3000'}) as ws:
        assert [a['asset']['id'] for a in ws.receive_json()['assets']]==[PLANT]
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect('/api/ws/telemetry?workspace=wrong',headers={'Origin':'http://localhost:3000'}):pass


def test_operator_can_report_without_telemetry_but_cannot_approve(foundation):
    c=foundation;onboard(c)
    r=c.post('/api/admin/identities',json=dict(username='line.operator',password='operator-fixture-password',role='operator',enabled=True,asset_ids=[PLANT],source_ids=[],document_ids=[],reason='Test field report without permission to read telemetry.'))
    assert r.status_code==200,r.text
    operator=r.json()['id']
    token=c.post('/api/auth/login',json={'username':'line.operator','password':'operator-fixture-password'}).json()['access_token']
    c.headers.update({'Authorization':'Bearer '+token,**REAL})
    i=report(c,PLANT);assert i.status_code==200,i.text
    assert i.json()['created_by']==operator
    assert c.get('/api/assets/'+PLANT+'/telemetry').json()['history']==[]
    assert issue_update(c,i.json(),'close').status_code==403
    assert c.post('/api/risks/evaluate',json={'asset_ids':[PLANT]}).status_code==403
    assert report(c,ASSET).status_code==404
    with transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM work_orders').fetchone()[0]==0
        assert conn.execute('SELECT COUNT(*) FROM risk_assessments').fetchone()[0]==0
    login(c,'admin');assert c.get('/api/admin/identities').status_code==200


def test_report_idempotency_validation_and_workspace_persistence(foundation):
    c=foundation;onboard(c);c.headers.update(REAL)
    body=dict(asset_id=PLANT,title='Cooling pump leak',description='Fluid observed under the pump housing at shift start.',request_key=str(uuid4()))
    first=c.post('/api/issues',json=body);assert first.status_code==200,first.text
    assert c.post('/api/issues',json=body).json()['id']==first.json()['id']
    assert c.post('/api/issues',json={**body,'title':'Another report'}).status_code==409
    assert c.post('/api/issues',json={**body,'title':'     '}).status_code==422
    assert c.get('/api/issues',headers=DEMO).json()==[]
    migrate();assert c.get('/api/issues').json()[0]['id']==first.json()['id']


def test_assign_investigate_close_reopen_and_revocation(foundation):
    c=foundation;i=report(c).json()
    login(c,'engineer');assert issue_update(c,i,'assign',assignee_id='engineer').status_code==403
    login(c,'manager')
    assert issue_update(c,i,'assign',assignee_id='admin').status_code==422
    i=issue_update(c,i,'assign',assignee_id='engineer').json();assert i['status']=='assigned'
    login(c,'engineer');assert c.get('/api/my-work').json()['issues'][0]['id']==i['id']
    r=issue_update(c,i,'start');assert r.status_code==200,r.text
    assert issue_update(c,i,'note').status_code==409
    i=r.json();assert i['status']=='investigating'
    i=issue_update(c,i,'note').json()
    assert issue_update(c,i,'close').status_code==403
    login(c,'manager');i=issue_update(c,i,'close').json();assert i['status']=='closed'
    login(c,'engineer');assert c.get('/api/my-work').json()['issues']==[]
    login(c,'manager');i=issue_update(c,i,'reopen').json();assert i['status']=='reported' and i['assignee_id'] is None
    with transaction() as conn:conn.execute('DELETE FROM asset_grants WHERE user_id=? AND asset_id=?',('engineer',ASSET))
    assert issue_update(c,i,'assign',assignee_id='engineer').status_code==422
    login(c,'engineer');assert c.get('/api/issues').json()==[]
    assert issue_update(c,i,'note').status_code==404


def test_concurrent_updates_preserve_history(foundation):
    c=foundation;i=report(c).json();login(c,'manager');i=issue_update(c,i,'assign',assignee_id='engineer').json()
    body=Change(action='note',expected_version=i['version'],notes='Concurrent field inspection notes remain version protected.')
    user=dict(id='engineer',plant_id='demo-plant',role='engineer',workspace='demo')
    def write():
        try:return update(i['id'],body,user)['version']
        except HTTPException as exc:return exc.status_code
    with ThreadPoolExecutor(max_workers=2) as pool:assert sorted(pool.map(lambda _:write(),range(2)))==[3,409]
    assert len(c.get('/api/issues').json()[0]['events'])==3


def test_issue_photos_normalize_and_recheck_access(foundation):
    c=foundation;i=report(c).json();url='/api/issues/'+i['id']+'/photos'
    buf=io.BytesIO();img=Image.new('RGB',(20,10),'red');exif=Image.Exif();exif[0x010E]='PRIVATE EXIF TEST';img.save(buf,format='JPEG',exif=exif)
    assert c.post(url,files={'file':('bad.jpg',b'not an image','image/jpeg')}).status_code==422
    r=c.post(url,files={'file':('photo.jpg',buf.getvalue(),'image/jpeg')});assert r.status_code==200,r.text
    photo=url+'/'+r.json()['id'];r=c.get(photo);assert r.status_code==200
    assert r.headers['cache-control']=='private, no-store' and b'PRIVATE EXIF TEST' not in r.content
    assert not Image.open(io.BytesIO(r.content)).getexif()
    assert c.get(photo,headers=REAL).status_code==404
    login(c,'engineer');assert c.post(url,files={'file':('photo.jpg',buf.getvalue(),'image/jpeg')}).status_code==403
    with transaction() as conn:conn.execute('DELETE FROM asset_grants WHERE user_id=? AND asset_id=?',('engineer',ASSET))
    assert c.get(photo).status_code==404


def test_named_maintenance_assignments_are_snapshotted_and_enforced(planning):
    c=planning;config=c.get('/api/planning/config').json()['config']
    # Only one named mechanical resource, so the solver's assignment is deterministic.
    tech=next(t for t in config['technicians'] if 'mechanical' in t['skills']);tech['user_id']='engineer';config['technicians']=[tech]
    assert c.put('/api/planning/config',json=config).status_code==200
    login(c,'engineer');wid=draft(c).json()['id'];login(c,'manager')
    assert c.post('/api/work-orders/'+wid+'/review',json=APPROVAL).status_code==200
    p=proposed(c,[wid]).json();r=c.post('/api/schedules/'+p['id']+'/approve',json=APPROVAL);assert r.status_code==200,r.text
    assert c.get('/api/my-work').json()['work_orders']==[]
    login(c,'reliability');assert change(c,wid,'start',site_clearance_confirmed=True).status_code==403
    login(c,'engineer');assert c.get('/api/my-work').json()['work_orders'][0]['id']==wid
    assert c.get('/api/my-work',headers=REAL).json()['work_orders']==[]
    assert change(c,wid,'start',site_clearance_confirmed=True).status_code==200
    assert change(c,wid,'report_fixed',1).status_code==200
    login(c,'manager');assert change(c,wid,'verify',2).status_code==200
    login(c,'engineer');assert c.get('/api/my-work').json()['work_orders']==[]


@pytest.mark.parametrize('revoke',['role','source','asset','disabled'])
def test_schedule_revalidates_assignee_access_atomically(planning,revoke):
    c=planning;config=c.get('/api/planning/config').json()['config'];tech=next(t for t in config['technicians'] if 'mechanical' in t['skills']);tech['user_id']='engineer';config['technicians']=[tech]
    assert c.put('/api/planning/config',json=config).status_code==200
    login(c,'engineer');wid=draft(c).json()['id'];login(c,'manager');c.post('/api/work-orders/'+wid+'/review',json=APPROVAL);p=proposed(c,[wid]).json()
    with transaction() as conn:
        if revoke=='role':conn.execute("UPDATE users SET role='operator' WHERE id='engineer'")
        elif revoke=='disabled':conn.execute("UPDATE users SET enabled=0 WHERE id='engineer'")
        elif revoke=='source':conn.execute("DELETE FROM source_grants WHERE user_id='engineer'")
        else:conn.execute("DELETE FROM asset_grants WHERE user_id='engineer'")
    assert c.post('/api/schedules/'+p['id']+'/approve',json=APPROVAL).status_code in (404,409)
    with transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM work_user_assignments').fetchone()[0]==0
        assert conn.execute('SELECT COUNT(*) FROM schedule_assignments').fetchone()[0]==0


def test_physical_asset_cannot_use_demo_procedures(copilot):
    c=copilot;onboard(c)
    with transaction() as conn:
        conn.execute("UPDATE assets SET equipment_type='conveyor_motor' WHERE id=?",(PLANT,))
        conn.execute("UPDATE documents SET asset_ids='[]'")
    assert c.get('/api/assets/'+PLANT+'/procedures').json()==[]


def test_demo_history_never_exposes_other_origin_source(foundation):
    from tests.test_phase1_foundation import record
    c=foundation
    source=c.post('/api/admin/sources',json={'name':'Imported physical observations','reason':'Explicit mixed historical source scope regression fixture.'}).json()['id']
    assert c.post('/api/ingestion/'+source+'/records',json={'records':[record(source=source)]}).json()['results'][0]['status']=='accepted'
    assert len(c.get('/api/assets/'+ASSET+'/telemetry').json()['history'])==1
    scoped=c.get('/api/assets/'+ASSET+'/telemetry',headers=DEMO).json()
    assert scoped['history']==[] and all(r['observation'] is None and r['last_ingestion'] is None for r in scoped['latest'])
    login(c,'engineer');assert c.get('/api/admin/assets/'+ASSET+'/telemetry').status_code==403
