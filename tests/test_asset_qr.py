"""Decode actual generated QR pixels and verify grant boundaries."""
import io
from PIL import Image
import zxingcpp
from tests.test_phase1_foundation import foundation
from tests.test_machine_onboarding import onboard
from tests.test_phase6_scheduling import login
from backend.core.database import transaction


def test_qr_decodes_to_scoped_link_not_credentials(foundation):
    c=foundation;r=c.get('/api/assets/DEMO-CONVEYOR-001/qr',params={'origin':'http://localhost:3000'})
    assert r.status_code==200 and r.headers['cache-control']=='private, no-store'
    decoded=zxingcpp.read_barcode(Image.open(io.BytesIO(r.content)))
    assert decoded and decoded.text=='http://localhost:3000/operations/scan?asset=DEMO-CONVEYOR-001'
    for origin in ['https://attacker.invalid','http://[bad']:
        assert c.get('/api/assets/DEMO-CONVEYOR-001/qr',params={'origin':origin}).status_code==422
    assert c.get('/api/assets/DEMO-CONVEYOR-001/qr',params={'origin':'http://localhost:3000'},headers={'X-Maintenance-Workspace':'plant'}).status_code==404


def test_resolution_changes_only_origin_view_never_grants(foundation):
    c=foundation;onboard(c)
    r=c.get('/api/asset-access/FACTORY-PUMP-01',headers={'X-Maintenance-Workspace':'demo'})
    assert r.json()=={'asset_id':'FACTORY-PUMP-01','workspace':'plant'}
    login(c,'engineer');assert c.get('/api/asset-access/FACTORY-PUMP-01').status_code==404
    with transaction() as conn:conn.execute("DELETE FROM asset_grants WHERE user_id='engineer' AND asset_id='DEMO-CONVEYOR-001'")
    assert c.get('/api/asset-access/DEMO-CONVEYOR-001').status_code==404
    c.headers.pop('Authorization');c.cookies.clear();assert c.get('/api/asset-access/DEMO-CONVEYOR-001').status_code==401
