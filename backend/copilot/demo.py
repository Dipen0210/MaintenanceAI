"""Clearly synthetic, demo-approved reference documents and imported maintenance records."""
import csv,io,json,os
from datetime import timedelta
from backend.core.database import ROOT,transaction,utcnow,iso,audit
from backend.copilot.contracts import DocumentMetadata,DocumentApproval,ChunkReview
from backend.copilot.documents import import_document,approve
from backend.copilot.records import import_csv


def seed_copilot():
    if os.getenv('MAINTENANCEAI_MODE','demo')!='demo':return
    user=dict(id='admin',plant_id='demo-plant',role='administrator')
    with transaction() as conn:
        if not conn.execute("SELECT 1 FROM users WHERE id='admin'").fetchone():return
        exists=conn.execute("SELECT 1 FROM documents WHERE plant_id='demo-plant' AND title='Synthetic conveyor investigation guide' AND revision='DEMO-2'").fetchone()
    if not exists:
        content=(ROOT/'fixtures/demo/conveyor-guide.md').read_bytes()
        result=import_document(content,'conveyor-guide.md',DocumentMetadata(title='Synthetic conveyor investigation guide',revision='DEMO-2',
                    equipment_types=['conveyor_motor'],asset_ids=['DEMO-CONVEYOR-001'],reader_ids=['admin','engineer','manager','reliability'],simulation_only=True),user)
        with transaction() as conn:chunks=[dict(r) for r in conn.execute('SELECT id,section FROM document_chunks WHERE document_id=?',(result['id'],))]
        reviews=[]
        for chunk in chunks:
            prefix=chunk['section'].split(':')[0].lower()
            kind={'hypothesis':'hypothesis','alternative':'alternative','check':'check','safety':'safety'}.get(prefix,'reference')
            faults=['imbalance_like'] if 'Imbalance-related' in chunk['section'] else ['bearing_impulse_like'] if kind=='hypothesis' else []
            reviews.append(ChunkReview(id=chunk['id'],kind=kind,fault_classes=faults))
        approve(result['id'],DocumentApproval(chunks=reviews,reason='Seeded approval for the explicitly synthetic offline demonstration only; no real OEM or facility approval.'),user)
    with transaction() as conn:
        retired=conn.execute("UPDATE documents SET status='revoked' WHERE plant_id='demo-plant' AND title='Synthetic conveyor investigation guide' AND revision='DEMO-1' AND status='approved'").rowcount
        if retired:audit(conn,user,'replace_demo_document','DEMO-1','Replaced by DEMO-2 with explicit hypothesis applicability')
    with transaction() as conn:exists=conn.execute("SELECT 1 FROM maintenance_records WHERE source_id='demo-import' AND external_id='DEMO-HISTORY-1'").fetchone()
    if not exists:
        rows=[dict(asset_id='DEMO-CONVEYOR-001',external_id='DEMO-HISTORY-1',record_type='inspection',timestamp_utc=iso(utcnow()-timedelta(days=7)),
                   summary='Synthetic history: an earlier inspection documented elevated bearing vibration and a loose sensor mounting; the mounting was corrected. This does not establish the current cause.',details=json.dumps({'simulation_only':True,'status':'closed'})),
              dict(asset_id='DEMO-CONVEYOR-001',external_id='DEMO-WO-1',record_type='work_order',timestamp_utc=iso(utcnow()-timedelta(days=7)),
                   summary='Synthetic historical work order: review vibration sensor mounting and compare the recorded baseline. Closed after human review in the simulated record.',details=json.dumps({'simulation_only':True,'status':'closed','duration_minutes':30})),
              dict(asset_id='DEMO-CONVEYOR-001',external_id='DEMO-SCHEDULE-1',record_type='schedule',timestamp_utc=iso(utcnow()-timedelta(days=7)),
                   summary='Synthetic historical maintenance window completed seven days before this demo. No future maintenance is approved by this record.',details=json.dumps({'simulation_only':True,'status':'completed'}))]
        buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        import_csv(buffer.getvalue().encode(),'demo-import',user)
