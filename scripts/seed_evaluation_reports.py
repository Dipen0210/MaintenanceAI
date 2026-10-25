"""Import existing measured repository reports into an explicitly simulated demo database."""
import json,os
from backend.core.database import ROOT,migrate,transaction,iso
from backend.core.demo import seed_demo
from backend.copilot.configuration import manifest


def run():
    if os.getenv('MAINTENANCEAI_MODE','demo')!='demo':raise RuntimeError('Measured demo reports cannot be seeded into production')
    migrate();seed_demo();imported=[]
    for mode in ['no_rag','docs_rag','tools_rag']:
        candidates=[ROOT/f'docs/evaluation/meter-rounds/copilot-evaluation-{mode}.json',ROOT/f'docs/evaluation/qr-pm/copilot-evaluation-{mode}.json',ROOT/f'docs/evaluation/r1-intake/copilot-evaluation-{mode}.json',ROOT/f'docs/evaluation/field-work/copilot-evaluation-{mode}.json',ROOT/f'docs/evaluation/machine-onboarding/copilot-evaluation-{mode}.json',ROOT/f'docs/evaluation/copilot-evaluation-{mode}.json']
        reports=[json.loads(path.read_text()) for path in candidates if path.is_file()]
        report=next((r for r in reports if r['measurement_kind']=='actual_local_inference' and r['configuration']['id']==manifest(mode)['id']),None)
        if report is None:continue
        config=report['configuration']
        with transaction() as conn:
            conn.execute('INSERT OR IGNORE INTO copilot_configs VALUES(?,?,?)',(config['id'],json.dumps(config['manifest']),iso()))
            conn.execute('INSERT OR IGNORE INTO evaluation_runs VALUES(?,?,?,?,?,?,?,?,?,?,?)',(report['id'],'demo-plant','admin',config['id'],report['dataset_version'],report['dataset_sha256'],mode,report['started_at'],report['finished_at'],report['status'],json.dumps(report)))
        imported.append(report['id'])
    return imported


if __name__=='__main__':print(json.dumps({'imported_actual_report_ids':run()}))
