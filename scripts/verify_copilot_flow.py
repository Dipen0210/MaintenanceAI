"""Execute the actual local RAG service against an isolated, explicitly simulated incident."""
import json,os,tempfile
from pathlib import Path
from backend.core.database import ROOT,migrate,transaction
from backend.core.demo import seed_demo
from backend.ml.registry import initialize_registry
from backend.risk.service import seed_risk,evaluate
from backend.copilot.demo import seed_copilot
from backend.copilot.contracts import ChatRequest
from backend.copilot.service import chat
from backend.copilot.providers import language_provider
from backend.routes.ml import demo_prediction,DemoRun


class RecordingProvider:
    def complete(self,*args,**kwargs):
        result=language_provider().complete(*args,**kwargs)
        (ROOT/'docs/evaluation/copilot-generation-debug.json').write_text(json.dumps(result.__dict__,indent=2)+'\n')
        print('MODEL OUTPUT',result.text,flush=True)
        return result


def main():
    with tempfile.TemporaryDirectory(prefix='maintenanceai-copilot-') as directory:
        os.environ['MAINTENANCEAI_DB']=str(Path(directory)/'app.db');os.environ['MAINTENANCEAI_MODE']='demo'
        migrate();seed_demo();initialize_registry();seed_risk();seed_copilot()
        user=dict(id='engineer',username='engineer',role='engineer',plant_id='demo-plant')
        demo_prediction('DEMO-CONVEYOR-001',DemoRun(),user);evaluate(['DEMO-CONVEYOR-001'],user)
        result=chat(ChatRequest(asset_id='DEMO-CONVEYOR-001',message='Why is this motor flagged for abnormal vibration? Explain the evidence, possible cause, alternatives and approved next checks.'),user,provider=RecordingProvider())
        (ROOT/'docs/evaluation/copilot-flow-smoke.json').write_text(json.dumps(result,indent=2)+'\n')
        with transaction() as conn:trace=json.loads(conn.execute('SELECT details FROM copilot_traces WHERE id=?',(result['trace_id'],)).fetchone()[0])
        (ROOT/'docs/evaluation/copilot-trace-smoke.json').write_text(json.dumps(trace,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        assert result['status']=='answered',result['status']
        assert result['observations'] and result['predictions'] and result['hypotheses'] and result['alternatives'] and result['approved_checks']
if __name__=='__main__':main()
