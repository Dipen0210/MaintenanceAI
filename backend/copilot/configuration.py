"""Content-addressed configuration identities; no provider credentials or raw prompts in traces."""
import hashlib,json,os
from backend.core.database import ROOT,iso,transaction
from backend.copilot.contracts import ChatResponse,ToolArguments,Selection

RETRIEVAL={'version':'equipment-approved-cosine-v1','minimum_similarity':0.2,'maximum_chunks':12,
           'chunk_words':150,'chunk_stride':120,'embedding_tokens':256,'current_prediction_seconds':60}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def manifest(mode='tools_rag'):
    if mode not in ['no_rag','docs_rag','tools_rag']:raise ValueError('Unsupported retrieval mode')
    sources=['backend/copilot/'+name for name in ['service.py','tools.py','documents.py','evidence.py','safety.py','providers.py','contracts.py','configuration.py']]
    sources+=['backend/copilot/prompts/evidence_selection_v3.txt','fixtures/demo/conveyor-guide.md']
    sources+=['backend/core/security.py','backend/ingestion/service.py','backend/ml/service.py','backend/risk/service.py','backend/risk/engine.py',
              'backend/copilot/evaluation.py','scripts/evaluate_copilot.py']
    sources+=['backend/scheduling/service.py','backend/scheduling/solver.py','backend/scheduling/contracts.py','backend/scheduling/execution.py']
    sources+=['backend/routes/machines.py','backend/routes/simulation.py','backend/ingestion/condition.py','backend/ingestion/adapters.py','scripts/opcua_gateway.py']
    sources+=['backend/preventive/service.py','backend/preventive/calendar.py','backend/preventive/contracts.py']
    sources+=['backend/preventive/meters.py','backend/routes/meters.py']
    sources+=['backend/scheduling/lifecycle.py','backend/core/database.py','backend/notifications/service.py','backend/notifications/events.py','backend/notifications/conditions.py','backend/routes/notifications.py']
    sources+=['backend/scheduling/coordination.py','backend/routes/coordination.py']
    sources+=['backend/inspections/service.py','backend/routes/inspections.py']
    sources+=['backend/core/workspaces.py','backend/routes/issues.py']
    sources+=['backend/routes/administration.py','backend/core/demo.py','backend/routes/ml.py','backend/ml/registry.py']
    languages=json.loads((ROOT/'models/language-models.json').read_text())
    result={'format':'maintenanceai-copilot-config-v1','mode':mode,
            'prompt_version':'evidence-selection-v3','retrieval':RETRIEVAL,
            'schemas':{'response':ChatResponse.model_json_schema(),'tool_arguments':ToolArguments.model_json_schema(),'selection':Selection.model_json_schema()},
            'sources':{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources},
            'language_artifacts':languages,'provider':os.getenv('MAINTENANCEAI_LLM_PROVIDER','local'),
            'remote_model':os.getenv('MAINTENANCEAI_LLM_MODEL') if os.getenv('MAINTENANCEAI_LLM_PROVIDER')=='http' else None,
            'generation':{'temperature':0,'maximum_output_tokens':256,'maximum_context_tokens':6000,'schema_constrained':True},
            'predictive_registry_sha256':hashlib.sha256((ROOT/'models/synthetic-registry.json').read_bytes()).hexdigest()}
    return {'id':digest(result),'manifest':result}


def register(mode='tools_rag'):
    config=manifest(mode)
    with transaction() as conn:
        conn.execute('INSERT OR IGNORE INTO copilot_configs VALUES(?,?,?)',(config['id'],canonical(config['manifest']),iso()))
    return config
