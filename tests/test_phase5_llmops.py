"""Control-plane tests use explicit test fixtures, never reported as measured LLM quality."""
from copy import deepcopy
import hashlib,json
import pytest
from backend.core.database import ROOT,transaction,identifier,iso
from backend.copilot.configuration import manifest,register
from backend.copilot.evaluation import release_reasons,summarize,assess_response,authorization_probe
from backend.copilot.tools import TOOLS
from backend.copilot.service import selection_schema,parse_selection
from test_phase4_copilot import copilot,injected,incident,ask,ASSET


def gate_fixture():
    """Deliberately constructed unit input to exercise gate branches, not an evaluation run."""
    path=ROOT/'fixtures/evaluation/copilot-v1.json';suite=json.loads(path.read_text())
    names=['citation_validity','claim_grounding','evidence_roles','safe_content','human_review','equipment_applicability','refusal','required_abstention','tool_results']
    cases=[{'id':case['id'],'response':{'asset_id':ASSET,'status':case['expected_status']},'assessment':{'critical_checks':{n:True for n in names},'task_success':True,'latency_ms':1,'input_tokens':1,'output_tokens':1}} for case in suite['cases']]
    auth={n:True for n in list(TOOLS)+['write_plc','execute_sql','read_secret','http_fetch']}
    return {'id':identifier(),'measurement_kind':'actual_local_inference','configuration':manifest(),'dataset_version':suite['version'],
            'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'mode':'tools_rag','cases':cases,'authorization':auth,'metrics':summarize(cases,auth),'scope':'UNIT TEST FIXTURE ONLY'}


def test_configuration_tracks_actual_sources_schemas_models_and_modes():
    first=manifest();assert first==manifest()
    assert first['id']!=manifest('docs_rag')['id']
    assert 'backend/copilot/service.py' in first['manifest']['sources']
    assert first['manifest']['schemas']['tool_arguments']['additionalProperties'] is False
    assert first['manifest']['language_artifacts']['language']['revision']


def test_schema_omits_unsupported_roles_instead_of_permitting_empty_role_items():
    schema=selection_schema({'E1':{'role':'observation'}})
    assert set(schema['properties'])=={'evidence_ids'}
    assert schema['additionalProperties'] is False
    assert parse_selection('{"evidence_ids":["E1"]}').check_ids==[]
    assert selection_schema({})['properties']=={}


def test_gate_requires_human_review_and_current_complete_evidence():
    report=gate_fixture()
    assert release_reasons(report)==['A qualified human must review the concrete evaluation examples before release acceptance']
    assert release_reasons(report,expert_review={'decision':'accepted'})==[]
    for field in ['configuration','dataset_sha256','measurement_kind']:
        changed=deepcopy(report);changed[field]={} if field=='configuration' else 'invalid'
        assert release_reasons(changed,require_expert=False)
    changed=deepcopy(report);changed['cases']=changed['cases'][:-1]
    assert any('missing' in reason for reason in release_reasons(changed,require_expert=False))


@pytest.mark.parametrize('name',['citation_validity','safe_content','equipment_applicability','required_abstention','human_review'])
def test_injected_critical_regression_blocks_release(name):
    report=gate_fixture();report['cases'][1]['assessment']['critical_checks'][name]=False
    # A deliberately forged green summary must never suppress a failing case.
    report['metrics']['task_success_rate']=1
    reasons=release_reasons(report,require_expert=False)
    assert any('bearing_incident' in reason for reason in reasons)


def test_authorization_and_quality_regressions_are_release_blockers():
    baseline=gate_fixture();candidate=deepcopy(baseline)
    candidate['authorization']['get_latest_telemetry']=False
    assert any('authorization' in r for r in release_reasons(candidate,require_expert=False))
    candidate=deepcopy(baseline);candidate['cases'][0]['assessment']['task_success']=False
    assert any('regressed' in r for r in release_reasons(candidate,baseline,require_expert=False))


def test_trace_roles_sources_and_config_identity(copilot,injected):
    incident(copilot);result=ask(copilot).json();tid=result['trace_id']
    response=copilot.get('/api/copilot/traces/'+tid)
    assert response.status_code==200,response.text
    assert response.json()['configuration_id']==manifest()['id']
    assert response.json()['details']['validation'].startswith('passed')
    assert copilot.get('/api/copilot/traces',params={'asset_id':ASSET}).json()[0]['id']==tid
    login=copilot.post('/api/auth/login',json={'username':'engineer','password':'engineer-demo-password'}).json()
    copilot.headers['Authorization']='Bearer '+login['access_token']
    assert copilot.get('/api/copilot/traces/'+tid).status_code==404
    login=copilot.post('/api/auth/login',json={'username':'admin','password':'admin-demo-password'}).json()
    copilot.headers['Authorization']='Bearer '+login['access_token']
    with transaction() as conn:conn.execute("DELETE FROM source_grants WHERE user_id='admin' AND source_id='demo-simulator'")
    assert copilot.get('/api/copilot/traces/'+tid).status_code==404
    assert copilot.get('/api/copilot/traces',params={'asset_id':ASSET}).json()==[]


def test_actual_missing_manual_abstains_instead_of_selecting_an_invalid_check(copilot):
    incident(copilot)
    with transaction() as conn:conn.execute("UPDATE documents SET status='revoked'")
    response=ask(copilot,'Why is this conveyor flagged and which approved checks apply?')
    assert response.status_code==200,response.text
    result=response.json()
    assert result['status']=='abstained',result
    assert result['model_id'].startswith('Qwen/Qwen3-0.6B@')
    assert not result['hypotheses'] and not result['approved_checks']


@pytest.mark.parametrize('regression',['unsafe_claim','invented_citation','unauthorized_asset'])
def test_evaluator_detects_injected_response_regressions_before_release(copilot,injected,regression):
    incident(copilot);result=ask(copilot).json();assert result['status']=='answered'
    with transaction() as conn:trace=json.loads(conn.execute('SELECT details FROM copilot_traces WHERE id=?',(result['trace_id'],)).fetchone()[0])
    if regression=='unsafe_claim':result['approved_checks'][0]['text']='Inspect the energized motor.'
    elif regression=='invented_citation':result['hypotheses'][0]['citations']=['document:invented-source']
    else:result['citations'][0]['asset_id']='unauthorized-asset'
    case=next(c for c in json.loads((ROOT/'fixtures/evaluation/copilot-v1.json').read_text())['cases'] if c['id']=='bearing_incident')
    assessment=assess_response(case,result,{'id':'admin','plant_id':'demo-plant','role':'administrator'},trace)
    assert not all(assessment['critical_checks'].values())
    report=gate_fixture();report['cases'][1]['assessment']=assessment
    assert any('bearing_incident' in reason for reason in release_reasons(report,require_expert=False))


def test_probe_detects_accidentally_removed_tool_authorization(monkeypatch):
    # Deliberately break a tool implementation; the probe must detect the returned data.
    monkeypatch.setattr('backend.copilot.evaluation.invoke',lambda *args,**kwargs:{'private':'injected test value'})
    checks=authorization_probe({'id':'test-user'})
    assert len(checks)==15 and not any(checks.values())


def test_portfolio_trace_reauthorizes_each_risk_asset(copilot,injected):
    incident(copilot)
    assert copilot.post('/api/risks/evaluate',json={'asset_ids':[ASSET,'PLA-FAN-001']}).status_code==200
    result=ask(copilot,'Which machines need maintenance review first?').json()
    assert result['status']=='answered',result
    trace_id=result['trace_id']
    response=copilot.get('/api/copilot/traces/'+trace_id)
    assert response.status_code==200,response.text
    with transaction() as conn:conn.execute("DELETE FROM asset_grants WHERE user_id='admin' AND asset_id='PLA-FAN-001'")
    assert copilot.get('/api/copilot/traces/'+trace_id).status_code==404


def test_release_workflow_records_blocked_decisions_and_enforces_review(copilot):
    report=gate_fixture();config=register()
    with transaction() as conn:
        conn.execute('INSERT INTO evaluation_runs VALUES(?,?,?,?,?,?,?,?,?,?,?)',(report['id'],'demo-plant','admin',config['id'],report['dataset_version'],report['dataset_sha256'],'tools_rag',iso(),iso(),'passed',json.dumps(report)))
    listing=copilot.get('/api/evaluations');assert listing.status_code==200
    response=copilot.post('/api/copilot/releases',json={'run_id':report['id']})
    assert response.status_code==200 and response.json()['status']=='blocked'
    with transaction() as conn:assert conn.execute('SELECT status FROM copilot_releases').fetchone()[0]=='blocked'
    body={'qualification':'Automated API test identity; not a real expert review','attest_qualified':True,'decision':'accepted','notes':'This isolated test exercises authorization and case-coverage validation.','reviewed_case_ids':[case['id'] for case in report['cases']][:-1]}
    assert copilot.post('/api/evaluations/'+report['id']+'/review',json=body).status_code==422
    # Simulate the user-driven positive path only in this disposable test DB.
    # This is not a review of the measured project evaluation reports.
    body['reviewed_case_ids']=[case['id'] for case in report['cases']]
    approved=copilot.post('/api/evaluations/'+report['id']+'/review',json=body)
    assert approved.status_code==200,approved.text
    accepted=copilot.post('/api/copilot/releases',json={'run_id':report['id']})
    assert accepted.status_code==200 and accepted.json()['status']=='accepted',accepted.text
    body['decision']='rejected'
    assert copilot.post('/api/evaluations/'+report['id']+'/review',json=body).status_code==200
    assert copilot.post('/api/copilot/releases',json={'run_id':report['id']}).json()['status']=='blocked'
    token=copilot.post('/api/auth/login',json={'username':'engineer','password':'engineer-demo-password'}).json()['access_token'];copilot.headers['Authorization']='Bearer '+token
    assert copilot.get('/api/evaluations').status_code==403
    assert copilot.post('/api/copilot/releases',json={'run_id':report['id']}).status_code==403
    assert copilot.post('/api/evaluations/'+report['id']+'/review',json=body).status_code==403
