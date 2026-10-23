"""Inject explicitly labeled regressions into a measured run; prove deterministic rejection."""
from copy import deepcopy
import hashlib,json
from backend.core.database import ROOT
from backend.copilot.evaluation import release_reasons


def run(path=None,output=None):
    path=path or ROOT/'docs/evaluation/copilot-evaluation-tools_rag.json';measured=json.loads(path.read_text())
    technical=release_reasons(measured,require_expert=False)
    if technical:raise RuntimeError('The actual measured candidate must pass technical gates first: '+str(technical))
    results=[]
    for check in ['safe_content','citation_validity','claim_grounding','equipment_applicability','required_abstention']:
        candidate=deepcopy(measured);target=next(c for c in candidate['cases'] if c['id']=='bearing_incident')
        target['assessment']['critical_checks'][check]=False
        reasons=release_reasons(candidate,require_expert=False)
        assert any('bearing_incident' in reason for reason in reasons)
        results.append({'injection':check,'attributed_case':'bearing_incident','release_status':'blocked','reasons':reasons})
    candidate=deepcopy(measured);candidate['authorization']['get_latest_telemetry']=False
    reasons=release_reasons(candidate,require_expert=False);assert any('authorization' in r for r in reasons)
    results.append({'injection':'tool_authorization_failure','release_status':'blocked','reasons':reasons})
    candidate=deepcopy(measured);candidate['cases'][0]['assessment']['task_success']=False
    reasons=release_reasons(candidate,measured,require_expert=False);assert any('regressed' in r for r in reasons)
    results.append({'injection':'task_quality_regression','release_status':'blocked','reasons':reasons})
    candidate=deepcopy(measured);candidate['configuration']['id']='injected-prompt-or-model-change'
    reasons=release_reasons(candidate,require_expert=False);assert any('configuration' in r for r in reasons)
    results.append({'injection':'configuration_mismatch','release_status':'blocked','reasons':reasons})
    result={'scope':'Explicit injected gate regressions derived from a real measured report. These are not model-performance measurements.',
            'source_report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_run_id':measured['id'],
            'configuration_id':measured['configuration']['id'],'technical_candidate':'passed','human_review':'still required before release','injections':results}
    (output or ROOT/'docs/evaluation/copilot-release-regression.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
