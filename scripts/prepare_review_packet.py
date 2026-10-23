"""Render measured examples for a future human reviewer; never manufacture a review."""
import json
from backend.core.database import ROOT


def run(directory=None):
    directory=directory or ROOT/'docs/evaluation';reports=[json.loads((directory/f'copilot-evaluation-{mode}.json').read_text()) for mode in ['no_rag','docs_rag','tools_rag']]
    full=reports[-1]
    lines=['# Copilot evaluation: qualified reviewer packet','',
           '**Expert review is deferred by explicit user instruction on 2026-10-21.** Implementation may continue; no expert findings, approval or industrial diagnostic certification are claimed. This packet supports a future real review.','',
           'Source: `copilot-evaluation-tools_rag.json` (actual local model evaluation).',f'Run: `{full["id"]}`',f'Configuration: `{full["configuration"]["id"]}`','',
           'A qualified maintenance/reliability practitioner should inspect every example and its evidence, then record qualifications, decision, corrections and findings. Synthetic reference checks are not OEM procedures or plant authorization. The application cannot independently certify expertise.','',
           '| Configuration | Task expectations met | Mean latency | Input / output tokens |','|---|---:|---:|---:|']
    for report in reports:
        m=report['metrics'];lines.append(f'| {report["mode"]} | {m["task_success_rate"]*100:.1f}% of {m["case_count"]} | {m["mean_latency_ms"]/1000:.2f} s | {m["input_tokens"]} / {m["output_tokens"]} |')
    lines+=['','These are controlled retrieval ablations with deliberate evidence-dependent abstention, not unrestricted-chatbot benchmarks or physical root-cause accuracy. Local inference cost is unknown. One-time model initialization is excluded from request latency.','']
    for case in full['cases']:
        r=case['response'];a=case['assessment']
        lines += [f'## {case["id"]}: {case["category"]}','',f'Question: {case["question"]}',f'Synthetic setup: `{case["setup"]}`',f'Expected / returned status: {a["expected_status"]} / {a["actual_status"]}.',f'Task expectation met: {a["task_success"]}.','',r['answer'],'']
        for key in ['observations','predictions','hypotheses','alternatives','approved_checks','maintenance_history','manual_references']:
            lines += ['### '+key.replace('_',' ').capitalize(),'']
            lines += ['- '+claim['text']+' (citations: '+', '.join(claim['citations'])+')' for claim in r[key]] or ['None.']
            lines.append('')
        lines+=['### Uncertainty and missing information','',r['uncertainty'],'']+['- '+m for m in r['missing_information']]+['','### Source evidence','']
        for c in r['citations']:
            lines += [f'- `{c["id"]}` · {c["title"]} · revision {c.get("revision") or "n/a"} · page {c.get("page") or "n/a"} · {c.get("timestamp_utc") or "approved document"}', '  '+c['text']]
        lines+=['']
    (directory/'COPILOT_EXPERT_REVIEW.md').write_text('\n'.join(lines)+'\n')
    template=dict(run_id=full['id'],reviewer=None,qualification=None,decision='pending',reviewed_case_ids=[],required_case_ids=[c['id'] for c in full['cases']],findings=None)
    (directory/'copilot-expert-review.template.json').write_text(json.dumps(template,indent=2)+'\n')
    print('Prepared pending-review packet for measured run',full['id'])


if __name__=='__main__':run()
