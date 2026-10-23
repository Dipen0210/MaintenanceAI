"""Execute real neural embedding and local generation, recording usage and model identity."""
import json
from backend.core.database import ROOT
from backend.copilot.providers import embed,language_provider


def main():
    vectors,version=embed(['Conveyor bearing vibration investigation','Motor bearing vibration diagnosis','Office lunch menu'])
    related=float(vectors[0]@vectors[1]);unrelated=float(vectors[0]@vectors[2])
    assert vectors.shape==(3,384) and related>unrelated
    result=language_provider().complete([
        {'role':'system','content':'Select relevant evidence IDs for the question. Return only JSON with key evidence_ids and a list of IDs. Do not invent evidence.'},
        {'role':'user','content':'Question: Why is the conveyor flagged? Evidence: E1: conveyor vibration is 7.2 mm/s. E2: a compatible synthetic model reports a bearing-impulse-like pattern. E3: office lunch menu. Return only the relevant evidence IDs.'}],128)
    report={'embedding_model':version,'embedding_shape':list(vectors.shape),'related_similarity':related,'unrelated_similarity':unrelated,
            'language_model':result.model_id,'generated_text':result.text,'input_tokens':result.input_tokens,'output_tokens':result.output_tokens,
            'latency_ms':result.latency_ms,'estimated_cost_usd':result.estimated_cost_usd,'cost_basis':result.cost_basis,
            'scope':'Actual local embedding and generation smoke test; not a copilot quality certification.'}
    (ROOT/'docs/evaluation/language-model-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
