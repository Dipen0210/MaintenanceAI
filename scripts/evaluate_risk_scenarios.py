"""Deterministic scenario benchmark. Expected priorities are separate authored fixtures."""
from datetime import datetime,timezone
import hashlib,json,time
from backend.core.database import ROOT
from backend.risk.engine import assess
from backend.risk.service import DEFAULT_PROFILE,DEFAULT_POLICY


def run():
    file=ROOT/'docs/evaluation/risk-scenarios.json';suite=json.loads(file.read_text());now=datetime(2026,10,20,tzinfo=timezone.utc)
    results={};cases={c['id']:c for c in suite['cases']};start=time.perf_counter()
    for c in suite['cases']:
        profile=dict(DEFAULT_PROFILE,**dict(zip(['safety','production','maintenance','damage'],c['criticality'])))
        rows=[dict(sensor={'tag_id':tag},effective_quality='stale' if tag in c.get('missing',[]) else 'good',
                   observation=dict(id=c['id']+':'+tag,value=value,unit=unit,source='scenario-fixture',timestamp_utc=now.isoformat()))
              for tag,value,unit in [('temperature',c['temperature'],'degC'),('vibration_rms',c['vibration'],'mm/s'),('current',c['current'],'A')]]
        predictions=[] if c['anomaly'] is None else [dict(id='fixture:'+c['id'],model_id='authored-fixture',family='fixture',input_id='fixture',
                  data_mode='simulation',status='SUCCEEDED',historical=False,signal_timestamp=now.isoformat(),timestamp_utc=now.isoformat(),
                  drift={'flagged':False},output={'is_anomaly':c['anomaly']>.95,'anomaly_score':c['anomaly']})]
        results[c['id']]=assess(profile,DEFAULT_POLICY,rows,predictions,now)
    group=suite['ranking_group'];arrival=group['arrival_order'];expected=group['expected_order']
    methods={'first_come_first_served':arrival,
             'anomaly_only':sorted(arrival,key=lambda key:-cases[key]['anomaly']),
             'fixed_rule':sorted(arrival,key=lambda key:-(2 if cases[key]['temperature']>=95 else 1 if cases[key]['anomaly']>.95 else 0)),
             'risk_based':sorted(arrival,key=lambda key:(results[key]['rank'],-(results[key]['score'] or 0),key))}
    pairs=[(expected[i],expected[j]) for i in range(len(expected)) for j in range(i+1,len(expected))]
    comparisons={name:{'order':order,'correct_pair_fraction':sum(order.index(a)<order.index(b) for a,b in pairs)/len(pairs)} for name,order in methods.items()}
    failures=[{'id':key,'expected':cases[key]['expected'],'actual':r['priority']} for key,r in results.items() if r['priority']!=cases[key]['expected']]
    report={'suite':suite['version'],'scenario_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'scope':suite['scope'],
            'policy':DEFAULT_POLICY,'cases':len(cases),'failures':failures,'comparisons':comparisons,'elapsed_ms':(time.perf_counter()-start)*1000,
            'results':{k:{'priority':r['priority'],'score':r['score'],'missing_evidence':r['missing_evidence']} for k,r in results.items()}}
    (ROOT/'docs/evaluation/risk-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    if failures or comparisons['risk_based']['correct_pair_fraction']!=1:raise RuntimeError('Risk scenario gate failed')
    return report

if __name__=='__main__':print(json.dumps(run(),indent=2))
