"""Measured CP-SAT versus two feasible-first greedy baselines on authored synthetic fixtures."""
from copy import deepcopy
from itertools import combinations
import json,time,hashlib
from backend.core.database import ROOT,iso
from backend.scheduling.solver import solve,validate,minute,stamp,PRIORITY,VERSION


def scenarios():
    start=minute('2030-01-01T08:00:00+00:00');period=dict(start_utc=stamp(start),end_utc=stamp(start+240))
    def work(wid,priority):return dict(id=wid,asset_id='SIM-A',content=dict(duration_minutes=60,required_skills=['mechanical'],team_size=1,parts={'kit':1},dependencies=[],due_at=stamp(start+90),priority=priority))
    base=dict(**period,orders=[work('low','Monitor'),work('high','High priority')],existing=[],
              config=dict(technicians=[dict(id='A',name='Synthetic technician A',skills=['mechanical','electrical'],availability=[period])],
                          parts=[dict(id='kit',name='Synthetic inspection kit',quantity=5,available_at=stamp(start))],
                          equipment_windows=[dict(**period,id='window-A',asset_id='SIM-A',disruption_cost_per_minute=1)],blackouts=[],simulation_only=True))
    result=[dict(id='priority_order',purpose='Earlier high-priority work avoids overdue exposure.',snapshot=deepcopy(base))]
    s=deepcopy(base);s['orders'][0]['content']['dependencies']=['high']
    result.append(dict(id='dependency',purpose='Precedence overrides arrival order.',snapshot=s))
    s=deepcopy(base);s['config']['parts'][0]['available_at']=stamp(start+60)
    s['config']['blackouts']=[dict(asset_id='SIM-A',start_utc=stamp(start),end_utc=stamp(start+90),reason='Synthetic production restriction')]
    result.append(dict(id='delivery_and_restriction',purpose='Parts arrival and operational blackout postpone work.',snapshot=s))
    s=deepcopy(base);s['config']['parts'][0]['quantity']=1
    result.append(dict(id='parts_shortage',purpose='No algorithm may invent missing stock.',snapshot=s))
    s=deepcopy(base);s['end_utc']=stamp(start+60);s['orders'][0]['content']['priority']='High priority';s['orders'][1]['content'].update(priority='Monitor',required_skills=['electrical'])
    s['orders'][1]['asset_id']='SIM-B';s['config']['technicians'].append(dict(id='B',name='Synthetic mechanical technician B',skills=['mechanical'],availability=[period]))
    s['config']['equipment_windows'].append(dict(**period,id='window-B',asset_id='SIM-B',disruption_cost_per_minute=1))
    result.append(dict(id='scarce_skill_assignment',purpose='Greedy use of a dual-skilled technician can block otherwise feasible parallel work.',snapshot=s))
    s=deepcopy(base);s['config']['equipment_windows'][0]['disruption_cost_per_minute']=1000
    s['config']['equipment_windows'].append(dict(id='low-disruption',asset_id='SIM-A',start_utc=stamp(start+60),end_utc=stamp(start+240),disruption_cost_per_minute=0))
    result.append(dict(id='disruption_tradeoff',purpose='Compare explicit disruption and delay preferences under identical hard constraints.',snapshot=s))
    return result


def greedy(snapshot,priority=False):
    started=time.perf_counter();orders=list(snapshot['orders']);assigned=[];done=[]
    if priority:orders.sort(key=lambda w:-PRIORITY[w['content']['priority']])
    while orders:
        # Precedence-aware, but never backtracks an earlier resource choice.
        possible=[w for w in orders if set(w['content']['dependencies'])<={x['id'] for x in done}|{r['work_order_id'] for r in snapshot['existing']}]
        if not possible:break
        w=possible[0];data=w['content'];chosen=None
        for start in range(minute(snapshot['start_utc'],True),minute(snapshot['end_utc'])-data['duration_minutes']+1):
            for team in combinations(sorted(snapshot['config']['technicians'],key=lambda t:t['id']),data['team_size']):
                for window in sorted(snapshot['config']['equipment_windows'],key=lambda w:w['id']):
                    a=dict(work_order_id=w['id'],asset_id=w['asset_id'],start_utc=stamp(start),end_utc=stamp(start+data['duration_minutes']),
                           technician_ids=[t['id'] for t in team],parts=data['parts'],window_id=window['id'],duration_minutes=data['duration_minutes'],priority=data['priority'],disruption_cost=data['duration_minutes']*window['disruption_cost_per_minute'])
                    if not validate(assigned+[a],{**snapshot,'orders':done+[w]}):chosen=a;break
                if chosen:break
            if chosen:break
        if not chosen:break
        assigned.append(chosen);done.append(w);orders.remove(w)
    success=not orders
    return dict(feasibility='feasible' if success else 'no_feasible_greedy_plan',assignments=assigned if success else [],elapsed_ms=round((time.perf_counter()-started)*1000,3),
                conflicts=[] if success else ['Greedy strategy did not find a complete plan; this is not a proof of infeasibility.'])


def metrics(result,snapshot):
    if result['feasibility']!='feasible':return dict(feasible=False,hard_constraint_violations=None,overdue_priority_work=None,weighted_completion=None,production_disruption=None,technician_utilization=None)
    a=result['assignments'];orders={w['id']:w['content'] for w in snapshot['orders']};lo=minute(snapshot['start_utc'])
    return dict(feasible=True,hard_constraint_violations=len(validate(a,snapshot)),
        overdue_priority_work=sum(PRIORITY[orders[r['work_order_id']]['priority']]>=100 and minute(r['end_utc'])>minute(orders[r['work_order_id']]['due_at']) for r in a),
        weighted_completion=sum(PRIORITY[orders[r['work_order_id']]['priority']]*(minute(r['end_utc'])-lo) for r in a),
        production_disruption=sum(r['disruption_cost'] for r in a),
        technician_utilization=round(sum(r['duration_minutes']*len(r['technician_ids']) for r in a)/(len(snapshot['config']['technicians'])*(minute(snapshot['end_utc'])-lo)),4))


def run():
    fixtures=scenarios();cases=[]
    for fixture in fixtures:
        runs={}
        for mode,method in [('earliest_available',greedy),('priority_only',lambda s:greedy(s,True)),('constraint_optimized',solve)]:
            result=method(fixture['snapshot']);runs[mode]=dict(result=result,metrics=metrics(result,fixture['snapshot']))
            assert not runs[mode]['metrics']['hard_constraint_violations'],(fixture['id'],mode)
        cases.append({**fixture,'runs':runs})
    report=dict(timestamp_utc=iso(),solver_version=VERSION,scope='Six authored synthetic software scenarios; not measured industrial downtime or maintenance effectiveness.',
        comparison='Both greedy baselines enforce the same hard constraints and precedence. Arrival/priority order and earliest placement differ; neither backtracks. Failed greedy search is not proof of infeasibility. Utilization is scheduled team-minutes divided by configured technician count times the horizon.',
        fixture_sha256=hashlib.sha256(json.dumps(fixtures,sort_keys=True).encode()).hexdigest(),cases=cases)
    path=ROOT/'docs/evaluation/scheduling-benchmark.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for case in cases:print(case['id'],{k:v['metrics'] for k,v in case['runs'].items()})
    return report


if __name__=='__main__':run()
