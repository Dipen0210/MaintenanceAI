"""Deterministic minute-resolution CP-SAT scheduling with independent feasibility validation."""
from datetime import datetime,timezone
from itertools import combinations
import math,time
import ortools
from ortools.sat.python import cp_model

VERSION='maintenance-cpsat-v1'
PRIORITY={'Critical review required':1000,'High priority':100,'Insufficient evidence':100,'Planned maintenance':10,'Monitor':1}


def minute(value,ceil=False):
    number=datetime.fromisoformat(value).timestamp()/60 if isinstance(value,str) else value.timestamp()/60
    return math.ceil(number) if ceil else math.floor(number)


def stamp(value):return datetime.fromtimestamp(value*60,timezone.utc).isoformat()


def overlap(a,b):return a[0]<b[1] and b[0]<a[1]


def merged(periods):
    result=[]
    for lo,hi in sorted(periods):
        if lo>=hi:continue
        if result and lo<=result[-1][1]:result[-1]=(result[-1][0],max(hi,result[-1][1]))
        else:result.append((lo,hi))
    return result


def intersections(left,right):
    return merged([(max(a,c),min(b,d)) for a,b in left for c,d in right if max(a,c)<min(b,d)])


def validate(assignments,snapshot):
    """Check a solver/proposal output independently before it can reserve real resources."""
    issues=[];config=snapshot['config'];orders={w['id']:w for w in snapshot['orders']}
    techs={t['id']:t for t in config['technicians']};parts={p['id']:p for p in config['parts']}
    by_id={a['work_order_id']:a for a in assignments};existing=snapshot['existing']
    if len(by_id)!=len(assignments) or set(by_id)!=set(orders):issues.append('Every selected work order must be scheduled exactly once')
    usage={}
    for row in existing:
        for pid,qty in row['parts'].items():usage[pid]=usage.get(pid,0)+qty
    for a in assignments:
        order=orders.get(a['work_order_id'])
        if not order:continue
        data=order['content'];start=minute(a['start_utc'],True);end=minute(a['end_utc'])
        if datetime.fromisoformat(a['start_utc']).timestamp()!=start*60 or datetime.fromisoformat(a['end_utc']).timestamp()!=end*60:
            issues.append('Assignment timestamps must align to the one-minute planning resolution')
        if a['asset_id']!=order['asset_id'] or end-start!=data['duration_minutes']:issues.append('Asset or duration mismatch: '+order['id'])
        if start<minute(snapshot['start_utc'],True) or end>minute(snapshot['end_utc']):issues.append('Assignment is outside the planning horizon')
        team=a['technician_ids']
        if len(team)!=data['team_size'] or len(set(team))!=len(team) or any(t not in techs for t in team):
            issues.append('Invalid technician team: '+order['id']);continue
        if not set(data['required_skills'])<=set().union(*(set(techs[t]['skills']) for t in team)):issues.append('Required skills missing: '+order['id'])
        for t in team:
            availability=merged([(minute(p['start_utc'],True),minute(p['end_utc'])) for p in techs[t]['availability']])
            if not any(lo<=start and end<=hi for lo,hi in availability):issues.append('Technician is unavailable: '+t)
        window=next((w for w in config['equipment_windows'] if w['id']==a['window_id'] and w['asset_id']==order['asset_id']),None)
        if not window or not minute(window['start_utc'],True)<=start<end<=minute(window['end_utc']):issues.append('Equipment maintenance window unavailable: '+order['id'])
        for b in config['blackouts']:
            if b['asset_id']==order['asset_id'] and overlap((start,end),(minute(b['start_utc']),minute(b['end_utc'],True))):issues.append('Operational blackout: '+b['reason'])
        if a['parts']!=data['parts']:issues.append('Required parts differ from reviewed work order')
        for pid,qty in data['parts'].items():
            usage[pid]=usage.get(pid,0)+qty
            if pid not in parts or minute(parts[pid]['available_at'],True)>start:issues.append('Part unavailable at task start: '+pid)
        for dep in data['dependencies']:
            prior=by_id.get(dep) or next((r for r in existing if r['work_order_id']==dep),None)
            if not prior or minute(prior['end_utc'],True)>start:issues.append('Task dependency unmet: '+dep)
    for pid,qty in usage.items():
        if pid not in parts or qty>parts[pid]['quantity']:issues.append('Insufficient unreserved parts: '+pid)
    for i,a in enumerate(assignments):
        for b in assignments[i+1:]+existing:
            if a['asset_id']==b['asset_id'] or set(a['technician_ids'])&set(b['technician_ids']):
                if overlap((minute(a['start_utc']),minute(a['end_utc'],True)),(minute(b['start_utc']),minute(b['end_utc'],True))):issues.append('Equipment or technician overlap')
    return sorted(set(issues))


def solve(snapshot):
    began=time.perf_counter();config=snapshot['config'];orders=sorted(snapshot['orders'],key=lambda w:w['id'])
    lo=minute(snapshot['start_utc'],True);hi=minute(snapshot['end_utc']);horizon=hi-lo
    base={'solver_version':VERSION,'ortools_version':ortools.__version__,'resolution_minutes':1,'assignments':[],
          'conflicts':[],'objective':None,'rationale':'Hard constraints precede all optimization. Weighted completion time, overdue exposure, production disruption and makespan are minimized.',
          'approval_required':True,'simulation_only':config['simulation_only']}
    def finish(status,conflicts=None,**extra):
        return {**base,'feasibility':status,'conflicts':conflicts or [],'elapsed_ms':round((time.perf_counter()-began)*1000,3),**extra}
    if horizon<=0:return finish('infeasible',['No complete scheduling minute exists in this horizon'])
    techs=sorted(config['technicians'],key=lambda t:t['id']);parts={p['id']:p for p in config['parts']};existing=snapshot['existing']
    quantities={}
    for row in existing:
        for pid,qty in row['parts'].items():quantities[pid]=quantities.get(pid,0)+qty
    for order in orders:
        for pid,qty in order['content']['parts'].items():quantities[pid]=quantities.get(pid,0)+qty
    shortages=[f'Insufficient unreserved parts: {pid}; required {qty}, configured {parts.get(pid,{}).get("quantity",0)}' for pid,qty in quantities.items() if pid not in parts or qty>parts[pid]['quantity']]
    if shortages:return finish('infeasible',shortages)
    model=cp_model.CpModel();starts={};ends={};options={};resources={};costs=[];conflicts=[]
    busy={}
    for row in existing:
        for key in ['asset:'+row['asset_id']]+['tech:'+t for t in row['technician_ids']]:busy.setdefault(key,[]).append((minute(row['start_utc'])-lo,minute(row['end_utc'],True)-lo))
    for row in config['blackouts']:busy.setdefault('asset:'+row['asset_id'],[]).append((minute(row['start_utc'])-lo,minute(row['end_utc'],True)-lo))
    for key,periods in busy.items():
        for a,b in merged([(max(0,a),min(horizon,b)) for a,b in periods]):resources.setdefault(key,[]).append(model.new_fixed_size_interval_var(a,b-a,'existing:'+key))
    for order in orders:
        wid=order['id'];data=order['content'];duration=data['duration_minutes']
        if duration>horizon:conflicts.append('Task exceeds horizon: '+wid);continue
        start=model.new_int_var(0,horizon-duration,'start:'+wid);end=model.new_int_var(duration,horizon,'end:'+wid)
        starts[wid]=start;ends[wid]=end;model.add(end==start+duration)
        resources.setdefault('asset:'+order['asset_id'],[]).append(model.new_interval_var(start,duration,end,'asset:'+wid))
        options[wid]=[];required=set(data['required_skills']);eligible=[]
        for team in combinations(techs,data['team_size']):
            if required<=set().union(*(set(t['skills']) for t in team)):eligible.append(team)
        if not eligible:conflicts.append('No technician team covers required skills: '+wid);continue
        earliest=max([lo]+[minute(parts[pid]['available_at'],True) for pid in data['parts']])
        for window in sorted(config['equipment_windows'],key=lambda w:w['id']):
            if window['asset_id']!=order['asset_id']:continue
            for team in eligible:
                periods=[(max(earliest,minute(window['start_utc'],True)),min(hi,minute(window['end_utc'])))]
                for technician in team:
                    periods=intersections(periods,[(minute(p['start_utc'],True),minute(p['end_utc'])) for p in technician['availability']])
                for a,b in periods:
                    if b-a<duration:continue
                    option=model.new_bool_var(f'option:{wid}:{len(options[wid])}')
                    model.add(start>=a-lo).only_enforce_if(option);model.add(end<=b-lo).only_enforce_if(option)
                    interval=model.new_optional_interval_var(start,duration,end,option,'assignment:'+wid)
                    for technician in team:resources.setdefault('tech:'+technician['id'],[]).append(interval)
                    options[wid].append((option,[t['id'] for t in team],window))
                    costs.append(option*duration*window['disruption_cost_per_minute'])
                    if len(options[wid])>2000:return finish('unknown',['Candidate assignment limit exceeded; narrow the planning horizon or resource windows'])
        if not options[wid]:conflicts.append('No common technician, equipment and parts-availability window: '+wid)
        else:model.add_exactly_one([option[0] for option in options[wid]])
        weight=PRIORITY[data['priority']];due=minute(data['due_at'])-lo
        tardy=model.new_int_var(0,max(horizon-due,0),'overdue:'+wid);model.add(tardy>=end-due)
        costs.extend([weight*end,5*weight*tardy])
    if conflicts:return finish('infeasible',conflicts)
    for order in orders:
        for dep in order['content']['dependencies']:
            if dep in ends:model.add(starts[order['id']]>=ends[dep])
            else:
                prior=next((r for r in existing if r['work_order_id']==dep),None)
                if not prior:return finish('infeasible',['Dependency must be included or already scheduled: '+dep])
                model.add(starts[order['id']]>=minute(prior['end_utc'],True)-lo)
    for intervals in resources.values():model.add_no_overlap(intervals)
    makespan=model.new_int_var(0,horizon,'makespan');model.add_max_equality(makespan,list(ends.values()))
    model.minimize(sum(costs)+makespan)
    solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.random_seed=42
    solver.parameters.max_deterministic_time=2;solver.parameters.max_time_in_seconds=15
    status=solver.solve(model)
    if status==cp_model.INFEASIBLE:return finish('infeasible',['No joint assignment satisfies all equipment/technician non-overlap, operational blackout and dependency constraints'])
    if status not in [cp_model.OPTIMAL,cp_model.FEASIBLE]:return finish('unknown',['Solver limit reached without a validated feasible assignment'],solver_status=solver.status_name(status))
    assignments=[]
    for order in orders:
        wid=order['id'];_,team,window=next(option for option in options[wid] if solver.value(option[0]))
        assignments.append({'work_order_id':wid,'asset_id':order['asset_id'],'start_utc':stamp(lo+solver.value(starts[wid])),
                            'end_utc':stamp(lo+solver.value(ends[wid])),'technician_ids':team,'window_id':window['id'],
                            'parts':order['content']['parts'],'duration_minutes':order['content']['duration_minutes'],'priority':order['content']['priority'],
                            'disruption_cost':order['content']['duration_minutes']*window['disruption_cost_per_minute']})
    invalid=validate(assignments,snapshot)
    if invalid:return finish('unknown',['Independent feasibility validation rejected solver output']+invalid)
    return finish('feasible',assignments=assignments,objective=solver.objective_value,optimal=status==cp_model.OPTIMAL,solver_status=solver.status_name(status))
