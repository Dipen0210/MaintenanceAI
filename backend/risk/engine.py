"""Deterministic risk proxy. Class scores are never treated as failure probabilities."""
from datetime import datetime

PRIORITIES={'Critical review required':0,'High priority':1,'Insufficient evidence':2,'Planned maintenance':3,'Monitor':4}


def assess(profile,policy,readings,predictions,now):
    missing=[];notes=[];evidence=[];sources=set();levels=[];critical=False
    by_tag={r['sensor']['tag_id']:r for r in readings}
    for tag in profile['required_sensors']:
        r=by_tag.get(tag);limit=profile['limits'][tag]
        if not r or not r['observation'] or r['effective_quality']!='good':
            missing.append(f'{tag}: '+(r['effective_quality'] if r else 'missing'));continue
        if r.get('last_ingestion') and r['last_ingestion']['status']=='rejected':
            missing.append(f'{tag}: latest ingestion rejected; verify sensor evidence');continue
        obs=r['observation']
        if obs.get('operating_state','running') not in profile.get('operating_states',['running']):
            missing.append(f'{tag}: operating state outside approved limits');continue
        if obs['unit']!=limit['unit']:
            missing.append(f'{tag}: incompatible unit');continue
        value=obs['value'];level=0.0
        if value>=limit['critical']:level=1.;critical=True
        elif value>=limit['warning']:level=.5+.5*(value-limit['warning'])/(limit['critical']-limit['warning'])
        levels.append(level);sources.add(obs['source'])
        evidence.append({'type':'telemetry','id':obs['id'],'tag':tag,'value':value,'unit':obs['unit'],
                         'timestamp_utc':obs['timestamp_utc'],'source_id':obs['source'],'warning':limit['warning'],'critical':limit['critical']})
        notes.append(f'{tag}={value:g} {obs["unit"]}; approved warning/critical limits {limit["warning"]:g}/{limit["critical"]:g}.')
    model=None;seen_families=set()
    for p in predictions:
        family=p.get('family',p['model_id'])
        if family in seen_families:continue
        seen_families.add(family)
        if p['status']!='SUCCEEDED' or p['historical']:continue
        stamp=p.get('signal_timestamp')
        if not stamp or (now-datetime.fromisoformat(stamp)).total_seconds()>policy['prediction_max_age_seconds']:continue
        if p.get('drift') and p['drift']['flagged']:
            notes.append('Model input distribution warning: this prediction is excluded from the risk proxy.');continue
        model=p;break
    if model:
        output=model['output'];anomaly=output.get('anomaly_score')
        model_level=float(anomaly) if output.get('is_anomaly') and anomaly is not None else .75 if output.get('fault_class') not in [None,'normal','Normal'] else 0.
        levels.append(model_level)
        evidence.append({'type':'prediction','id':model['id'],'model_id':model['model_id'],'timestamp_utc':model['timestamp_utc'],
                         'input_id':model['input_id'],'data_mode':model['data_mode']})
        if model.get('source_id'):sources.add(model['source_id'])
        notes.append('Compatible model evidence contributes a degradation proxy; uncalibrated scores are not failure probabilities.')
    else:notes.append('No current compatible prediction contributes; assessment uses available telemetry only. RUL is not invented.')
    degradation=max(levels,default=0.)
    consequence=sum(profile[key]*policy[key+'_weight']/5 for key in ['safety','production','maintenance','damage'])
    score=round(100*degradation*consequence,2)
    urgent=critical and profile['safety']>=4
    if urgent:priority='Critical review required'
    elif missing:priority='Insufficient evidence';score=None
    elif score>=policy['critical_score']:priority='Critical review required'
    elif score>=policy['high_score']:priority='High priority'
    elif score>=policy['planned_score'] or (model and degradation>=.95):priority='Planned maintenance'
    else:priority='Monitor'
    notes.append(f'Degradation proxy {degradation:.3f} × weighted failure consequence {consequence:.3f} × 100. Policy defines priority thresholds; this is not predicted failure likelihood.')
    if missing:notes.append('Missing critical channels prevent a complete assessment; a low-risk conclusion is not supported.')
    if missing:score=None
    suspected=(model['output'].get('fault_class') if model else None) or ('Operating limit excursion' if degradation else 'No observed limit excursion')
    if priority=='Insufficient evidence':suspected='Condition cannot be established from available evidence'
    action={'Critical review required':'Request immediate maintenance-manager review under the configured site escalation procedure.',
            'High priority':'Propose an inspection for qualified human review at the earliest approved maintenance window.',
            'Planned maintenance':'Propose a planned inspection for maintenance-manager review.',
            'Monitor':'Continue evidence collection and periodic human review; this assessment does not authorize equipment operation.',
            'Insufficient evidence':'Request sensor/data verification before ranking maintenance urgency.'}[priority]
    return dict(priority=priority,rank=PRIORITIES[priority],score=score,suspected_issue=suspected,
                severity='critical limit excursion' if critical else 'elevated evidence' if degradation else 'not established' if missing else 'no excursion observed',
                components={'degradation_proxy':degradation,'failure_consequence':round(consequence,4),'safety':profile['safety'],
                            'production':profile['production'],'maintenance':profile['maintenance'],'equipment_damage':profile['damage'],
                            'time_sensitivity':'immediate review' if urgent else 'review at approved window',
                            'data_completeness':round((len(profile['required_sensors'])-len(missing))/len(profile['required_sensors']),3),
                            'production_dependencies':profile['dependencies']},
                evidence=evidence,model_output=model['output'] if model else None,criticality={k:profile[k] for k in ['safety','production','maintenance','damage']},
                explanation=notes,missing_evidence=missing,confidence='insufficient critical observations' if missing else 'uncalibrated model and telemetry evidence' if model else 'telemetry evidence only',
                required_verification=['Verify sensor quality, units and operating state with qualified personnel.',
                                       'Confirm the suspected issue against applicable approved documentation before intervention.',
                                       'Any physical inspection requires approved site safety and isolation procedures.'],
                suggested_action=action,human_review_required=True,escalation=profile['escalation'] if urgent or (missing and profile['safety']>=4) else None,
                source_ids=sorted(sources))
