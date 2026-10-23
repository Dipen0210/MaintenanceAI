"""Current authorized sensor-limit observations, never a physical health diagnosis."""
import json
from backend.ingestion.service import latest_telemetry


def describe_asset(conn,user,asset):
    readings=latest_telemetry(conn,user,asset['id'])
    observations=[r['observation'] for r in readings if r['observation']]
    quality='missing' if not observations else 'good' if all(r['effective_quality']=='good' for r in readings) else 'attention'
    # Data mode follows the evidence/mapping, not the plant's demo label.
    sources={r['source'] for r in observations} or {r['sensor']['source_id'] for r in readings}
    kinds={row['kind'] for sid in sources for row in conn.execute('SELECT kind FROM sources WHERE id=?',(sid,))}
    mode='simulation' if kinds=={'simulation'} else 'gateway' if kinds and kinds<={'gateway','opcua','historian'} else 'historical' if kinds=={'file'} else 'mixed' if kinds else 'unconfigured'
    profile=conn.execute('SELECT config FROM asset_profiles WHERE asset_id=? AND active=1',(asset['id'],)).fetchone()
    config=json.loads(profile['config']) if profile else None
    issues=[];level='unknown';label='Awaiting readings';detail='No authorized sensor observations have arrived.'
    if observations:
        level='unknown';label='Needs verification';detail='Configure and review operating limits before interpreting condition.'
        if config:
            unavailable=[];critical=[];warning=[]
            by_tag={r['sensor']['tag_id']:r for r in readings}
            for tag in set(config.get('required_sensors',[]))|set(config['limits']):
                r=by_tag.get(tag);limit=config['limits'].get(tag)
                if not r or not limit or r['effective_quality']!='good' or not r['observation'] or r['last_ingestion'] and r['last_ingestion']['status']!='accepted':
                    unavailable.append(tag);continue
                o=r['observation']
                if o['unit']!=limit['unit'] or o['operating_state'] not in config.get('operating_states',['running']):unavailable.append(tag);continue
                if o['value']>=limit['critical']:critical.append(tag)
                elif o['value']>=limit['warning']:warning.append(tag)
            issues=critical+warning+unavailable
            if critical:level='critical';label='Critical readings';detail='Critical limit exceeded: '+', '.join(critical)+'. Review the equipment.'
            elif warning:level='warning';label='Warning readings';detail='Warning limit exceeded: '+', '.join(warning)+'. Investigate the evidence.'
            elif unavailable:level='unknown';label='Needs verification';detail='Missing, stale, rejected or inapplicable evidence: '+', '.join(unavailable)+'.'
            else:level='normal';label='Within limits';detail='Fresh mapped readings are within configured limits; this does not confirm physical health.'
            if unavailable and (critical or warning):detail+=' Some sensor evidence also needs verification.'
    return {**asset,'data_quality':quality,'data_mode':mode,'condition':level,'condition_label':label,'condition_detail':detail,
            'condition_tags':issues,'last_observed':max((o['timestamp_utc'] for o in observations),default=None)}
