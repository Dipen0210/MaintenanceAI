"""Read approved OPC UA nodes and forward original measurements to MaintenanceAI.

Run on the plant gateway, not in the browser. No browse, write or method-call API.
"""
import argparse,asyncio,json,logging,os,stat
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from asyncua import Client,ua
from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
from backend.ingestion.adapters import OPCUAReadOnlyAdapter


def validate(config):
    if config.get('ot_owner_approved') is not True:raise ValueError('Record OT-owner approval in the gateway configuration')
    source=config['source_id'];api=urlsplit(config['endpoint']);opc=urlsplit(config['opcua_endpoint'])
    if api.username or api.password or api.query or api.fragment or api.path!=f'/api/gateways/{source}/telemetry':raise ValueError('Use the exact telemetry endpoint without embedded credentials')
    if api.scheme!='https' and not (api.scheme=='http' and api.hostname in ['localhost','127.0.0.1','::1']):raise ValueError('Remote telemetry endpoints require HTTPS')
    if opc.scheme!='opc.tcp' or not opc.hostname or opc.username or opc.password:raise ValueError('Expected an approved opc.tcp endpoint without embedded credentials')
    insecure=config.get('allow_insecure_loopback') is True and opc.hostname in ['localhost','127.0.0.1','::1']
    if not insecure and not all(config.get(k) for k in ['client_certificate','client_private_key','server_certificate']):raise ValueError('Remote OPC UA requires client credentials and a pinned server certificate')
    if not config.get('mappings') or len(config['mappings'])>1000:raise ValueError('Configure between 1 and 1000 approved sensor mappings')
    for mapping in config['mappings']:
        if not all(isinstance(mapping.get(k),str) and mapping[k] for k in ['node_id','asset_id','tag_id','unit']):raise ValueError('Every sensor needs an explicit node, asset, tag and unit')
    if not 1<=config.get('interval_seconds',10)<=3600:raise ValueError('Polling interval must be 1–3600 seconds')
    if not config.get('token','').startswith('mai_gateway_'):raise ValueError('Provide the source-specific gateway credential')
    return insecure


async def forward_once(opc,config,http,last_seen=None):
    validate(config)
    adapter=OPCUAReadOnlyAdapter(opc,config['mappings'],config['source_id'],approved=True)
    records=await adapter.records()
    # Source timestamps must come from OPC UA, never the time of this HTTP call.
    for row in records:row['timestamp_utc']=row['timestamp_utc'].isoformat()
    if last_seen is not None:
        records=[r for r in records if last_seen.get((r['asset_id'],r['tag_id']))!=json.dumps(r,sort_keys=True)]
    if not records:return {'results':[]}
    response=http.post(config['endpoint'],json={'records':records},headers={'Authorization':'Bearer '+config['token']})
    response.raise_for_status()
    report=response.json()
    if last_seen is not None:
        for record,result in zip(records,report['results']):
            if result['status'] in ['accepted','flagged']:last_seen[(record['asset_id'],record['tag_id'])]=json.dumps(record,sort_keys=True)
    return report


async def run(config,once=False):
    insecure=validate(config)
    client=Client(config['opcua_endpoint'],timeout=10)
    if not insecure:
        await client.set_security(SecurityPolicyBasic256Sha256,config['client_certificate'],config['client_private_key'],
                                  server_certificate=config['server_certificate'],mode=ua.MessageSecurityMode.SignAndEncrypt)
    if config.get('username_env'):
        client.set_user(os.environ[config['username_env']]);client.set_password(os.environ[config['password_env']])
    async with client:
        with httpx.Client(timeout=15,follow_redirects=False) as http:
            last_seen={}
            while True:
                report=await forward_once(client,config,http,last_seen)
                counts={key:sum(r['status']==key for r in report['results']) for key in ['accepted','flagged','rejected']}
                print(json.dumps(counts),flush=True)
                if once:return counts
                await asyncio.sleep(config.get('interval_seconds',10))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    # Suppress third-party endpoint details in errors; never print keys/configs.
    logging.disable(logging.CRITICAL)
    try:
        if os.name=='posix' and stat.S_IMODE(args.config.stat().st_mode)&0o077:raise ValueError('Gateway credential files must be private (chmod 600)')
        asyncio.run(run(json.loads(args.config.read_text()),args.once))
    except KeyboardInterrupt:print('Gateway stopped. No equipment control was issued.')
    except Exception as exc:raise SystemExit(f'Gateway unavailable ({type(exc).__name__}). Check the private configuration, certificates and connectivity; no readings were invented.') from None
