"""Reproducible loopback-only native deployment with strict artifact preflight and owned-process cleanup."""
import argparse,json,os,platform,shutil,signal,socket,subprocess,sys,time
from pathlib import Path
import httpx
from backend.core.database import ROOT,iso
from backend.ml.registry import catalog,verified_path
from backend.copilot.providers import verified_directory


def preflight():
    if sys.prefix==sys.base_prefix:raise RuntimeError('Run this project with venv/bin/python')
    if os.name!='posix':raise RuntimeError('This native launcher targets the verified macOS/Linux process model')
    node=shutil.which('node')
    if not node:raise RuntimeError('Node.js is required')
    next_cli=ROOT/'frontend/node_modules/next/dist/bin/next'
    if not next_cli.is_file() or not (ROOT/'frontend/.next/BUILD_ID').is_file():raise RuntimeError('Install frontend dependencies and build the production frontend first')
    versions=catalog()
    if not any(v['family']=='conveyor-synthetic' and v['active'] for v in versions):raise RuntimeError('Train the synthetic conveyor model before starting the complete demo')
    for version in versions:verified_path(version)
    languages={}
    for kind in ['embedding','language']:
        _,manifest=verified_directory(kind);languages[kind]=manifest['repository']+'@'+manifest['revision']
    for port in [8000,3000]:
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=.3):raise RuntimeError(f'Port {port} is already in use; existing services will not be terminated')
        except (ConnectionRefusedError,TimeoutError):pass
    return node,next_cli,dict(python=sys.version.split()[0],platform=platform.platform(),predictive_artifacts=len(versions),language_models=languages)


def stop_owned(processes):
    for process in reversed(processes):
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)


def run(database,smoke=False,output=None):
    node,next_cli,evidence=preflight();processes=[];logs=[];started=iso();logdir=ROOT/'artifacts/deployment';logdir.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'MAINTENANCEAI_MODE':'demo','MAINTENANCEAI_DB':str(database.resolve()),'MAINTENANCEAI_LLM_PROVIDER':'local','MAINTENANCEAI_CPU_THREADS':'2'}
    def interrupted(signum,frame):raise KeyboardInterrupt()
    previous={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    try:
        for name,command,cwd in [('backend',[sys.executable,'-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000'],ROOT),
                                 ('frontend',[node,str(next_cli),'start','--hostname','127.0.0.1','--port','3000'],ROOT/'frontend')]:
            log=(logdir/(name+'.log')).open('w');logs.append(log)
            processes.append(subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        deadline=time.monotonic()+120;health=None
        with httpx.Client(timeout=5,trust_env=False) as client:
            while time.monotonic()<deadline:
                if any(p.poll() is not None for p in processes):raise RuntimeError('A service exited; inspect artifacts/deployment logs')
                try:
                    api=client.get('http://127.0.0.1:8000/');web=client.get('http://127.0.0.1:3000/operations')
                    if api.status_code==200 and web.status_code==200:health=api.json();break
                except httpx.HTTPError:pass
                time.sleep(.25)
            if not health:raise RuntimeError('Services did not become ready within 120 seconds')
            if health['status']!='healthy' or health['models_loaded']['total']!=12:raise RuntimeError('Legacy model startup was degraded')
            print('Demo ready: http://localhost:3000/operations (loopback-only services)',flush=True)
            if smoke:
                login=client.post('http://127.0.0.1:8000/api/auth/login',json={'username':'engineer','password':'engineer-demo-password'});login.raise_for_status()
                auth={'Authorization':'Bearer '+login.json()['access_token']}
                statuses={}
                for route in ['/api/assets','/api/planning/config','/api/work-orders','/api/schedules']:
                    response=client.get('http://127.0.0.1:8000'+route,headers=auth);response.raise_for_status();statuses[route]=response.status_code
                statuses['unauthenticated_assets']=httpx.get('http://127.0.0.1:8000/api/assets',trust_env=False).status_code
                if statuses['unauthenticated_assets']!=401:raise RuntimeError('Authentication boundary failed')
                evidence.update(started_at=started,finished_at=iso(),scope='Actual native loopback deployment smoke. No Docker, remote CI, public hosting or real OT connection claimed.',backend_health=health,frontend_status=web.status_code,authenticated_routes=statuses)
            else:
                while all(p.poll() is None for p in processes):time.sleep(.5)
                raise RuntimeError('A demo service exited unexpectedly; inspect artifacts/deployment logs')
    finally:
        stop_owned(processes)
        for log in logs:log.close()
        for sig,handler in previous.items():signal.signal(sig,handler)
    if smoke:
        evidence['owned_processes_stopped']=all(p.poll() is not None for p in processes)
        if output:output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(evidence,indent=2)+'\n')
        print(json.dumps(evidence,indent=2));return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--database',type=Path,default=ROOT/'artifacts/maintenanceai.db')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:run(args.database,args.smoke,args.output)
    except KeyboardInterrupt:print('Owned demo processes stopped; application data preserved.')
