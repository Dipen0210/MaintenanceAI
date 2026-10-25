'use client';
import {useEffect,useState,Suspense} from 'react';
import {useParams,useSearchParams,useRouter} from 'next/navigation';
import Link from 'next/link';
import {useWorkspace} from '@/components/workspace/WorkspaceContext';
import Icon from '@/components/workspace/Icon';
import {ConditionBadge} from '@/components/workspace/MachineCondition';
import {operationsRequest,TELEMETRY_WS,Telemetry,Observation,Asset} from '@/lib/operations';
import PredictivePanel from '@/components/operations/PredictivePanel';
import RiskPanel from '@/components/operations/RiskPanel';
import MachineQR from '@/components/operations/MachineQR';
import MachineVisual from '@/components/operations/MachineVisual';
import OperationsOverview from '@/components/operations/OperationsOverview';
function Trend({values,unit}:{values:Observation[];unit:string}){
 if(values.length<2)return <p className="muted">Insufficient history</p>;
 const lo=Math.min(...values.map(v=>v.value)),hi=Math.max(...values.map(v=>v.value));
 const points=values.map((v,i)=>`${i/(values.length-1)*300},${70-(v.value-lo)/(hi-lo||1)*60}`).join(' ');
 return <figure><svg role="img" aria-label={`Recent trend in ${unit}`} viewBox="0 0 300 80"><polyline points={points} fill="none" stroke="currentColor" strokeWidth="2"/></svg><figcaption>{lo.toFixed(2)}–{hi.toFixed(2)} {unit} · {values.length} observations</figcaption></figure>;
}
function Detail(){
 const {id}=useParams<{id:string}>(),params=useSearchParams(),router=useRouter();const {assets,user,loading,openCopilot,workspace}=useWorkspace();
 const scopedAsset=assets.find(a=>a.id===id),tab=['overview','telemetry','predictions','risk'].includes(params.get('tab')||'')?params.get('tab')!:'overview';
 const allowed=Boolean(scopedAsset);
 const [latestAsset,setLatestAsset]=useState<Asset|null>(null);
 const [storedTelemetry,setTelemetry]=useState<Telemetry|null>(null),[error,setError]=useState(''),[stream,setStream]=useState(false);
 useEffect(()=>{if(!allowed)return;let active=true;async function load(){try{const [t,a]=await Promise.all([operationsRequest<Telemetry>(`/assets/${encodeURIComponent(id)}/telemetry`),operationsRequest<Asset>(`/assets/${encodeURIComponent(id)}`)]);if(active){setTelemetry(t);setLatestAsset(a);setError('');}}catch(e){if(active){setTelemetry(null);setError(e instanceof Error?e.message:'Readings unavailable');}}}load();const timer=setInterval(load,5000);return()=>{active=false;clearInterval(timer);};},[id,allowed,workspace]);
 useEffect(()=>{if(!allowed)return;let disposed=false,socket:WebSocket|null=null,timer:ReturnType<typeof setTimeout>|null=null,attempts=0;function connect(){if(disposed)return;socket=new WebSocket(`${TELEMETRY_WS}?workspace=${workspace}`);socket.onopen=()=>{attempts=0;setStream(true);};socket.onerror=()=>setStream(false);socket.onclose=()=>{if(disposed)return;setStream(false);timer=setTimeout(connect,Math.min(30000,1000*2**attempts++));};}connect();return()=>{disposed=true;if(timer)clearTimeout(timer);if(socket){socket.onopen=null;socket.onclose=null;socket.onerror=null;socket.close();}};},[id,allowed,workspace]);
 const telemetry=storedTelemetry?.asset_id===id?storedTelemetry:null;
 const currentAsset=scopedAsset&&(latestAsset?.id===id?latestAsset:scopedAsset);
 const asset=currentAsset&&(error?{...currentAsset,condition:'unknown',condition_label:'Unavailable',condition_detail:'Current sensor evidence cannot be confirmed.'}:currentAsset);
 if(loading)return <p className="empty-state">Loading equipment…</p>;
 if(!asset)return <main className="empty-state"><Icon name="assets" size={32}/><h1>Equipment unavailable</h1><p>This asset does not exist or is outside your access.</p><Link className="secondary-button" href="/operations/assets">Back to assets</Link></main>;
 return <main className="asset-detail"><Link className="back-link" href="/operations/assets">← All assets</Link><div className="page-heading"><div><span className="eyebrow">{asset.area.replaceAll('_',' ')} / {asset.equipment_type.replaceAll('_',' ')}</span><h1>{asset.name}</h1><div className="inline-meta"><span className="badge badge-muted">{asset.data_mode}</span><ConditionBadge asset={asset}/><span>Recorded state: {asset.operating_state}</span><span className={stream?'connection-good':'muted'}>{stream?'Telemetry stream connected':'Stream reconnecting · REST fallback'}</span></div></div><div className="heading-actions"><Link className="secondary-button" href={`/operations/issues?asset=${encodeURIComponent(id)}`}>Report issue</Link><button className="secondary-button" onClick={openCopilot}><Icon name="spark" size={17}/>Investigate with Copilot</button><Link className="primary-button" href={`/operations/scheduling?asset=${encodeURIComponent(id)}&view=new`}>Plan maintenance<Icon name="arrow" size={17}/></Link></div></div>
 <MachineQR key={id} assetId={id} name={asset.name}/>
 <p className="condition-explanation">{asset.condition_detail}</p>
 <nav aria-label="Equipment views" className="page-tabs">{[['overview','Overview'],['telemetry','Sensor readings'],['predictions','Predictions'],['risk','Maintenance risk']].map(([key,label])=><button key={key} aria-current={key===tab?'page':undefined} onClick={()=>router.replace(`/operations/assets/${encodeURIComponent(id)}?tab=${key}`,{scroll:false})}>{label}</button>)}</nav>
 {error&&<p role="alert" className="error-message">{error}. Current readings cannot be confirmed.</p>}
 {(tab==='overview'||tab==='telemetry')&&<><MachineVisual key={id} asset={asset} telemetry={telemetry} canEdit={user.role==='administrator'}/><div className="section-heading"><div><h2>{tab==='telemetry'?'Sensor readings & trends':'Latest readings'}</h2><p>Source measurements with quality and timestamps. No physical cause is inferred.</p></div>{tab==='overview'&&<button className="text-link" onClick={()=>router.replace(`/operations/assets/${encodeURIComponent(id)}?tab=telemetry`,{scroll:false})}>View trends <Icon name="arrow" size={15}/></button>}</div>{!telemetry?<div className="empty-state">{error?'Readings unavailable':'Loading validated observations…'}</div>:<div className="sensor-grid">{telemetry.latest.map(r=><article className="sensor-card" key={r.sensor.tag_id}><div><h3>{r.sensor.tag_id.replaceAll('_',' ')}</h3><span className={`badge ${r.effective_quality==='good'?'badge-teal':'badge-amber'}`}>Quality: {r.effective_quality}</span></div><p className="sensor-value">{r.observation?.value.toFixed(2)??'—'} <span>{r.sensor.unit}</span></p><Trend values={telemetry.history.filter(v=>v.tag_id===r.sensor.tag_id)} unit={r.sensor.unit}/><div className="sensor-provenance"><span>{r.observation?new Date(r.observation.timestamp_utc).toLocaleString():'No observation'}</span><span>Source: {r.observation?.source??'none'}</span></div>{r.last_ingestion&&r.last_ingestion.status!=='accepted'&&<p className="sensor-warning">{r.last_ingestion.status}: {r.last_ingestion.reason}</p>}</article>)}</div>}</>}
 {tab==='overview'&&<><div className="asset-next-step"><div className="flow-icon"><Icon name="model" size={24}/></div><div><h2>Understand this equipment’s condition</h2><p>Review the model evidence, evaluate its risk, then ask Copilot to help investigate.</p></div><Link className="secondary-button" href={`/operations/assets/${encodeURIComponent(id)}?tab=predictions`}>Review predictions <Icon name="arrow" size={17}/></Link></div><OperationsOverview assetId={id}/></>}
 {tab==='predictions'&&<><PredictivePanel assetId={id} role={user.role}/><div className="next-step-footer"><span>Next: assess the maintenance priority using this evidence.</span><Link href={`/operations/assets/${encodeURIComponent(id)}?tab=risk`} className="secondary-button">Review maintenance risk <Icon name="arrow" size={16}/></Link></div></>}
 {tab==='risk'&&<><RiskPanel assetId={id}/><div className="next-step-footer"><span>Need help understanding the assessment?</span><button onClick={openCopilot} className="secondary-button"><Icon name="spark" size={17}/>Investigate with Copilot</button></div></>}
 </main>;
}
export default function AssetDetailPage(){return <Suspense fallback={<p>Loading equipment…</p>}><Detail/></Suspense>;}
