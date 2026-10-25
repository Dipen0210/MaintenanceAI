'use client';
import {Suspense,useEffect,useRef,useState} from 'react';
import {useSearchParams} from 'next/navigation';
import {useWorkspace} from '@/components/workspace/WorkspaceContext';
import {operationsRequest,WorkspaceMode} from '@/lib/operations';
import {scannedAsset} from '@/lib/asset-access';
type Detector={detect:(source:HTMLVideoElement)=>Promise<{rawValue:string}[]>};
type DetectorType={new(options:{formats:string[]}):Detector};
function Scanner(){
 const {switchWorkspace}=useWorkspace(),params=useSearchParams();const [value,setValue]=useState(params.get('asset')||''),[error,setError]=useState(''),[busy,setBusy]=useState(false),[camera,setCamera]=useState(false);
 const video=useRef<HTMLVideoElement>(null),stream=useRef<MediaStream|null>(null),frame=useRef<number>(0),generation=useRef(0);
 const requested=params.get('asset');
 useEffect(()=>{if(!requested)return;let active=true;operationsRequest<{asset_id:string;workspace:WorkspaceMode}>(`/asset-access/${encodeURIComponent(requested)}`).then(result=>{if(active)switchWorkspace(result.workspace,`/operations/assets/${encodeURIComponent(result.asset_id)}`);}).catch(e=>{if(active)setError(e.message);});return()=>{active=false;};},[requested,switchWorkspace]);
 function stop(){generation.current++;cancelAnimationFrame(frame.current);stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;setCamera(false);}
 useEffect(()=>()=>{generation.current++;cancelAnimationFrame(frame.current);stream.current?.getTracks().forEach(t=>t.stop());},[]);
 async function open(input:string){setBusy(true);setError('');try{const aid=scannedAsset(input,location.origin),result=await operationsRequest<{asset_id:string;workspace:WorkspaceMode}>(`/asset-access/${encodeURIComponent(aid)}`);stop();switchWorkspace(result.workspace,`/operations/assets/${encodeURIComponent(result.asset_id)}`);}catch(e){setError(e instanceof Error?e.message:'Machine unavailable');}finally{setBusy(false);}}
 async function start(){setError('');const DetectorClass=(window as unknown as {BarcodeDetector?:DetectorType}).BarcodeDetector;if(!DetectorClass||!navigator.mediaDevices?.getUserMedia){setError('Camera scanning is unavailable in this browser. Use your phone’s camera to open a label, or enter its machine ID below.');return;}
  stop();const attempt=generation.current;
  try{const media=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});if(attempt!==generation.current){media.getTracks().forEach(t=>t.stop());return;}stream.current=media;setCamera(true);if(video.current){video.current.srcObject=media;await video.current.play();}const detector=new DetectorClass({formats:['qr_code']});
   async function scan(){if(attempt!==generation.current)return;try{if(video.current&&video.current.readyState>=2){const found=await detector.detect(video.current);if(attempt!==generation.current)return;if(found[0]){stop();setValue(found[0].rawValue);await open(found[0].rawValue);return;}}}catch{stop();setError('Could not read the label. Enter the machine ID below.');return;}frame.current=requestAnimationFrame(scan);}frame.current=requestAnimationFrame(scan);
  }catch{stop();setError('Camera could not be opened. Allow camera access or enter the machine ID below.');}
 }
 return <main><div className="page-heading"><div><span className="eyebrow">Field quick access</span><h1>Scan a machine</h1><p>Open equipment, its sensor map and maintenance history. Your account permissions still apply.</p></div></div><section className="surface intake-panel scan-panel"><h2>{params.get('asset')?'Machine label ready':'Scan or enter a machine ID'}</h2><video ref={video} muted playsInline hidden={!camera} aria-label="Machine QR camera preview"/>{camera?<button className="secondary-button" onClick={stop}>Stop camera</button>:<button className="secondary-button" disabled={busy} onClick={start}>Start camera</button>}<form className="intake-form" onSubmit={e=>{e.preventDefault();stop();void open(value);}}><label>Machine ID or QR link<input required maxLength={500} autoComplete="off" value={value} onChange={e=>setValue(e.target.value)}/></label><button className="primary-button" disabled={busy}>{busy?'Checking access…':'Open machine'}</button></form>{error&&<p role="alert">{error}</p>}<p className="muted">Labels contain a machine link, never a password or access token. A label from another site does not grant access. Localhost labels work only on the computer running this app; phones need a reachable deployed application address.</p></section></main>;
}
export default function ScanPage(){return <Suspense fallback={<p>Loading machine access…</p>}><Scanner/></Suspense>;}
