'use client';
import {useCallback,useEffect,useState} from 'react';
import Link from 'next/link';
import {operationsRequest} from '@/lib/operations';

type Prediction={id:string;family:string;model_version:string;dataset_id:string;status:string;reason:string|null;timestamp_utc:string;signal_timestamp:string|null;input_quality:string;data_mode:string;historical:boolean;limitations:string;uncertainty:string;latency_ms:number;output:{fault_class?:string;anomaly_score?:number;is_anomaly?:boolean;rul?:number;units?:string;class_scores?:Record<string,number>;health_score_proxy?:number};evidence:{signal_features?:Record<string,number>;spectrum_peaks?:{frequency_hz:number;amplitude:number}[]};drift:{flagged:boolean;max_z?:number;reason?:string}|null};
export default function PredictivePanel({assetId,role}:{assetId:string;role:string}) {
  const [rows,setRows]=useState<Prediction[]>([]),[error,setError]=useState<string|null>(null),[busy,setBusy]=useState(false);
  const [bindings,setBindings]=useState<{family:string;dataset_id:string}[]>([]);
  const refresh=useCallback(async()=>{const [r,b]=await Promise.all([operationsRequest<Prediction[]>(`/assets/${encodeURIComponent(assetId)}/predictions`),operationsRequest<{bindings:{family:string;dataset_id:string}[]}>(`/assets/${encodeURIComponent(assetId)}/models`)]);return {r,b};},[assetId]);
  useEffect(()=>{let active=true;async function update(){try{const {r,b}=await refresh();if(active){setRows(r);setBindings(b.bindings);setError(null);}}catch(e){if(active)setError(e instanceof Error?e.message:'Predictions unavailable');}}update();const timer=setInterval(update,5000);return()=>{active=false;clearInterval(timer);};},[refresh]);
  async function run(){setBusy(true);try{await operationsRequest(`/assets/${encodeURIComponent(assetId)}/demo-prediction`,{method:'POST',body:JSON.stringify({state:'bearing_impulse_like',seed:50000})});const {r,b}=await refresh();setRows(r);setBindings(b.bindings);setError(null);}catch(e){setError(e instanceof Error?e.message:'Prediction failed');}finally{setBusy(false);}}
  const latest=rows[0];
  return <section aria-label="Predictive evidence" className="rounded border border-slate-700 p-5 space-y-4"><div className="flex flex-wrap justify-between gap-3"><h3 className="text-lg font-semibold">Predictive evidence</h3><Link href="/operations/models" className="text-sky-400">Model registry & monitoring →</Link></div>
    <p className="text-sm text-slate-400">{bindings.length?`Compatible datasets: ${bindings.map(b=>b.dataset_id).join(', ')}`:'No compatible model is approved for this asset. Predictions are unavailable.'} C-MAPSS RUL is restricted to turbofan reference datasets.</p>
    {assetId==='DEMO-CONVEYOR-001'&&['administrator','engineer','reliability_engineer'].includes(role)&&<button disabled={busy} onClick={run} className="rounded bg-sky-800 px-4 py-2 disabled:opacity-50">{busy?'Running trained demo models…':'Analyze synthetic conveyor incident'}</button>}
    {error&&<p role="alert" className="text-red-300">{error}</p>}
    {!latest?<p className="text-slate-400">No predictions recorded.</p>:<article className="space-y-3"><p className="font-medium">{latest.status==='SUCCEEDED'?'Model inference completed':latest.status}: {latest.reason||latest.output.fault_class||'Dataset estimate'}</p><p className="text-sm text-amber-200">{latest.limitations}</p>
      <div className="flex flex-wrap gap-6">{latest.output.anomaly_score!==undefined&&<p>Anomaly percentile: <strong>{(latest.output.anomaly_score*100).toFixed(1)}%</strong> · {latest.output.is_anomaly?'flagged':'not flagged'}</p>}{latest.output.rul!==undefined&&<p>RUL: {latest.output.rul} {latest.output.units}</p>}<p>Input quality: {latest.input_quality}</p><p className={latest.drift?.flagged?'text-amber-300':'text-slate-400'}>{latest.drift?.flagged?'Input distribution warning':'No distribution warning recorded'}</p></div>
      <p className="text-sm text-slate-400">{latest.uncertainty} Predictions are evidence for human review; they do not authorize maintenance.</p>
      {latest.output.class_scores&&<div className="flex flex-wrap gap-4">{Object.entries(latest.output.class_scores).map(([name,score])=><span key={name} className="text-sm">{name.replaceAll('_',' ')}: {(score*100).toFixed(1)}% score</span>)}</div>}
      {latest.evidence.spectrum_peaks&&<div><h4 className="font-medium">Measured signal spectrum</h4><div className="flex flex-wrap gap-3 text-sm text-slate-300">{latest.evidence.spectrum_peaks.map((p,i)=><span key={i}>{p.frequency_hz.toFixed(1)} Hz / {p.amplitude.toFixed(3)}</span>)}</div><p className="text-xs text-slate-400 mt-1">Frequency peaks support inspection; they do not prove a physical root cause.</p></div>}
      <p className="text-xs text-slate-400">{latest.family} · version {latest.model_version} · {latest.latency_ms.toFixed(1)} ms · {new Date(latest.timestamp_utc).toLocaleString()} · {latest.historical?'historical recording':latest.data_mode}</p>
      <details><summary className="cursor-pointer text-sky-400">Prediction history ({rows.length})</summary><ul className="text-sm mt-2 space-y-2">{rows.map(r=><li key={r.id}>{new Date(r.timestamp_utc).toLocaleString()} · {r.status} · {r.output.fault_class||r.reason||r.family} · {r.model_version}</li>)}</ul></details>
    </article>}
  </section>;
}
