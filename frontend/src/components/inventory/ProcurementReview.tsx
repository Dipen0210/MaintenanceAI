'use client';
import {useState} from 'react';
import {operationsRequest} from '@/lib/operations';
export default function ProcurementReview({endpoint,version,actions,onUpdated,confirmation='I checked this record against its supporting evidence.'}:{endpoint:string;version:number;actions:{value:string;label:string}[];onUpdated:()=>Promise<void>;confirmation?:string}){
 const [reason,setReason]=useState(''),[verified,setVerified]=useState(false),[base,setBase]=useState(version),[busy,setBusy]=useState(false),[error,setError]=useState('');
 async function act(action:string){setBusy(true);setError('');try{await operationsRequest(endpoint,{method:'POST',body:JSON.stringify({action,expected_version:base,reason,verified})});setReason('');setVerified(false);await onUpdated();}catch(e){setError(e instanceof Error?e.message:'Review failed');}finally{setBusy(false);}}
 return <div className="intake-form"><label>Review reason<textarea minLength={15} maxLength={500} value={reason} onChange={e=>{if(!reason)setBase(version);setReason(e.target.value);}}/></label><label className="field-clearance"><input type="checkbox" checked={verified} onChange={e=>setVerified(e.target.checked)}/>{confirmation}</label>{reason&&base!==version&&<p role="alert">This record changed. Clear the reason and review its latest version.</p>}<div className="heading-actions">{actions.map(a=><button key={a.value} type="button" className="secondary-button" disabled={busy||reason.trim().length<15||base!==version} onClick={()=>act(a.value)}>{a.label}</button>)}</div>{error&&<p role="alert">{error}</p>}</div>;
}
