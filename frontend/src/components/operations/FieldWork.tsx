'use client';
import OperationsHandover from './OperationsHandover';
import {useState} from 'react';
import {operationsRequest,User} from '@/lib/operations';
import {WorkOrder,workStatusLabel} from '@/lib/planning';

export default function FieldWork({work,user,onUpdated}:{work:WorkOrder;user:User|null;onUpdated:()=>Promise<void>}){
 const [notes,setNotes]=useState(''),[clearance,setClearance]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState(''),[version,setVersion]=useState(work.execution.version);
 if(!['scheduled','in_progress','awaiting_verification','awaiting_return_to_service','completed'].includes(work.status))return null;
 const engineer=user&&['engineer','reliability_engineer'].includes(user.role),manager=user?.role==='maintenance_manager';
 const changed=Boolean(notes)&&version!==work.execution.version,disabled=busy||notes.trim().length<15||changed;
 async function update(action:string){setBusy(true);setError('');setNotice('');try{
  const updated=await operationsRequest<WorkOrder>(`/work-orders/${work.id}/field-update`,{method:'POST',body:JSON.stringify({action,notes,expected_version:version,site_clearance_confirmed:clearance})});
  setVersion(updated.execution.version);setNotes('');setClearance(false);await onUpdated();setNotice(action==='report_fixed'?'Repair reported. Awaiting independent manager verification.':action==='verify'?(updated.status==='awaiting_return_to_service'?'Repair verified. Operations must confirm return to production.':'Work verified complete. Sensor condition remains based on current readings.'):'Field update saved and shared with the authorized team.');
 }catch(e){setError(e instanceof Error?e.message:'Unable to save field update');}finally{setBusy(false);}}
 return <section className="field-work" aria-label="Field work and handover"><div className="field-work-heading"><h4>Field work & team handover</h4><span className={`badge ${work.status==='completed'?'badge-teal':'badge-amber'}`}>{workStatusLabel(work.status)}</span></div>
 <p>Record the work here so your maintenance team can follow progress. A repair report requires manager verification; it never clears sensor alerts or starts equipment.</p>
 {work.execution.window&&<p className="field-window">Approved window: {new Date(work.execution.window.start_utc).toLocaleString(undefined,{timeZone:'UTC'})} → {new Date(work.execution.window.end_utc).toLocaleString(undefined,{timeZone:'UTC'})} UTC</p>}
 {work.content.simulation_only&&<p className="field-window">Demo rehearsal: field steps can be tried before the simulated window. No physical work is recorded.</p>}
 {error&&<p role="alert" className="error-message">{error}</p>}{notice&&<p role="status">{notice}</p>}
 <OperationsHandover work={work} user={user} onUpdated={onUpdated}/>
 {work.status!=='completed'&&(engineer||manager||user?.role==='administrator')&&<><label>Work notes / operations handover<textarea disabled={busy} aria-label="Work notes / operations handover" maxLength={3000} minLength={15} value={notes} onChange={e=>{if(!notes)setVersion(work.execution.version);setNotes(e.target.value);}} placeholder={work.status==='awaiting_verification'?'Record inspection findings, post-repair checks and the verification decision.':'Describe the issue, work performed, parts used, checks and coordination with operations.'}/></label>
 {changed&&<p role="alert">The team added an update. Read the latest history, then <button className="text-link" onClick={()=>setVersion(work.execution.version)}>Review latest update</button> before submitting your notes.</p>}
 {engineer&&work.status==='scheduled'&&<label className="field-clearance"><input type="checkbox" checked={clearance} onChange={e=>setClearance(e.target.checked)}/>I have coordinated with operations and confirmed applicable site permits and isolation.</label>}
 <div className="heading-actions"><button className="secondary-button" disabled={disabled} onClick={()=>update('note')}>Post team update</button>
 {engineer&&work.status==='scheduled'&&<button className="primary-button" disabled={disabled||!clearance||Boolean(work.operations&&work.operations.status!=='released')} onClick={()=>update('start')}>{work.content.simulation_only?'Start demo work':'Record work started'}</button>}
 {engineer&&work.status==='in_progress'&&<button className="primary-button" disabled={disabled} onClick={()=>update('report_fixed')}>Report issue fixed</button>}
 {manager&&work.status==='awaiting_verification'&&<><button className="primary-button" disabled={disabled} onClick={()=>update('verify')}>{work.operations?'Verify repair for operations':'Verify repair & close work'}</button><button className="secondary-button" disabled={disabled} onClick={()=>update('return_to_work')}>Return for further work</button></>}
 {manager&&work.status==='awaiting_return_to_service'&&<button className="secondary-button" disabled={disabled} onClick={()=>update('return_to_work')}>Reopen repair — require fresh release</button>}
 </div></>}
 {work.execution.events.length>0&&<details open className="field-history"><summary>Work history ({work.execution.events.length})</summary><ol>{work.execution.events.map(event=><li key={event.id}><div><strong>{event.action.replaceAll('_',' ')} · {event.actor_id}</strong><time dateTime={event.created_at}>{new Date(event.created_at).toLocaleString()}</time></div><p>{event.notes}</p><small>{workStatusLabel(event.status)}</small></li>)}</ol></details>}
 </section>;
}
