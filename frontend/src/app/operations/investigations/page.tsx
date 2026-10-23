'use client';
import {useWorkspace} from '@/components/workspace/WorkspaceContext';
import CopilotPanel from '@/components/operations/CopilotPanel';
import Icon from '@/components/workspace/Icon';
export default function InvestigationsPage(){
 const {assets,assetId,setAssetId}=useWorkspace();
 return <main className="investigations-page"><div className="page-heading"><div><span className="eyebrow">Evidence-led investigation</span><h1>Investigations</h1><p>Ask about equipment, inspect the sources and save a report for your team.</p></div></div><div className="investigation-context surface"><span className="flow-icon"><Icon name="assets" size={22}/></span><label>Equipment context<select aria-label="Investigation equipment" value={assetId} onChange={e=>setAssetId(e.target.value)}>{assets.map(a=><option key={a.id} value={a.id}>{a.name}</option>)}</select></label><p>Conversations and citations stay linked to this equipment.</p></div>{assetId?<CopilotPanel key={assetId} assetId={assetId}/>:<p className="empty-state">No accessible equipment is available.</p>}</main>;
}
