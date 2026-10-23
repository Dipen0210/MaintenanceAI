const BASE=process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
export interface User { id:string; username:string; role:string; plant_id:string }
export interface Asset { id:string; name:string; equipment_type:string; area:string; operating_state:string; data_quality:string; data_mode:string; condition?:string; condition_label?:string; condition_detail?:string; last_observed?:string|null }
export interface Observation { timestamp_utc:string; value:number; unit:string; tag_id:string; quality:string; source:string; operating_state:string }
export interface SensorReading { sensor:{tag_id:string;unit:string}; observation:Observation|null; effective_quality:string; last_ingestion:{status:string;reason:string}|null }
export interface Telemetry { asset_id:string;latest:SensorReading[];history:Observation[];server_time:string }
export interface Source { id:string;name:string;kind:string;status:string;read_only:number;last_received?:string|null }

export async function operationsRequest<T>(path:string,options?:RequestInit):Promise<T> {
  const response=await fetch(`${BASE}/api${path}`,{credentials:'include',...options,
    headers:options?.body instanceof FormData?options.headers:{'Content-Type':'application/json',...options?.headers}});
  if(!response.ok) {
    if(response.status===401&&path!=='/auth/login'&&path!=='/auth/me'&&typeof window!=='undefined')window.dispatchEvent(new Event('operations-session-expired'));
    let detail=`Request failed (${response.status})`;
    try {
      const error=await response.json();
      if(typeof error.detail==='string')detail=error.detail;
      else if(Array.isArray(error.detail)) {
        const messages=error.detail.filter((item:unknown):item is {msg:string}=>!!item&&typeof item==='object'&&'msg' in item&&typeof item.msg==='string').map((item:{msg:string})=>item.msg.replace(/^Value error, /,''));
        if(messages.length)detail=messages.join('; ');
      }
    } catch {}
    throw new Error(detail);
  }
  return response.json();
}

export const TELEMETRY_WS=(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/^http/,'ws')+'/api/ws/telemetry';

export async function operationsDownload(path:string):Promise<Blob>{
  const response=await fetch(`${BASE}/api${path}`,{credentials:'include'});
  if(!response.ok)throw new Error(`Report unavailable (${response.status}); verify your evidence access.`);
  return response.blob();
}
