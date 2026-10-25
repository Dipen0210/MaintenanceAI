import {afterEach,describe,expect,it,vi} from 'vitest';
import {operationsRequest,operationsDownload,setRequestWorkspace} from '@/lib/operations';
afterEach(()=>{vi.unstubAllGlobals();setRequestWorkspace('demo');});
describe('workspace request isolation',()=>{
 it('scopes requests without breaking multipart photos',async()=>{
  const fetcher=vi.fn().mockResolvedValue({ok:true,json:async()=>({id:'issue'})});vi.stubGlobal('fetch',fetcher);
  setRequestWorkspace('plant');const body=new FormData();body.append('file',new Blob(['photo']));
  await operationsRequest('/issues/id/photos',{method:'POST',body});
  const options=fetcher.mock.calls[0][1];expect(options.headers.get('X-Maintenance-Workspace')).toBe('plant');expect(options.headers.has('Content-Type')).toBe(false);expect(options.credentials).toBe('include');
 });
 it('discards a late response from the previous workspace',async()=>{
  let finish!:(value:unknown)=>void;vi.stubGlobal('fetch',vi.fn(()=>new Promise(resolve=>{finish=resolve;})));
  const request=operationsRequest('/assets');setRequestWorkspace('plant');finish({ok:true,json:async()=>[{id:'DEMO'}]});
  await expect(request).rejects.toThrow('Workspace changed');
 });
 it('also rejects a switch while a JSON body or download is being read',async()=>{
  let finish!:(value:unknown)=>void;const json=vi.fn(()=>new Promise(resolve=>{finish=resolve;}));vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json}));
  const request=operationsRequest('/assets');await Promise.resolve();setRequestWorkspace('plant');finish([{id:'DEMO'}]);await expect(request).rejects.toThrow('Workspace changed');
  let blob!:(value:Blob)=>void;vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,blob:()=>new Promise(resolve=>{blob=resolve;})}));
  const download=operationsDownload('/issues/i/photos/p');await Promise.resolve();setRequestWorkspace('demo');blob(new Blob(['private']));await expect(download).rejects.toThrow('Workspace changed');
 });
 it('does not expire the new workspace session for a late unauthorized response',async()=>{
  let finish!:(value:unknown)=>void;const listener=vi.fn();window.addEventListener('operations-session-expired',listener);
  vi.stubGlobal('fetch',vi.fn(()=>new Promise(resolve=>{finish=resolve;})));
  const request=operationsRequest('/assets');setRequestWorkspace('plant');finish({ok:false,status:401});await expect(request).rejects.toThrow('Workspace changed');expect(listener).not.toHaveBeenCalled();window.removeEventListener('operations-session-expired',listener);
 });
});
