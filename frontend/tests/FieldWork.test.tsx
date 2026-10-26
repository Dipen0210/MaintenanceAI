import {act,fireEvent,render,screen,cleanup} from '@testing-library/react';
import {afterEach,expect,it,vi} from 'vitest';
import FieldWork from '@/components/operations/FieldWork';
import {operationsRequest} from '@/lib/operations';
import {WorkOrder} from '@/lib/planning';
vi.mock('@/lib/operations',()=>({operationsRequest:vi.fn()}));
afterEach(()=>{cleanup();vi.resetAllMocks();});
it('prevents new field notes being entered while an earlier save can still clear the form',async()=>{
 let complete!:(value:WorkOrder)=>void;const pending=new Promise<WorkOrder>(resolve=>{complete=resolve;});vi.mocked(operationsRequest).mockReturnValueOnce(pending);
 const work={id:'recorded-work',status:'scheduled',operations:null,execution:{version:0,status:null,updated_at:null,window:null,events:[]},content:{simulation_only:true}} as unknown as WorkOrder;
 const onUpdated=vi.fn().mockResolvedValue(undefined);
 render(<FieldWork work={work} user={{id:'engineer',username:'engineer',role:'engineer',plant_id:'demo-plant'}} onUpdated={onUpdated}/>);
 const notes=screen.getByLabelText('Work notes / operations handover');fireEvent.change(notes,{target:{value:'Recorded site clearance for the synthetic inspection.'}});fireEvent.click(screen.getByRole('checkbox'));
 await act(async()=>{fireEvent.click(screen.getByRole('button',{name:'Start demo work'}));});
 expect(notes).toBeDisabled();expect(screen.getByRole('button',{name:'Post team update'})).toBeDisabled();
 await act(async()=>{complete({...work,status:'in_progress',execution:{...work.execution,version:1}});await pending;});
 expect(notes).toBeEnabled();expect(notes).toHaveValue('');expect(onUpdated).toHaveBeenCalledOnce();
});
