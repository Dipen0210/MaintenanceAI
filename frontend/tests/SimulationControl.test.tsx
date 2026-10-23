import {act,fireEvent,render,screen,cleanup} from '@testing-library/react';
import {afterEach,beforeEach,expect,it,vi} from 'vitest';
import SimulationControl from '@/components/workspace/SimulationControl';
import {operationsRequest} from '@/lib/operations';

const refresh=vi.hoisted(()=>vi.fn());
vi.mock('@/components/workspace/WorkspaceContext',()=>({useWorkspace:()=>({refresh})}));
vi.mock('@/lib/operations',()=>({operationsRequest:vi.fn()}));
const running={available:true,enabled:true,can_control:true,interval_seconds:10,scenario:'mixed'};
beforeEach(()=>{vi.useFakeTimers();vi.mocked(operationsRequest).mockReset();refresh.mockReset();});
afterEach(()=>{cleanup();vi.useRealTimers();});

it('ignores a delayed running snapshot after the user has switched simulation off',async()=>{
 let resolvePoll!:(value:typeof running)=>void;
 const poll=new Promise<typeof running>(resolve=>{resolvePoll=resolve;});
 vi.mocked(operationsRequest).mockResolvedValueOnce(running).mockReturnValueOnce(poll).mockResolvedValueOnce({...running,enabled:false});
 await act(async()=>{render(<SimulationControl/>);});
 await act(async()=>{await vi.advanceTimersByTimeAsync(5000);});
 await act(async()=>{fireEvent.click(screen.getByRole('switch',{name:'Run simulation'}));});
 expect(screen.getByRole('switch')).toHaveAttribute('aria-checked','false');expect(refresh).toHaveBeenCalledTimes(1);
 await act(async()=>{resolvePoll(running);await poll;});
 expect(screen.getByRole('switch')).toHaveAttribute('aria-checked','false');
 expect(screen.getByText('Simulation paused')).toBeVisible();
});

it('keeps the confirmed running state when the pause request fails',async()=>{
 vi.mocked(operationsRequest).mockResolvedValueOnce(running).mockRejectedValueOnce(new Error('Unable to pause simulation'));
 await act(async()=>{render(<SimulationControl/>);});
 await act(async()=>{fireEvent.click(screen.getByRole('switch'));});
 expect(screen.getByRole('switch')).toHaveAttribute('aria-checked','true');
 expect(screen.getByRole('alert')).toHaveTextContent('Unable to pause simulation');expect(refresh).not.toHaveBeenCalled();
});
