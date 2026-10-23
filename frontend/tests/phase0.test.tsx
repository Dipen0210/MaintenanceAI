import { act, render, renderHook, screen } from '@testing-library/react';
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { useWebSocket } from '@/hooks/useWebSocket';
import { apiClient, ApiError, MachineStatus } from '@/lib/api';
import Dashboard from '@/app/legacy/page';
import AreaPage from '@/app/area/[id]/page';
import MachinePage from '@/app/machine/[id]/page';

vi.mock('next/navigation', () => ({ useParams: () => ({ id: 'test' }) }));
vi.mock('next/link', () => ({ default: ({children, href}: {children: React.ReactNode,href: string}) => <a href={href}>{children}</a> }));

class Socket {
  static OPEN = 1;
  static CONNECTING = 0;
  static instances: Socket[] = [];
  readyState = 0;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor() { Socket.instances.push(this); }
  send = vi.fn();
  close() { this.readyState = 3; queueMicrotask(() => this.onclose?.()); }
}

const machine = (id: string,rul: number): MachineStatus => ({
  machine_id: id, machine_type: 'pump', area: 'test', anomaly_score: .1,
  fault_type: 'Normal', fault_confidence: .9, rul_cycles: rul, health_score: .7,
  last_updated: '2026-10-20T00:00:00Z',
});

beforeEach(() => { Socket.instances=[]; vi.stubGlobal('WebSocket',Socket); });
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('WebSocket lifecycle', () => {
  it('does not reconnect after cleanup, including a queued close event', async () => {
    vi.useFakeTimers();
    const hook=renderHook(() => useWebSocket());
    const socket=Socket.instances[0];
    const queuedClose=socket.onclose;
    hook.unmount();
    await act(async () => { queuedClose?.(); await vi.advanceTimersByTimeAsync(60000); });
    expect(Socket.instances).toHaveLength(1);
    expect(socket.readyState).toBe(3);
  });
  it('reconnects after a real disconnect and receives updates', async () => {
    vi.useFakeTimers();
    const {result}=renderHook(() => useWebSocket());
    const socket=Socket.instances[0];
    act(() => {socket.readyState=1;socket.onopen?.();});
    expect(result.current.isConnected).toBe(true);
    act(() => socket.onmessage?.({data:JSON.stringify({type:'machine_update',data:machine('a',0)})}));
    expect(result.current.machineUpdates[0].rul_cycles).toBe(0);
    await act(async () => { socket.close(); await vi.advanceTimersByTimeAsync(1001); });
    expect(Socket.instances).toHaveLength(2);
  });
});

it('shows unavailable data instead of a fabricated all-clear dashboard', async () => {
  vi.spyOn(apiClient,'getPlantSummary').mockRejectedValue(new Error('offline'));
  vi.spyOn(apiClient,'getAllMachines').mockResolvedValue([]);
  vi.spyOn(apiClient,'getMaintenanceQueue').mockResolvedValue([]);
  vi.spyOn(apiClient,'getAreas').mockResolvedValue({});
  render(<Dashboard/>);
  expect(await screen.findByText(/Plant data unavailable/)).toBeInTheDocument();
  expect(screen.queryByText('All Systems Operational')).not.toBeInTheDocument();
  expect(screen.queryByText('No maintenance tasks pending')).not.toBeInTheDocument();
});

it('handles a missing machine without invented demo values', async () => {
  vi.spyOn(apiClient,'getMachine').mockRejectedValue(new ApiError(404,'missing'));
  render(<MachinePage/>);
  expect(await screen.findByText('Machine not found')).toBeInTheDocument();
  expect(screen.queryByText('Ball_014')).not.toBeInTheDocument();
});

it('renders zero remaining life as critical', async () => {
  vi.spyOn(apiClient,'getMachine').mockResolvedValue(machine('test',0));
  render(<MachinePage/>);
  const title=await screen.findByText('Remaining Life');
  expect(title.closest('div.relative.p-6')?.className).toContain('border-red-500/20');
  expect(await screen.findByText(/Approximately 0 cycles remaining/)).toBeInTheDocument();
});

it('includes zero remaining life in area averages', async () => {
  vi.spyOn(apiClient,'getAreas').mockResolvedValue({test:{name:'Test area',description:'Test',icon:'X'}});
  vi.spyOn(apiClient,'getAllMachines').mockResolvedValue([machine('a',0),machine('b',100)]);
  vi.spyOn(apiClient,'getMaintenanceQueue').mockResolvedValue([]);
  render(<AreaPage/>);
  const label=await screen.findByText('Average RUL');
  expect(label.parentElement).toHaveTextContent('50');
});

it('finishes loading with an explicit error when area lookup fails', async () => {
  vi.spyOn(apiClient,'getAreas').mockRejectedValue(new Error('offline'));
  vi.spyOn(apiClient,'getAllMachines').mockResolvedValue([]);
  vi.spyOn(apiClient,'getMaintenanceQueue').mockResolvedValue([]);
  render(<AreaPage/>);
  expect(await screen.findByText('Area data unavailable')).toBeInTheDocument();
});
