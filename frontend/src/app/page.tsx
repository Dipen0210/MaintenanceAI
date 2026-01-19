'use client';

import { useEffect, useState, useMemo } from 'react';
import { StatCard, CircularProgress, AlertBadge } from '@/components/Dashboard';
import { apiClient, PlantSummary, MachineStatus, MaintenanceRecommendation, PlantArea } from '@/lib/api';
import { useWebSocket } from '@/hooks/useWebSocket';
import Link from 'next/link';

// Demo data
const demoPlantSummary: PlantSummary = {
  total_machines: 14,
  healthy_count: 10,
  warning_count: 2,
  critical_count: 2,
  average_health: 0.76,
  machines_needing_attention: ['PLA-PUMP-002', 'HVAC-FAN-001'],
};

const demoAreas: Record<string, PlantArea> = {
  production_line_a: { name: 'Production Line A', icon: '🏭', description: 'Main production floor' },
  production_line_b: { name: 'Production Line B', icon: '🏭', description: 'Secondary production' },
  hvac_system: { name: 'HVAC System', icon: '❄️', description: 'Climate control' },
  utilities: { name: 'Utilities', icon: '⚡', description: 'Power & water systems' },
};

const demoMachines: MachineStatus[] = [];
const demoMaintenanceQueue: MaintenanceRecommendation[] = [];

export default function DashboardPage() {
  const [plantSummary, setPlantSummary] = useState<PlantSummary>(demoPlantSummary);
  const [machines, setMachines] = useState<MachineStatus[]>(demoMachines);
  const [areas, setAreas] = useState<Record<string, PlantArea>>(demoAreas);
  const [maintenanceQueue, setMaintenanceQueue] = useState<MaintenanceRecommendation[]>(demoMaintenanceQueue);
  const [isLive, setIsLive] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // WebSocket for real-time updates
  const { isConnected: wsConnected, plantSummary: wsSummary, alerts: wsAlerts } = useWebSocket();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [summary, machineList, queue, areaData] = await Promise.all([
          apiClient.getPlantSummary(),
          apiClient.getAllMachines(),
          apiClient.getMaintenanceQueue(),
          apiClient.getAreas(),
        ]);
        setPlantSummary(summary);
        setMachines(machineList);
        setMaintenanceQueue(queue);
        setAreas(areaData);
        setIsLive(true);
        setLastUpdate(new Date());
      } catch (error) {
        console.log('Using demo data');
        setIsLive(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Get critical and warning machines
  const criticalMachines = useMemo(() =>
    machines.filter(m => m.health_score < 0.5).sort((a, b) => a.health_score - b.health_score),
    [machines]);

  const warningMachines = useMemo(() =>
    machines.filter(m => m.health_score >= 0.5 && m.health_score < 0.8),
    [machines]);

  // Calculate area stats
  const areaStats = useMemo(() => {
    const stats: Record<string, { total: number; healthy: number; warning: number; critical: number; avgHealth: number }> = {};
    Object.keys(areas).forEach(areaKey => {
      const areaMachines = machines.filter(m => m.area === areaKey);
      const healthy = areaMachines.filter(m => m.health_score >= 0.8).length;
      const warning = areaMachines.filter(m => m.health_score >= 0.5 && m.health_score < 0.8).length;
      const critical = areaMachines.filter(m => m.health_score < 0.5).length;
      const avgHealth = areaMachines.length > 0
        ? areaMachines.reduce((sum, m) => sum + m.health_score, 0) / areaMachines.length
        : 1;
      stats[areaKey] = { total: areaMachines.length, healthy, warning, critical, avgHealth };
    });
    return stats;
  }, [machines, areas]);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const getAreaForMachine = (machineId: string): PlantArea | undefined => {
    const machine = machines.find(m => m.machine_id === machineId);
    return machine ? areas[machine.area] : undefined;
  };

  return (
    <div className="min-h-screen gradient-bg">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-float" />
        <div className="absolute top-1/2 -left-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '-3s' }} />
        <div className="absolute -bottom-40 right-1/3 w-80 h-80 bg-emerald-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '-6s' }} />
      </div>

      {/* Header */}
      <header className="relative z-20 border-b border-white/5 glass-strong sticky top-0">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="text-3xl">🏭</div>
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">Plant Overview</h1>
                <p className="text-gray-500 text-sm">AI-Powered Predictive Maintenance</p>
              </div>
            </div>
            <div className="flex items-center gap-6">
              <div className="text-right hidden sm:block">
                <p className="text-xs text-gray-500 uppercase tracking-wider">Last Update</p>
                <p className="text-sm text-gray-300 font-mono">{formatTime(lastUpdate)}</p>
              </div>
              {/* WebSocket indicator */}
              <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full glass text-xs
                ${wsConnected ? 'ring-1 ring-blue-500/30' : 'ring-1 ring-gray-500/30'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${wsConnected ? 'bg-blue-500 animate-pulse' : 'bg-gray-500'}`} />
                <span className={wsConnected ? 'text-blue-400' : 'text-gray-500'}>
                  {wsConnected ? 'WS' : 'WS Off'}
                </span>
              </div>
              <div className={`flex items-center gap-2 px-4 py-2 rounded-full glass
                ${isLive ? 'ring-1 ring-emerald-500/30' : 'ring-1 ring-amber-500/30'}`}>
                <span className={`status-dot ${isLive ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                <span className={`text-sm font-medium ${isLive ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {isLive ? 'Live' : 'Demo Mode'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8 space-y-8">

        {/* Critical Alerts Banner */}
        {criticalMachines.length > 0 && (
          <section className="p-4 rounded-2xl bg-red-500/10 border border-red-500/30 animate-pulse">
            <div className="flex items-center gap-3 mb-3">
              <span className="text-2xl">🚨</span>
              <h2 className="text-lg font-bold text-red-400">Critical Alerts</h2>
              <span className="bg-red-500 text-white text-xs px-2 py-1 rounded-full">{criticalMachines.length}</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {criticalMachines.slice(0, 6).map(machine => {
                const area = getAreaForMachine(machine.machine_id);
                return (
                  <Link key={machine.machine_id} href={`/machine/${machine.machine_id}`}
                    className="flex items-center gap-3 p-3 rounded-xl bg-red-500/10 border border-red-500/20 hover:bg-red-500/20 transition-colors">
                    <div className="text-xl">{area?.icon || '⚙️'}</div>
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-white truncate">{machine.machine_id}</p>
                      <p className="text-xs text-gray-400">{area?.name || 'Unknown Area'}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-red-400 font-bold">{Math.round(machine.health_score * 100)}%</p>
                      <p className="text-xs text-gray-500">health</p>
                    </div>
                  </Link>
                );
              })}
            </div>
          </section>
        )}

        {/* Warning Alerts */}
        {warningMachines.length > 0 && (
          <section className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30">
            <div className="flex items-center gap-3 mb-3">
              <span className="text-2xl">⚠️</span>
              <h2 className="text-lg font-semibold text-amber-400">Warnings</h2>
              <span className="bg-amber-500 text-black text-xs px-2 py-1 rounded-full">{warningMachines.length}</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {warningMachines.slice(0, 8).map(machine => {
                const area = getAreaForMachine(machine.machine_id);
                return (
                  <Link key={machine.machine_id} href={`/machine/${machine.machine_id}`}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20 hover:bg-amber-500/20 transition-colors">
                    <span className="text-sm">{area?.icon || '⚙️'}</span>
                    <span className="text-white text-sm">{machine.machine_id}</span>
                    <span className="text-amber-400 text-sm font-medium">{Math.round(machine.health_score * 100)}%</span>
                  </Link>
                );
              })}
            </div>
          </section>
        )}

        {/* No Alerts */}
        {criticalMachines.length === 0 && warningMachines.length === 0 && (
          <section className="p-6 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 text-center">
            <span className="text-4xl">✅</span>
            <h2 className="text-lg font-semibold text-emerald-400 mt-2">All Systems Operational</h2>
            <p className="text-gray-400 text-sm">No critical or warning alerts at this time</p>
          </section>
        )}

        {/* Plant Overview Stats */}
        <section>
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span>📊</span> Plant Statistics
          </h2>
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <div className="col-span-2 lg:col-span-1 glass rounded-2xl p-6 border border-white/5 flex flex-col items-center justify-center">
              <CircularProgress value={plantSummary.average_health} size={90} strokeWidth={8} />
              <p className="text-gray-400 text-sm mt-3">Overall Health</p>
            </div>
            <StatCard title="Total Machines" value={plantSummary.total_machines} subtitle={`${Object.keys(areas).length} areas`} color="blue" icon={<span>🔧</span>} />
            <StatCard title="Healthy" value={plantSummary.healthy_count} subtitle={`${plantSummary.total_machines > 0 ? Math.round((plantSummary.healthy_count / plantSummary.total_machines) * 100) : 0}%`} color="green" icon={<span>✓</span>} />
            <StatCard title="Warning" value={plantSummary.warning_count} color="yellow" icon={<span>⚠</span>} />
            <StatCard title="Critical" value={plantSummary.critical_count} color="red" icon={<span>🚨</span>} />
          </div>
        </section>

        {/* Plant Areas */}
        <section>
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span>📍</span> Plant Areas
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(areas).map(([areaKey, area]) => {
              const stats = areaStats[areaKey] || { total: 0, healthy: 0, warning: 0, critical: 0, avgHealth: 1 };
              const hasCritical = stats.critical > 0;
              const hasWarning = stats.warning > 0;

              return (
                <Link key={areaKey} href={`/area/${areaKey}`}
                  className={`glass rounded-2xl p-5 border card-hover ${hasCritical ? 'border-red-500/30 hover:border-red-500/50' :
                    hasWarning ? 'border-amber-500/30 hover:border-amber-500/50' :
                      'border-white/5 hover:border-white/10'
                    }`}>
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="text-3xl">{area.icon}</div>
                      <div>
                        <h3 className="font-semibold text-white">{area.name}</h3>
                        <p className="text-xs text-gray-500">{stats.total} machines</p>
                      </div>
                    </div>
                    <CircularProgress value={stats.avgHealth} size={44} strokeWidth={3} />
                  </div>

                  {/* Status indicators */}
                  <div className="flex items-center gap-3 text-sm">
                    <div className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-500" />
                      <span className="text-gray-400">{stats.healthy}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-amber-500" />
                      <span className="text-gray-400">{stats.warning}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-red-500" />
                      <span className="text-gray-400">{stats.critical}</span>
                    </div>
                  </div>

                  {/* Alert indicator */}
                  {hasCritical && (
                    <div className="mt-3 pt-3 border-t border-white/5">
                      <span className="text-xs text-red-400 flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                        {stats.critical} critical
                      </span>
                    </div>
                  )}
                </Link>
              );
            })}
          </div>
        </section>

        {/* Maintenance Queue */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <span>📋</span> Maintenance Queue
            </h2>
            {maintenanceQueue.filter(m => m.priority === 'critical').length > 0 && (
              <span className="text-xs bg-red-500/20 text-red-400 px-3 py-1.5 rounded-full ring-1 ring-red-500/30">
                {maintenanceQueue.filter(m => m.priority === 'critical').length} critical
              </span>
            )}
          </div>

          {maintenanceQueue.length === 0 ? (
            <div className="glass rounded-2xl p-8 text-center border border-white/5">
              <div className="text-3xl mb-3">✅</div>
              <p className="text-gray-400">No maintenance tasks pending</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {maintenanceQueue.slice(0, 6).map((item, idx) => {
                const machine = machines.find(m => m.machine_id === item.machine_id);
                const area = machine ? areas[machine.area] : undefined;

                return (
                  <Link key={idx} href={`/machine/${item.machine_id}`}
                    className={`p-4 rounded-xl glass border-l-4 hover:bg-white/[0.03] transition-colors ${item.priority === 'critical' ? 'border-l-red-500 bg-red-500/5' :
                      item.priority === 'high' ? 'border-l-orange-500 bg-orange-500/5' :
                        item.priority === 'medium' ? 'border-l-amber-500 bg-amber-500/5' :
                          'border-l-emerald-500 bg-emerald-500/5'
                      }`}>
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span>{area?.icon || '⚙️'}</span>
                        <div>
                          <p className="font-semibold text-white">{item.machine_id}</p>
                          <p className="text-xs text-gray-500">{area?.name || 'Unknown'}</p>
                        </div>
                      </div>
                      <AlertBadge priority={item.priority} />
                    </div>
                    <p className="text-sm text-gray-300 truncate">{item.action}</p>
                  </Link>
                );
              })}
            </div>
          )}
        </section>

        {/* Footer */}
        <footer className="pt-8 border-t border-white/5">
          <div className="flex items-center justify-between text-xs text-gray-600">
            <p>Predictive Maintenance AI System v1.0</p>
            <p>Powered by Machine Learning</p>
          </div>
        </footer>
      </main>
    </div>
  );
}
