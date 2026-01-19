'use client';

import { useEffect, useState, useMemo } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { HealthGauge, StatCard, MachineCard, CircularProgress, MaintenanceItem } from '@/components/Dashboard';
import { apiClient, MachineStatus, PlantArea, MaintenanceRecommendation } from '@/lib/api';

export default function AreaDetailPage() {
    const params = useParams();
    const areaId = params.id as string;

    const [area, setArea] = useState<PlantArea | null>(null);
    const [machines, setMachines] = useState<MachineStatus[]>([]);
    const [maintenanceQueue, setMaintenanceQueue] = useState<MaintenanceRecommendation[]>([]);
    const [isLive, setIsLive] = useState(false);
    const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [areas, allMachines, queue] = await Promise.all([
                    apiClient.getAreas(),
                    apiClient.getAllMachines(),
                    apiClient.getMaintenanceQueue(),
                ]);

                setArea(areas[areaId] || null);
                setMachines(allMachines.filter(m => m.area === areaId));
                setMaintenanceQueue(queue.filter(q => {
                    const machine = allMachines.find(m => m.machine_id === q.machine_id);
                    return machine?.area === areaId;
                }));
                setIsLive(true);
                setLastUpdate(new Date());
            } catch {
                setIsLive(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, [areaId]);

    // Calculate area stats
    const stats = useMemo(() => {
        const healthy = machines.filter(m => m.health_score >= 0.8).length;
        const warning = machines.filter(m => m.health_score >= 0.5 && m.health_score < 0.8).length;
        const critical = machines.filter(m => m.health_score < 0.5).length;
        const avgHealth = machines.length > 0
            ? machines.reduce((sum, m) => sum + m.health_score, 0) / machines.length
            : 1;
        const avgRul = machines.filter(m => m.rul_cycles).length > 0
            ? machines.filter(m => m.rul_cycles).reduce((sum, m) => sum + (m.rul_cycles || 0), 0) / machines.filter(m => m.rul_cycles).length
            : 0;

        return { healthy, warning, critical, avgHealth, avgRul };
    }, [machines]);

    const formatTime = (date: Date) => {
        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    };

    if (!area) {
        return (
            <div className="min-h-screen gradient-bg flex items-center justify-center">
                <div className="glass rounded-2xl p-8 flex items-center gap-4">
                    <div className="animate-spin rounded-full h-8 w-8 border-2 border-t-blue-500 border-white/20" />
                    <span className="text-gray-400">Loading area...</span>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen gradient-bg">
            {/* Animated Background */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-float" />
                <div className="absolute bottom-20 -left-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '-3s' }} />
            </div>

            {/* Header */}
            <header className="relative z-20 border-b border-white/5 glass-strong sticky top-0">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center gap-4">
                        <Link href="/" className="p-2.5 rounded-xl glass hover:bg-white/10 transition-colors">
                            <span className="text-gray-400">← Back</span>
                        </Link>
                        <div className="flex items-center gap-3 flex-1">
                            <div className="text-3xl">{area.icon}</div>
                            <div>
                                <h1 className="text-2xl font-bold text-white tracking-tight">{area.name}</h1>
                                <p className="text-gray-500 text-sm">{area.description}</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-4">
                            <div className="text-right hidden sm:block">
                                <p className="text-xs text-gray-500">Last Update</p>
                                <p className="text-sm text-gray-300 font-mono">{formatTime(lastUpdate)}</p>
                            </div>
                            <div className={`flex items-center gap-2 px-4 py-2 rounded-full glass
                                ${isLive ? 'ring-1 ring-emerald-500/30' : 'ring-1 ring-amber-500/30'}`}>
                                <span className={`status-dot ${isLive ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                                <span className={`text-sm font-medium ${isLive ? 'text-emerald-400' : 'text-amber-400'}`}>
                                    {isLive ? 'Live' : 'Demo'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            <main className="relative z-10 max-w-7xl mx-auto px-6 py-8 space-y-8">
                {/* Area Summary */}
                <section className="grid grid-cols-2 lg:grid-cols-5 gap-4">
                    <div className="col-span-2 lg:col-span-1 glass rounded-2xl p-6 border border-white/5 flex flex-col items-center justify-center">
                        <CircularProgress value={stats.avgHealth} size={80} strokeWidth={6} />
                        <p className="text-gray-400 text-sm mt-3">Area Health</p>
                    </div>
                    <StatCard title="Total Machines" value={machines.length} color="blue" icon={<span>🔧</span>} />
                    <StatCard title="Healthy" value={stats.healthy} color="green" icon={<span>✓</span>} />
                    <StatCard title="Warning" value={stats.warning} color="yellow" icon={<span>⚠</span>} />
                    <StatCard title="Critical" value={stats.critical} color="red" icon={<span>🚨</span>} />
                </section>

                {/* Additional Stats */}
                <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="glass rounded-2xl p-5 border border-white/5">
                        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Average RUL</p>
                        <p className="text-3xl font-bold text-white">{Math.round(stats.avgRul)}</p>
                        <p className="text-gray-500 text-sm">cycles remaining</p>
                    </div>
                    <div className="glass rounded-2xl p-5 border border-white/5">
                        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Faults Detected</p>
                        <p className="text-3xl font-bold text-white">{machines.filter(m => m.fault_type && m.fault_type !== 'Normal').length}</p>
                        <p className="text-gray-500 text-sm">machines with faults</p>
                    </div>
                    <div className="glass rounded-2xl p-5 border border-white/5">
                        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Anomaly Alerts</p>
                        <p className="text-3xl font-bold text-white">{machines.filter(m => m.anomaly_score > 0.1).length}</p>
                        <p className="text-gray-500 text-sm">above threshold</p>
                    </div>
                </section>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Machines */}
                    <section className="lg:col-span-2">
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <span>⚙️</span> Machines in {area.name}
                        </h2>

                        {machines.length === 0 ? (
                            <div className="glass rounded-2xl p-12 text-center border border-white/5">
                                <div className="text-4xl mb-4">🔍</div>
                                <p className="text-gray-400">No machines in this area</p>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                {machines
                                    .sort((a, b) => a.health_score - b.health_score) // Critical first
                                    .map(machine => (
                                        <Link key={machine.machine_id} href={`/machine/${machine.machine_id}`}>
                                            <MachineCard machine={machine} />
                                        </Link>
                                    ))}
                            </div>
                        )}
                    </section>

                    {/* Area Maintenance */}
                    <section>
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <span>📋</span> Maintenance
                        </h2>

                        <div className="space-y-3">
                            {maintenanceQueue.length === 0 ? (
                                <div className="glass rounded-2xl p-8 text-center border border-white/5">
                                    <div className="text-3xl mb-3">✅</div>
                                    <p className="text-gray-400">All clear</p>
                                    <p className="text-gray-500 text-sm mt-1">No pending tasks</p>
                                </div>
                            ) : (
                                maintenanceQueue.map((item, idx) => (
                                    <MaintenanceItem key={idx} item={item} />
                                ))
                            )}
                        </div>
                    </section>
                </div>
            </main>
        </div>
    );
}
