'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { HealthGauge, StatCard, CircularProgress } from '@/components/Dashboard';
import { apiClient, MachineStatus } from '@/lib/api';

// Demo data
const getDemoMachine = (id: string): MachineStatus => ({
    machine_id: id,
    machine_type: id.toLowerCase().includes('fan') ? 'fan' :
        id.toLowerCase().includes('pump') ? 'pump' :
            id.toLowerCase().includes('valve') ? 'valve' : 'bearing',
    anomaly_score: 0.15,
    fault_type: 'Ball_014',
    fault_confidence: 0.85,
    rul_cycles: 45,
    health_score: 0.45,
    last_updated: new Date().toISOString(),
});

const machineIcons: Record<string, string> = {
    fan: '🌀',
    pump: '⚙️',
    valve: '🔧',
    bearing: '⭕',
};

export default function MachineDetailPage() {
    const params = useParams();
    const machineId = params.id as string;

    const [machine, setMachine] = useState<MachineStatus | null>(null);
    const [isLive, setIsLive] = useState(false);

    useEffect(() => {
        const fetchMachine = async () => {
            try {
                const data = await apiClient.getMachine(machineId);
                setMachine(data);
                setIsLive(true);
            } catch {
                setMachine(getDemoMachine(machineId));
                setIsLive(false);
            }
        };

        fetchMachine();
        const interval = setInterval(fetchMachine, 3000);
        return () => clearInterval(interval);
    }, [machineId]);

    if (!machine) {
        return (
            <div className="min-h-screen gradient-bg flex items-center justify-center">
                <div className="glass rounded-2xl p-8 flex items-center gap-4">
                    <div className="animate-spin rounded-full h-8 w-8 border-2 border-t-blue-500 border-white/20" />
                    <span className="text-gray-400">Loading machine data...</span>
                </div>
            </div>
        );
    }

    const getStatusConfig = () => {
        if (machine.health_score >= 0.8) return {
            status: 'Healthy',
            color: 'text-emerald-400',
            bg: 'bg-emerald-500/10',
            border: 'border-emerald-500/30',
            icon: '✓'
        };
        if (machine.health_score >= 0.5) return {
            status: 'Warning',
            color: 'text-amber-400',
            bg: 'bg-amber-500/10',
            border: 'border-amber-500/30',
            icon: '⚠'
        };
        if (machine.health_score >= 0.2) return {
            status: 'Critical',
            color: 'text-orange-400',
            bg: 'bg-orange-500/10',
            border: 'border-orange-500/30',
            icon: '⚡'
        };
        return {
            status: 'Failing',
            color: 'text-red-400',
            bg: 'bg-red-500/10',
            border: 'border-red-500/30',
            icon: '✕'
        };
    };

    const statusConfig = getStatusConfig();

    return (
        <div className="min-h-screen gradient-bg">
            {/* Animated Background */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-float" />
                <div className="absolute bottom-20 -left-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '-3s' }} />
            </div>

            {/* Header */}
            <header className="relative z-20 border-b border-white/5 glass-strong sticky top-0">
                <div className="max-w-5xl mx-auto px-6 py-4">
                    <div className="flex items-center gap-4">
                        <Link
                            href="/"
                            className="p-2.5 rounded-xl glass hover:bg-white/10 transition-colors"
                        >
                            <span className="text-gray-400">←</span>
                        </Link>
                        <div className="flex items-center gap-3 flex-1">
                            <div className="text-3xl">{machineIcons[machine.machine_type] || '🔩'}</div>
                            <div>
                                <h1 className="text-2xl font-bold text-white tracking-tight">{machine.machine_id}</h1>
                                <p className="text-gray-500 text-sm capitalize">{machine.machine_type}</p>
                            </div>
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
            </header>

            <main className="relative z-10 max-w-5xl mx-auto px-6 py-8 space-y-6">
                {/* Status Banner */}
                <div className={`p-6 rounded-2xl ${statusConfig.bg} border ${statusConfig.border} glass`}>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-6">
                            <CircularProgress value={machine.health_score} size={80} strokeWidth={6} />
                            <div>
                                <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Current Status</p>
                                <h2 className={`text-3xl font-bold ${statusConfig.color} flex items-center gap-2`}>
                                    <span>{statusConfig.icon}</span>
                                    {statusConfig.status}
                                </h2>
                            </div>
                        </div>
                        <div className="text-right hidden sm:block">
                            <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Last Updated</p>
                            <p className="text-white font-mono text-sm">
                                {new Date(machine.last_updated).toLocaleString()}
                            </p>
                        </div>
                    </div>
                </div>

                {/* Health Details */}
                <div className="p-6 rounded-2xl glass border border-white/5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <span>📊</span> Health Analysis
                    </h3>
                    <HealthGauge value={machine.health_score} size="lg" />
                    <p className="text-gray-500 text-sm mt-4">
                        Health score is calculated from anomaly detection, fault diagnosis, and remaining useful life predictions using AI models.
                    </p>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <StatCard
                        title="Anomaly Score"
                        value={(machine.anomaly_score * 100).toFixed(1) + '%'}
                        subtitle={machine.anomaly_score > 0.1 ? 'Above threshold ⚠️' : 'Normal range ✓'}
                        color={machine.anomaly_score > 0.1 ? 'red' : 'green'}
                    />
                    <StatCard
                        title="Remaining Life"
                        value={machine.rul_cycles !== null ? `${Math.round(machine.rul_cycles)}` : 'N/A'}
                        subtitle="Cycles remaining"
                        color={machine.rul_cycles && machine.rul_cycles < 50 ? 'red' : machine.rul_cycles && machine.rul_cycles < 80 ? 'yellow' : 'green'}
                    />
                    <StatCard
                        title="Fault Status"
                        value={machine.fault_type || 'None'}
                        subtitle={machine.fault_confidence > 0 ? `${(machine.fault_confidence * 100).toFixed(0)}% confidence` : 'No fault detected'}
                        color={machine.fault_type && machine.fault_type !== 'Normal' ? 'red' : 'green'}
                    />
                </div>

                {/* Recommendations */}
                <div className="p-6 rounded-2xl glass border border-white/5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <span>💡</span> AI Recommendations
                    </h3>
                    <div className="space-y-3">
                        {machine.health_score < 0.5 && (
                            <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20">
                                <p className="text-red-400 font-semibold flex items-center gap-2">
                                    <span>🚨</span> Critical Attention Required
                                </p>
                                <p className="text-gray-400 text-sm mt-2">
                                    This machine requires immediate inspection. Health score is below 50%.
                                    Schedule maintenance immediately to prevent failure.
                                </p>
                            </div>
                        )}
                        {machine.fault_type && machine.fault_type !== 'Normal' && (
                            <div className="p-4 rounded-xl bg-orange-500/10 border border-orange-500/20">
                                <p className="text-orange-400 font-semibold flex items-center gap-2">
                                    <span>🔧</span> Fault Detected: {machine.fault_type.replace('_', ' ')}
                                </p>
                                <p className="text-gray-400 text-sm mt-2">
                                    {machine.fault_type.includes('021')
                                        ? 'Severe fault detected. Replacement strongly recommended within 24 hours.'
                                        : machine.fault_type.includes('014')
                                            ? 'Moderate fault detected. Schedule inspection within 48 hours.'
                                            : 'Minor fault detected. Monitor progression and plan maintenance.'}
                                </p>
                            </div>
                        )}
                        {machine.rul_cycles !== null && machine.rul_cycles < 50 && (
                            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20">
                                <p className="text-amber-400 font-semibold flex items-center gap-2">
                                    <span>⏱️</span> Low Remaining Life
                                </p>
                                <p className="text-gray-400 text-sm mt-2">
                                    Approximately {Math.round(machine.rul_cycles)} cycles remaining.
                                    Order replacement parts and schedule maintenance window.
                                </p>
                            </div>
                        )}
                        {machine.health_score >= 0.8 && (
                            <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                                <p className="text-emerald-400 font-semibold flex items-center gap-2">
                                    <span>✅</span> Operating Normally
                                </p>
                                <p className="text-gray-400 text-sm mt-2">
                                    All systems nominal. Continue regular monitoring schedule.
                                    No immediate action required.
                                </p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Back Link */}
                <div className="pt-4">
                    <Link
                        href="/"
                        className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
                    >
                        ← Back to Dashboard
                    </Link>
                </div>
            </main>
        </div>
    );
}
