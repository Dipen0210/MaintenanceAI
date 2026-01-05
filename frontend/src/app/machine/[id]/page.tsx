'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { HealthGauge, StatCard } from '@/components/Dashboard';
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
            <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500" />
            </div>
        );
    }

    const getStatusInfo = () => {
        if (machine.health_score >= 0.8) return { status: 'Healthy', color: 'text-green-400', bg: 'bg-green-500/10' };
        if (machine.health_score >= 0.5) return { status: 'Warning', color: 'text-yellow-400', bg: 'bg-yellow-500/10' };
        if (machine.health_score >= 0.2) return { status: 'Critical', color: 'text-orange-400', bg: 'bg-orange-500/10' };
        return { status: 'Failing', color: 'text-red-400', bg: 'bg-red-500/10' };
    };

    const statusInfo = getStatusInfo();

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
            {/* Header */}
            <header className="border-b border-gray-700/50 backdrop-blur-sm bg-gray-900/50">
                <div className="max-w-7xl mx-auto px-4 py-4">
                    <div className="flex items-center gap-4">
                        <Link
                            href="/"
                            className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
                        >
                            ← Back
                        </Link>
                        <div>
                            <h1 className="text-2xl font-bold text-white">{machine.machine_id}</h1>
                            <p className="text-gray-400 capitalize">{machine.machine_type}</p>
                        </div>
                        <div className={`ml-auto flex items-center gap-2 px-3 py-1 rounded-full ${isLive ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                            <span className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`} />
                            {isLive ? 'Live' : 'Demo'}
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
                {/* Status Banner */}
                <div className={`p-6 rounded-2xl ${statusInfo.bg} border border-gray-700`}>
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-gray-400 mb-1">Current Status</p>
                            <h2 className={`text-3xl font-bold ${statusInfo.color}`}>{statusInfo.status}</h2>
                        </div>
                        <div className="text-right">
                            <p className="text-gray-400 mb-1">Last Updated</p>
                            <p className="text-white">{new Date(machine.last_updated).toLocaleString()}</p>
                        </div>
                    </div>
                </div>

                {/* Health Gauge */}
                <div className="p-6 rounded-2xl bg-gray-800/50 border border-gray-700">
                    <h3 className="text-lg font-semibold text-white mb-4">Overall Health</h3>
                    <HealthGauge value={machine.health_score} size="lg" />
                    <p className="text-gray-400 text-sm mt-3">
                        Health score is calculated from anomaly detection, fault diagnosis, and RUL prediction.
                    </p>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <StatCard
                        title="Anomaly Score"
                        value={machine.anomaly_score.toFixed(3)}
                        subtitle={machine.anomaly_score > 0.1 ? 'Above threshold' : 'Normal range'}
                        color={machine.anomaly_score > 0.1 ? 'red' : 'green'}
                    />
                    <StatCard
                        title="Remaining Life"
                        value={machine.rul_cycles !== null ? `${Math.round(machine.rul_cycles)}` : 'N/A'}
                        subtitle="Cycles remaining"
                        color={machine.rul_cycles && machine.rul_cycles < 50 ? 'red' : 'green'}
                    />
                    <StatCard
                        title="Fault Status"
                        value={machine.fault_type || 'None'}
                        subtitle={machine.fault_confidence > 0 ? `${(machine.fault_confidence * 100).toFixed(0)}% confidence` : 'No fault detected'}
                        color={machine.fault_type && machine.fault_type !== 'Normal' ? 'red' : 'green'}
                    />
                </div>

                {/* Recommendations */}
                <div className="p-6 rounded-2xl bg-gray-800/50 border border-gray-700">
                    <h3 className="text-lg font-semibold text-white mb-4">💡 Recommendations</h3>
                    <div className="space-y-3">
                        {machine.health_score < 0.5 && (
                            <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30">
                                <p className="text-red-400 font-medium">⚠️ Critical Attention Required</p>
                                <p className="text-gray-300 text-sm mt-1">
                                    This machine requires immediate inspection. Health score is below 50%.
                                </p>
                            </div>
                        )}
                        {machine.fault_type && machine.fault_type !== 'Normal' && (
                            <div className="p-3 rounded-lg bg-orange-500/10 border border-orange-500/30">
                                <p className="text-orange-400 font-medium">🔧 Fault Detected: {machine.fault_type}</p>
                                <p className="text-gray-300 text-sm mt-1">
                                    Schedule bearing inspection. {machine.fault_type.includes('021') ? 'Replacement recommended.' : 'Monitor progression.'}
                                </p>
                            </div>
                        )}
                        {machine.rul_cycles !== null && machine.rul_cycles < 50 && (
                            <div className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
                                <p className="text-yellow-400 font-medium">⏱️ Low Remaining Life</p>
                                <p className="text-gray-300 text-sm mt-1">
                                    Approximately {Math.round(machine.rul_cycles)} cycles remaining. Plan replacement.
                                </p>
                            </div>
                        )}
                        {machine.health_score >= 0.8 && (
                            <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/30">
                                <p className="text-green-400 font-medium">✅ Operating Normally</p>
                                <p className="text-gray-300 text-sm mt-1">
                                    Continue regular monitoring. No immediate action required.
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
