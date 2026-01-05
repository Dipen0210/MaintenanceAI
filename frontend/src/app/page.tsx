'use client';

import { useEffect, useState } from 'react';
import { StatCard, MachineCard, AlertBadge } from '@/components/Dashboard';
import { apiClient, PlantSummary, MachineStatus, MaintenanceRecommendation } from '@/lib/api';
import Link from 'next/link';

// Demo data for when backend is not running
const demoPlantSummary: PlantSummary = {
  total_machines: 8,
  healthy_count: 5,
  warning_count: 2,
  critical_count: 1,
  average_health: 0.76,
  machines_needing_attention: ['FAN-001', 'PUMP-003'],
};

const demoMachines: MachineStatus[] = [
  { machine_id: 'FAN-001', machine_type: 'fan', anomaly_score: 0.15, fault_type: null, fault_confidence: 0, rul_cycles: 45, health_score: 0.45, last_updated: new Date().toISOString() },
  { machine_id: 'FAN-002', machine_type: 'fan', anomaly_score: 0.02, fault_type: 'Normal', fault_confidence: 0.98, rul_cycles: 120, health_score: 0.95, last_updated: new Date().toISOString() },
  { machine_id: 'PUMP-001', machine_type: 'pump', anomaly_score: 0.01, fault_type: 'Normal', fault_confidence: 0.99, rul_cycles: 98, health_score: 0.92, last_updated: new Date().toISOString() },
  { machine_id: 'PUMP-002', machine_type: 'pump', anomaly_score: 0.03, fault_type: 'Normal', fault_confidence: 0.97, rul_cycles: 85, health_score: 0.88, last_updated: new Date().toISOString() },
  { machine_id: 'PUMP-003', machine_type: 'pump', anomaly_score: 0.22, fault_type: 'Ball_014', fault_confidence: 0.85, rul_cycles: 28, health_score: 0.35, last_updated: new Date().toISOString() },
  { machine_id: 'VALVE-001', machine_type: 'valve', anomaly_score: 0.04, fault_type: 'Normal', fault_confidence: 0.96, rul_cycles: 110, health_score: 0.91, last_updated: new Date().toISOString() },
  { machine_id: 'VALVE-002', machine_type: 'valve', anomaly_score: 0.08, fault_type: null, fault_confidence: 0, rul_cycles: 72, health_score: 0.78, last_updated: new Date().toISOString() },
  { machine_id: 'BEARING-001', machine_type: 'bearing', anomaly_score: 0.01, fault_type: 'Normal', fault_confidence: 0.99, rul_cycles: 130, health_score: 0.97, last_updated: new Date().toISOString() },
];

const demoMaintenanceQueue: MaintenanceRecommendation[] = [
  { machine_id: 'PUMP-003', priority: 'critical', risk_score: 82, action: 'Replace bearing; Investigate anomaly', estimated_time: 'immediately', details: 'Ball fault detected | RUL: 28 cycles' },
  { machine_id: 'FAN-001', priority: 'high', risk_score: 65, action: 'Plan replacement', estimated_time: 'within 24 hours', details: 'RUL: 45 cycles remaining' },
  { machine_id: 'VALVE-002', priority: 'medium', risk_score: 42, action: 'Monitor closely', estimated_time: 'within 1 week', details: 'Slightly elevated anomaly score' },
];

export default function DashboardPage() {
  const [plantSummary, setPlantSummary] = useState<PlantSummary>(demoPlantSummary);
  const [machines, setMachines] = useState<MachineStatus[]>(demoMachines);
  const [maintenanceQueue, setMaintenanceQueue] = useState<MaintenanceRecommendation[]>(demoMaintenanceQueue);
  const [isLive, setIsLive] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [summary, machineList, queue] = await Promise.all([
          apiClient.getPlantSummary(),
          apiClient.getAllMachines(),
          apiClient.getMaintenanceQueue(),
        ]);
        setPlantSummary(summary);
        setMachines(machineList);
        setMaintenanceQueue(queue);
        setIsLive(true);
      } catch (error) {
        console.log('Using demo data (backend not connected)');
        setIsLive(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Header */}
      <header className="border-b border-gray-700/50 backdrop-blur-sm bg-gray-900/50 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white">
                🏭 Predictive Maintenance Dashboard
              </h1>
              <p className="text-gray-400 text-sm">Real-time machine health monitoring</p>
            </div>
            <div className="flex items-center gap-4">
              <div className={`flex items-center gap-2 px-3 py-1 rounded-full ${isLive ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                <span className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`} />
                {isLive ? 'Live' : 'Demo Mode'}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {/* Summary Stats */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Total Machines"
            value={plantSummary.total_machines}
            subtitle="Being monitored"
            color="blue"
          />
          <StatCard
            title="Healthy"
            value={plantSummary.healthy_count}
            subtitle={`${Math.round((plantSummary.healthy_count / plantSummary.total_machines) * 100)}% of fleet`}
            color="green"
          />
          <StatCard
            title="Warning"
            value={plantSummary.warning_count}
            subtitle="Need attention"
            color="yellow"
          />
          <StatCard
            title="Critical"
            value={plantSummary.critical_count}
            subtitle="Immediate action required"
            color="red"
          />
        </section>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Machine Grid */}
          <section className="lg:col-span-2">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">Machine Fleet</h2>
              <span className="text-sm text-gray-400">
                Avg Health: {Math.round(plantSummary.average_health * 100)}%
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {machines.map((machine) => (
                <Link key={machine.machine_id} href={`/machine/${machine.machine_id}`}>
                  <MachineCard machine={machine} />
                </Link>
              ))}
            </div>
          </section>

          {/* Maintenance Queue */}
          <section>
            <h2 className="text-xl font-semibold text-white mb-4">Maintenance Queue</h2>
            <div className="space-y-3">
              {maintenanceQueue.length === 0 ? (
                <div className="p-4 rounded-xl bg-gray-800/50 border border-gray-700 text-center text-gray-400">
                  No maintenance tasks pending
                </div>
              ) : (
                maintenanceQueue.map((item, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-xl bg-gray-800/50 border border-gray-700 hover:border-gray-600 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <span className="font-medium text-white">{item.machine_id}</span>
                        <p className="text-sm text-gray-400">{item.estimated_time}</p>
                      </div>
                      <AlertBadge priority={item.priority} />
                    </div>
                    <p className="text-sm text-gray-300">{item.action}</p>
                    <p className="text-xs text-gray-500 mt-1">{item.details}</p>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
