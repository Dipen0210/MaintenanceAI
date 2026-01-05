'use client';

import { MachineStatus } from '@/lib/api';

interface HealthGaugeProps {
    value: number;
    size?: 'sm' | 'md' | 'lg';
    showLabel?: boolean;
}

export function HealthGauge({ value, size = 'md', showLabel = true }: HealthGaugeProps) {
    const percentage = Math.round(value * 100);

    const getColor = () => {
        if (value >= 0.8) return 'text-green-500';
        if (value >= 0.5) return 'text-yellow-500';
        if (value >= 0.2) return 'text-orange-500';
        return 'text-red-500';
    };

    const getBgColor = () => {
        if (value >= 0.8) return 'bg-green-500';
        if (value >= 0.5) return 'bg-yellow-500';
        if (value >= 0.2) return 'bg-orange-500';
        return 'bg-red-500';
    };

    const sizeClasses = {
        sm: 'h-2',
        md: 'h-3',
        lg: 'h-4',
    };

    return (
        <div className="w-full">
            <div className="flex justify-between mb-1">
                {showLabel && (
                    <>
                        <span className="text-sm text-gray-400">Health</span>
                        <span className={`text-sm font-semibold ${getColor()}`}>{percentage}%</span>
                    </>
                )}
            </div>
            <div className={`w-full bg-gray-700 rounded-full ${sizeClasses[size]}`}>
                <div
                    className={`${getBgColor()} ${sizeClasses[size]} rounded-full transition-all duration-500`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    );
}

interface MachineCardProps {
    machine: MachineStatus;
    onClick?: () => void;
}

export function MachineCard({ machine, onClick }: MachineCardProps) {
    const getStatusColor = () => {
        if (machine.health_score >= 0.8) return 'border-green-500/50 bg-green-500/5';
        if (machine.health_score >= 0.5) return 'border-yellow-500/50 bg-yellow-500/5';
        if (machine.health_score >= 0.2) return 'border-orange-500/50 bg-orange-500/5';
        return 'border-red-500/50 bg-red-500/5';
    };

    const getStatusBadge = () => {
        if (machine.health_score >= 0.8) return { text: 'Healthy', color: 'bg-green-500' };
        if (machine.health_score >= 0.5) return { text: 'Warning', color: 'bg-yellow-500' };
        if (machine.health_score >= 0.2) return { text: 'Critical', color: 'bg-orange-500' };
        return { text: 'Failing', color: 'bg-red-500' };
    };

    const badge = getStatusBadge();

    return (
        <div
            onClick={onClick}
            className={`p-4 rounded-xl border-2 ${getStatusColor()} cursor-pointer hover:scale-[1.02] transition-all duration-200`}
        >
            <div className="flex justify-between items-start mb-3">
                <div>
                    <h3 className="font-semibold text-white">{machine.machine_id}</h3>
                    <p className="text-sm text-gray-400 capitalize">{machine.machine_type}</p>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium text-white ${badge.color}`}>
                    {badge.text}
                </span>
            </div>

            <HealthGauge value={machine.health_score} size="sm" />

            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                {machine.rul_cycles !== null && (
                    <div>
                        <span className="text-gray-400">RUL:</span>
                        <span className="ml-1 text-white">{Math.round(machine.rul_cycles)} cycles</span>
                    </div>
                )}
                {machine.fault_type && machine.fault_type !== 'Normal' && (
                    <div>
                        <span className="text-gray-400">Fault:</span>
                        <span className="ml-1 text-red-400">{machine.fault_type}</span>
                    </div>
                )}
            </div>
        </div>
    );
}

interface StatCardProps {
    title: string;
    value: string | number;
    subtitle?: string;
    icon?: React.ReactNode;
    trend?: 'up' | 'down' | 'neutral';
    color?: 'green' | 'yellow' | 'red' | 'blue';
}

export function StatCard({ title, value, subtitle, icon, color = 'blue' }: StatCardProps) {
    const colors = {
        green: 'from-green-500/20 to-green-600/5 border-green-500/30',
        yellow: 'from-yellow-500/20 to-yellow-600/5 border-yellow-500/30',
        red: 'from-red-500/20 to-red-600/5 border-red-500/30',
        blue: 'from-blue-500/20 to-blue-600/5 border-blue-500/30',
    };

    return (
        <div className={`p-5 rounded-xl bg-gradient-to-br ${colors[color]} border backdrop-blur-sm`}>
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-sm text-gray-400">{title}</p>
                    <p className="text-3xl font-bold text-white mt-1">{value}</p>
                    {subtitle && <p className="text-sm text-gray-400 mt-1">{subtitle}</p>}
                </div>
                {icon && <div className="text-gray-400">{icon}</div>}
            </div>
        </div>
    );
}

interface AlertBadgeProps {
    priority: 'critical' | 'high' | 'medium' | 'low';
}

export function AlertBadge({ priority }: AlertBadgeProps) {
    const styles = {
        critical: 'bg-red-500 animate-pulse',
        high: 'bg-orange-500',
        medium: 'bg-yellow-500',
        low: 'bg-green-500',
    };

    return (
        <span className={`px-2 py-1 rounded text-xs font-medium text-white uppercase ${styles[priority]}`}>
            {priority}
        </span>
    );
}
