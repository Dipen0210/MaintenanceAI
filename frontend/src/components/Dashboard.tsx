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
        if (value >= 0.8) return { text: 'text-emerald-400', gradient: 'progress-gradient-green', glow: 'shadow-emerald-500/30' };
        if (value >= 0.5) return { text: 'text-amber-400', gradient: 'progress-gradient-yellow', glow: 'shadow-amber-500/30' };
        if (value >= 0.2) return { text: 'text-orange-400', gradient: 'progress-gradient-orange', glow: 'shadow-orange-500/30' };
        return { text: 'text-red-400', gradient: 'progress-gradient-red', glow: 'shadow-red-500/30' };
    };

    const colors = getColor();

    const sizeClasses = {
        sm: 'h-1.5',
        md: 'h-2',
        lg: 'h-3',
    };

    return (
        <div className="w-full">
            {showLabel && (
                <div className="flex justify-between mb-2">
                    <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Health</span>
                    <span className={`text-sm font-bold ${colors.text}`}>{percentage}%</span>
                </div>
            )}
            <div className={`w-full bg-gray-800/50 rounded-full ${sizeClasses[size]} overflow-hidden`}>
                <div
                    className={`${colors.gradient} ${sizeClasses[size]} rounded-full transition-all duration-700 ease-out shadow-lg ${colors.glow}`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    );
}

interface CircularProgressProps {
    value: number;
    size?: number;
    strokeWidth?: number;
}

export function CircularProgress({ value, size = 60, strokeWidth = 4 }: CircularProgressProps) {
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (value * circumference);
    const percentage = Math.round(value * 100);

    const getColor = () => {
        if (value >= 0.8) return '#22c55e';
        if (value >= 0.5) return '#eab308';
        if (value >= 0.2) return '#f97316';
        return '#ef4444';
    };

    return (
        <div className="relative" style={{ width: size, height: size }}>
            <svg className="transform -rotate-90" width={size} height={size}>
                <circle
                    className="text-gray-800"
                    strokeWidth={strokeWidth}
                    stroke="currentColor"
                    fill="transparent"
                    r={radius}
                    cx={size / 2}
                    cy={size / 2}
                />
                <circle
                    className="transition-all duration-700 ease-out"
                    strokeWidth={strokeWidth}
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                    stroke={getColor()}
                    fill="transparent"
                    r={radius}
                    cx={size / 2}
                    cy={size / 2}
                    style={{ filter: `drop-shadow(0 0 6px ${getColor()})` }}
                />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-sm font-bold text-white">{percentage}%</span>
            </div>
        </div>
    );
}

interface MachineCardProps {
    machine: MachineStatus;
    onClick?: () => void;
}

export function MachineCard({ machine, onClick }: MachineCardProps) {
    const getStatusConfig = () => {
        if (machine.health_score >= 0.8) return { 
            border: 'border-emerald-500/30 hover:border-emerald-500/50', 
            bg: 'bg-emerald-500/5 hover:bg-emerald-500/10',
            badge: 'bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/30',
            icon: '✓',
            text: 'Healthy',
            glow: 'hover:shadow-emerald-500/20'
        };
        if (machine.health_score >= 0.5) return { 
            border: 'border-amber-500/30 hover:border-amber-500/50', 
            bg: 'bg-amber-500/5 hover:bg-amber-500/10',
            badge: 'bg-amber-500/20 text-amber-400 ring-1 ring-amber-500/30',
            icon: '⚠',
            text: 'Warning',
            glow: 'hover:shadow-amber-500/20'
        };
        if (machine.health_score >= 0.2) return { 
            border: 'border-orange-500/30 hover:border-orange-500/50', 
            bg: 'bg-orange-500/5 hover:bg-orange-500/10',
            badge: 'bg-orange-500/20 text-orange-400 ring-1 ring-orange-500/30',
            icon: '⚡',
            text: 'Critical',
            glow: 'hover:shadow-orange-500/20'
        };
        return { 
            border: 'border-red-500/30 hover:border-red-500/50', 
            bg: 'bg-red-500/5 hover:bg-red-500/10',
            badge: 'bg-red-500/20 text-red-400 ring-1 ring-red-500/30 animate-pulse',
            icon: '✕',
            text: 'Failing',
            glow: 'hover:shadow-red-500/20'
        };
    };

    const config = getStatusConfig();

    const machineIcons: Record<string, string> = {
        fan: '🌀',
        pump: '⚙️',
        valve: '🔧',
        bearing: '⭕',
    };

    return (
        <div
            onClick={onClick}
            className={`relative p-5 rounded-2xl border ${config.border} ${config.bg} 
                cursor-pointer card-hover group backdrop-blur-sm 
                shadow-lg ${config.glow} hover:shadow-xl`}
        >
            {/* Background Glow Effect */}
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/[0.02] to-transparent pointer-events-none" />
            
            <div className="relative z-10">
                {/* Header */}
                <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-3">
                        <div className="text-2xl">{machineIcons[machine.machine_type] || '🔩'}</div>
                        <div>
                            <h3 className="font-semibold text-white text-lg group-hover:text-white/90 transition-colors">
                                {machine.machine_id}
                            </h3>
                            <p className="text-xs text-gray-500 uppercase tracking-wider font-medium">
                                {machine.machine_type}
                            </p>
                        </div>
                    </div>
                    <span className={`px-3 py-1.5 rounded-full text-xs font-semibold ${config.badge} flex items-center gap-1.5`}>
                        <span>{config.icon}</span>
                        {config.text}
                    </span>
                </div>

                {/* Health Gauge */}
                <div className="mb-4">
                    <HealthGauge value={machine.health_score} size="md" />
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-3">
                    {machine.rul_cycles !== null && (
                        <div className="glass rounded-xl p-3">
                            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">RUL</p>
                            <p className="text-lg font-bold text-white">
                                {Math.round(machine.rul_cycles)}
                                <span className="text-xs text-gray-500 font-normal ml-1">cycles</span>
                            </p>
                        </div>
                    )}
                    {machine.fault_type && machine.fault_type !== 'Normal' && (
                        <div className="glass rounded-xl p-3 border border-red-500/20">
                            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Fault</p>
                            <p className="text-sm font-semibold text-red-400 truncate">
                                {machine.fault_type.replace('_', ' ')}
                            </p>
                        </div>
                    )}
                    {machine.anomaly_score > 0.05 && (
                        <div className="glass rounded-xl p-3">
                            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Anomaly</p>
                            <p className="text-lg font-bold text-amber-400">
                                {(machine.anomaly_score * 100).toFixed(1)}%
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

interface StatCardProps {
    title: string;
    value: string | number;
    subtitle?: string;
    icon?: React.ReactNode;
    color?: 'green' | 'yellow' | 'red' | 'blue' | 'purple';
}

export function StatCard({ title, value, subtitle, icon, color = 'blue' }: StatCardProps) {
    const colorConfig = {
        green: { 
            gradient: 'from-emerald-500/20 via-emerald-500/10 to-transparent',
            border: 'border-emerald-500/20',
            accent: 'text-emerald-400',
            glow: 'shadow-emerald-500/10'
        },
        yellow: { 
            gradient: 'from-amber-500/20 via-amber-500/10 to-transparent',
            border: 'border-amber-500/20',
            accent: 'text-amber-400',
            glow: 'shadow-amber-500/10'
        },
        red: { 
            gradient: 'from-red-500/20 via-red-500/10 to-transparent',
            border: 'border-red-500/20',
            accent: 'text-red-400',
            glow: 'shadow-red-500/10'
        },
        blue: { 
            gradient: 'from-blue-500/20 via-blue-500/10 to-transparent',
            border: 'border-blue-500/20',
            accent: 'text-blue-400',
            glow: 'shadow-blue-500/10'
        },
        purple: { 
            gradient: 'from-purple-500/20 via-purple-500/10 to-transparent',
            border: 'border-purple-500/20',
            accent: 'text-purple-400',
            glow: 'shadow-purple-500/10'
        },
    };

    const config = colorConfig[color];

    return (
        <div className={`relative p-6 rounded-2xl bg-gradient-to-br ${config.gradient} 
            border ${config.border} backdrop-blur-sm card-hover shadow-lg ${config.glow}`}>
            {/* Decorative element */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-white/[0.02] to-transparent rounded-2xl pointer-events-none" />
            
            <div className="relative z-10 flex items-start justify-between">
                <div>
                    <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">{title}</p>
                    <p className={`text-4xl font-bold text-white mb-1 ${value === 0 ? 'text-gray-600' : ''}`}>
                        {value}
                    </p>
                    {subtitle && (
                        <p className="text-sm text-gray-400">{subtitle}</p>
                    )}
                </div>
                {icon && (
                    <div className={`${config.accent} text-3xl opacity-50`}>{icon}</div>
                )}
            </div>
        </div>
    );
}

interface AlertBadgeProps {
    priority: 'critical' | 'high' | 'medium' | 'low';
}

export function AlertBadge({ priority }: AlertBadgeProps) {
    const styles = {
        critical: 'bg-red-500/20 text-red-400 ring-1 ring-red-500/40 animate-pulse',
        high: 'bg-orange-500/20 text-orange-400 ring-1 ring-orange-500/40',
        medium: 'bg-amber-500/20 text-amber-400 ring-1 ring-amber-500/40',
        low: 'bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/40',
    };

    const icons = {
        critical: '🚨',
        high: '⚠️',
        medium: '📋',
        low: '✓',
    };

    return (
        <span className={`px-3 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wider flex items-center gap-2 ${styles[priority]}`}>
            <span>{icons[priority]}</span>
            {priority}
        </span>
    );
}

interface MaintenanceItemProps {
    item: {
        machine_id: string;
        priority: 'critical' | 'high' | 'medium' | 'low';
        risk_score: number;
        action: string;
        estimated_time: string;
        details: string;
    };
}

export function MaintenanceItem({ item }: MaintenanceItemProps) {
    const priorityColors = {
        critical: 'border-l-red-500 bg-red-500/5',
        high: 'border-l-orange-500 bg-orange-500/5',
        medium: 'border-l-amber-500 bg-amber-500/5',
        low: 'border-l-emerald-500 bg-emerald-500/5',
    };

    return (
        <div className={`p-4 rounded-xl glass border-l-4 ${priorityColors[item.priority]} 
            hover:bg-white/[0.03] transition-all duration-200 card-hover`}>
            <div className="flex items-start justify-between mb-3">
                <div>
                    <span className="font-semibold text-white text-base">{item.machine_id}</span>
                    <p className="text-xs text-gray-500 mt-0.5">{item.estimated_time}</p>
                </div>
                <AlertBadge priority={item.priority} />
            </div>
            <p className="text-sm text-gray-300 mb-2">{item.action}</p>
            <div className="flex items-center justify-between">
                <p className="text-xs text-gray-500">{item.details}</p>
                <div className="flex items-center gap-1">
                    <span className="text-xs text-gray-500">Risk:</span>
                    <span className={`text-sm font-bold ${item.risk_score >= 70 ? 'text-red-400' : item.risk_score >= 40 ? 'text-amber-400' : 'text-emerald-400'}`}>
                        {Math.round(item.risk_score)}%
                    </span>
                </div>
            </div>
        </div>
    );
}
