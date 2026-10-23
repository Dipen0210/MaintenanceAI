'use client';

import { useEffect, useState } from 'react';
import { MachineStatus, PlantSummary } from '@/lib/api';

interface WebSocketMessage {
    type: 'connected' | 'machine_update' | 'plant_summary' | 'alert' | 'pong' | 'error';
    timestamp?: string;
    data?: unknown;
    message?: string;
}

interface Alert {
    machine_id: string;
    alert_type: 'critical' | 'warning' | 'fault';
    details: string;
    timestamp: string;
}

interface UseWebSocketReturn {
    isConnected: boolean;
    lastUpdate: Date | null;
    machineUpdates: MachineStatus[];
    plantSummary: PlantSummary | null;
    alerts: Alert[];
    connectionCount: number;
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/updates';

export function useWebSocket(): UseWebSocketReturn {
    const [isConnected, setIsConnected] = useState(false);
    const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
    const [machineUpdates, setMachineUpdates] = useState<MachineStatus[]>([]);
    const [plantSummary, setPlantSummary] = useState<PlantSummary | null>(null);
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [connectionCount, setConnectionCount] = useState(0);

    useEffect(() => {
        let disposed = false;
        let ws: WebSocket | null = null;
        let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
        let attempts = 0;

        function connect() {
            if (disposed || (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING))) return;
            ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                console.log('🔌 WebSocket connected');
                setIsConnected(true);
                attempts = 0;
                setConnectionCount(prev => prev + 1);
            };

            ws.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data);
                    setLastUpdate(new Date());

                    switch (message.type) {
                        case 'machine_update':
                            const machineData = message.data as MachineStatus;
                            setMachineUpdates(prev => {
                                // Keep last 50 updates
                                const updated = [machineData, ...prev.slice(0, 49)];
                                return updated;
                            });
                            break;

                        case 'plant_summary':
                            setPlantSummary(message.data as PlantSummary);
                            break;

                        case 'alert':
                            const alertData = message.data as Omit<Alert, 'timestamp'>;
                            setAlerts(prev => [{
                                ...alertData,
                                timestamp: message.timestamp || new Date().toISOString()
                            }, ...prev.slice(0, 19)]); // Keep last 20 alerts
                            break;

                        case 'connected':
                            console.log('✅ WebSocket: ', message.message);
                            break;

                        case 'pong':
                            // Heartbeat response
                            break;
                    }
                } catch (err) {
                    console.error('WebSocket parse error:', err);
                }
            };

            ws.onclose = () => {
                if (disposed) return;
                console.log('🔌 WebSocket disconnected');
                setIsConnected(false);

                // Reconnect with exponential backoff
                const delay = Math.min(1000 * Math.pow(2, attempts), 30000);
                attempts++;

                reconnectTimer = setTimeout(() => {
                    console.log(`🔄 Reconnecting... (attempt ${attempts})`);
                    connect();
                }, delay);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

        }

        connect();
        const heartbeat = setInterval(() => {
            if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'ping' }));
        }, 30000);
        return () => {
            disposed = true;
            clearInterval(heartbeat);
            if (reconnectTimer) clearTimeout(reconnectTimer);
            if (ws) {
                ws.onopen = null;
                ws.onmessage = null;
                ws.onclose = null;
                ws.onerror = null;
                ws.close();
            }
        };
    }, []);

    return {
        isConnected,
        lastUpdate,
        machineUpdates,
        plantSummary,
        alerts,
        connectionCount
    };
}
