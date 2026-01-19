'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
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

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const reconnectAttempts = useRef(0);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        try {
            const ws = new WebSocket(WS_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('🔌 WebSocket connected');
                setIsConnected(true);
                reconnectAttempts.current = 0;
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
                console.log('🔌 WebSocket disconnected');
                setIsConnected(false);

                // Reconnect with exponential backoff
                const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
                reconnectAttempts.current++;

                reconnectTimeoutRef.current = setTimeout(() => {
                    console.log(`🔄 Reconnecting... (attempt ${reconnectAttempts.current})`);
                    connect();
                }, delay);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

        } catch (err) {
            console.error('WebSocket connection error:', err);
        }
    }, []);

    useEffect(() => {
        connect();

        // Heartbeat to keep connection alive
        const heartbeat = setInterval(() => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);

        return () => {
            clearInterval(heartbeat);
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [connect]);

    return {
        isConnected,
        lastUpdate,
        machineUpdates,
        plantSummary,
        alerts,
        connectionCount
    };
}
