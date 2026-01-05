/**
 * API Client for Predictive Maintenance Backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface PlantSummary {
  total_machines: number;
  healthy_count: number;
  warning_count: number;
  critical_count: number;
  average_health: number;
  machines_needing_attention: string[];
}

export interface MachineStatus {
  machine_id: string;
  machine_type: string;
  anomaly_score: number;
  fault_type: string | null;
  fault_confidence: number;
  rul_cycles: number | null;
  health_score: number;
  last_updated: string;
}

export interface MaintenanceRecommendation {
  machine_id: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  risk_score: number;
  action: string;
  estimated_time: string;
  details: string;
}

export interface RULPrediction {
  rul_cycles: number;
  health_score: number;
  status: string;
  recommendation: string;
}

export interface VibrationPrediction {
  fault_type: string;
  confidence: number;
  description: string;
  is_faulty: boolean;
  severity?: string;
}

export interface AudioPrediction {
  is_anomaly: boolean;
  anomaly_score: number;
  label: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Plant endpoints
  async getPlantSummary(): Promise<PlantSummary> {
    return this.fetch<PlantSummary>('/plant/summary');
  }

  async getAllMachines(): Promise<MachineStatus[]> {
    return this.fetch<MachineStatus[]>('/plant/machines');
  }

  async getMachine(machineId: string): Promise<MachineStatus> {
    return this.fetch<MachineStatus>(`/plant/machines/${machineId}`);
  }

  async getMaintenanceQueue(): Promise<MaintenanceRecommendation[]> {
    return this.fetch<MaintenanceRecommendation[]>('/plant/maintenance-queue');
  }

  // Prediction endpoints
  async predictVibration(signal: number[]): Promise<VibrationPrediction> {
    return this.fetch<VibrationPrediction>('/predict/vibration', {
      method: 'POST',
      body: JSON.stringify({ signal }),
    });
  }

  async predictRUL(sequence: number[][]): Promise<RULPrediction> {
    return this.fetch<RULPrediction>('/predict/rul', {
      method: 'POST',
      body: JSON.stringify({ sequence }),
    });
  }

  async predictAudio(file: File): Promise<AudioPrediction> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/predict/audio`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }

    return response.json();
  }

  async updateMachine(
    machineId: string,
    machineType: string,
    data: Partial<{
      anomaly_score: number;
      fault_type: string;
      fault_confidence: number;
      rul_cycles: number;
    }>
  ): Promise<MachineStatus> {
    const params = new URLSearchParams({
      machine_type: machineType,
      ...Object.fromEntries(
        Object.entries(data).map(([k, v]) => [k, String(v)])
      ),
    });

    return this.fetch<MachineStatus>(
      `/plant/machines/${machineId}/update?${params}`,
      { method: 'POST' }
    );
  }
}

export const apiClient = new ApiClient();
export default apiClient;
