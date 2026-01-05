"""
Plant-Level Intelligence Service.
Aggregates machine health data and generates maintenance priorities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


@dataclass
class MachineStatus:
    """Status of a single machine."""
    machine_id: str
    machine_type: str  # 'fan', 'pump', 'valve', 'bearing'
    anomaly_score: float = 0.0
    fault_type: Optional[str] = None
    fault_confidence: float = 0.0
    rul_cycles: Optional[float] = None
    health_score: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'machine_id': self.machine_id,
            'machine_type': self.machine_type,
            'anomaly_score': self.anomaly_score,
            'fault_type': self.fault_type,
            'fault_confidence': self.fault_confidence,
            'rul_cycles': self.rul_cycles,
            'health_score': self.health_score,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class MaintenanceRecommendation:
    """Maintenance recommendation for a machine."""
    machine_id: str
    priority: str  # 'critical', 'high', 'medium', 'low'
    risk_score: float
    action: str
    estimated_time: str  # e.g., "within 24 hours"
    details: str
    
    def to_dict(self) -> Dict:
        return {
            'machine_id': self.machine_id,
            'priority': self.priority,
            'risk_score': self.risk_score,
            'action': self.action,
            'estimated_time': self.estimated_time,
            'details': self.details
        }


class PlantIntelligence:
    """Aggregates machine data and generates plant-level insights."""
    
    def __init__(self):
        self.machines: Dict[str, MachineStatus] = {}
        self.history: Dict[str, List[MachineStatus]] = {}
    
    def update_machine(
        self,
        machine_id: str,
        machine_type: str,
        anomaly_score: float = 0.0,
        fault_type: Optional[str] = None,
        fault_confidence: float = 0.0,
        rul_cycles: Optional[float] = None
    ) -> MachineStatus:
        """Update status for a machine."""
        # Calculate health score
        health_score = self._calculate_health_score(
            anomaly_score, fault_type, rul_cycles
        )
        
        status = MachineStatus(
            machine_id=machine_id,
            machine_type=machine_type,
            anomaly_score=anomaly_score,
            fault_type=fault_type,
            fault_confidence=fault_confidence,
            rul_cycles=rul_cycles,
            health_score=health_score,
            last_updated=datetime.now()
        )
        
        self.machines[machine_id] = status
        
        # Track history
        if machine_id not in self.history:
            self.history[machine_id] = []
        self.history[machine_id].append(status)
        
        # Keep only last 100 records per machine
        if len(self.history[machine_id]) > 100:
            self.history[machine_id] = self.history[machine_id][-100:]
        
        return status
    
    def _calculate_health_score(
        self,
        anomaly_score: float,
        fault_type: Optional[str],
        rul_cycles: Optional[float]
    ) -> float:
        """Calculate overall health score (0-1, higher is healthier)."""
        scores = []
        
        # Anomaly component (invert: low anomaly = high health)
        anomaly_health = max(0, 1 - anomaly_score * 5)  # Scale anomaly score
        scores.append(anomaly_health)
        
        # Fault component
        if fault_type and fault_type != 'Normal':
            fault_health = 0.3  # Fault detected = low health
        else:
            fault_health = 1.0
        scores.append(fault_health)
        
        # RUL component
        if rul_cycles is not None:
            rul_health = min(1.0, rul_cycles / 125)  # Normalize to max 125
            scores.append(rul_health)
        
        # Weighted average
        return np.mean(scores)
    
    def calculate_risk_score(self, machine_id: str) -> float:
        """
        Calculate risk score for a machine.
        Higher score = higher risk = needs attention sooner.
        """
        if machine_id not in self.machines:
            return 0.0
        
        status = self.machines[machine_id]
        
        # Risk factors
        risk = 0.0
        
        # Anomaly risk
        risk += status.anomaly_score * 30
        
        # Fault risk
        if status.fault_type and status.fault_type != 'Normal':
            risk += (1 - status.fault_confidence) * 20 + 30
        
        # RUL risk (inverse: low RUL = high risk)
        if status.rul_cycles is not None:
            rul_risk = max(0, (125 - status.rul_cycles) / 125) * 40
            risk += rul_risk
        
        return min(100, risk)
    
    def get_maintenance_queue(self) -> List[MaintenanceRecommendation]:
        """Generate prioritized maintenance queue for all machines."""
        recommendations = []
        
        for machine_id, status in self.machines.items():
            risk_score = self.calculate_risk_score(machine_id)
            
            # Determine priority
            if risk_score >= 80:
                priority = 'critical'
                estimated_time = 'immediately'
            elif risk_score >= 60:
                priority = 'high'
                estimated_time = 'within 24 hours'
            elif risk_score >= 40:
                priority = 'medium'
                estimated_time = 'within 1 week'
            else:
                priority = 'low'
                estimated_time = 'next scheduled maintenance'
            
            # Generate action and details
            action, details = self._generate_recommendation(status, risk_score)
            
            recommendations.append(MaintenanceRecommendation(
                machine_id=machine_id,
                priority=priority,
                risk_score=risk_score,
                action=action,
                estimated_time=estimated_time,
                details=details
            ))
        
        # Sort by risk score (descending)
        recommendations.sort(key=lambda x: x.risk_score, reverse=True)
        
        return recommendations
    
    def _generate_recommendation(
        self,
        status: MachineStatus,
        risk_score: float
    ) -> tuple:
        """Generate action and details for a machine."""
        actions = []
        details = []
        
        # Check anomaly
        if status.anomaly_score > 0.1:
            actions.append('Investigate anomaly')
            details.append(f'Anomaly score: {status.anomaly_score:.3f}')
        
        # Check fault
        if status.fault_type and status.fault_type != 'Normal':
            actions.append(f'Inspect {status.fault_type.replace("_", " ")}')
            details.append(f'Fault confidence: {status.fault_confidence:.1%}')
        
        # Check RUL
        if status.rul_cycles is not None and status.rul_cycles < 50:
            actions.append('Plan replacement')
            details.append(f'RUL: {status.rul_cycles:.0f} cycles remaining')
        
        if not actions:
            actions.append('Continue monitoring')
            details.append('No immediate action required')
        
        return '; '.join(actions), ' | '.join(details)
    
    def get_plant_summary(self) -> Dict:
        """Get summary statistics for the entire plant."""
        if not self.machines:
            return {
                'total_machines': 0,
                'healthy_count': 0,
                'warning_count': 0,
                'critical_count': 0,
                'average_health': 1.0,
                'machines_needing_attention': []
            }
        
        health_scores = [m.health_score for m in self.machines.values()]
        
        healthy = sum(1 for h in health_scores if h >= 0.8)
        warning = sum(1 for h in health_scores if 0.5 <= h < 0.8)
        critical = sum(1 for h in health_scores if h < 0.5)
        
        # Get machines needing attention
        attention_needed = [
            m.machine_id for m in self.machines.values()
            if m.health_score < 0.8
        ]
        
        return {
            'total_machines': len(self.machines),
            'healthy_count': healthy,
            'warning_count': warning,
            'critical_count': critical,
            'average_health': np.mean(health_scores),
            'machines_needing_attention': attention_needed
        }
    
    def get_all_machines(self) -> List[Dict]:
        """Get status of all machines as dictionaries."""
        return [m.to_dict() for m in self.machines.values()]
    
    def get_machine(self, machine_id: str) -> Optional[Dict]:
        """Get status of a specific machine."""
        if machine_id in self.machines:
            return self.machines[machine_id].to_dict()
        return None
