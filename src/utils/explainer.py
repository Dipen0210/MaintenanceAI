"""
Explainability Module.
Converts model outputs to human-readable explanations and recommendations.
"""

from typing import Dict, Optional
from datetime import datetime


class Explainer:
    """Generates human-readable explanations for model outputs."""
    
    def __init__(self):
        self.fault_explanations = {
            'Normal': 'The bearing is operating within normal parameters.',
            'Ball_007': 'A minor ball defect (0.007" diameter) has been detected.',
            'Ball_014': 'A moderate ball defect (0.014" diameter) has been detected.',
            'Ball_021': 'A severe ball defect (0.021" diameter) has been detected.',
            'IR_007': 'A minor inner race defect (0.007" diameter) has been detected.',
            'IR_014': 'A moderate inner race defect (0.014" diameter) has been detected.',
            'IR_021': 'A severe inner race defect (0.021" diameter) has been detected.',
            'OR_007': 'A minor outer race defect (0.007" diameter) has been detected.',
            'OR_014': 'A moderate outer race defect (0.014" diameter) has been detected.',
            'OR_021': 'A severe outer race defect (0.021" diameter) has been detected.',
        }
        
        self.fault_actions = {
            'Normal': 'Continue normal operation. Schedule routine maintenance.',
            'Ball_007': 'Monitor closely. Plan inspection in next maintenance window.',
            'Ball_014': 'Schedule bearing inspection within 2 weeks.',
            'Ball_021': 'Replace bearing as soon as possible.',
            'IR_007': 'Monitor closely. Plan inspection in next maintenance window.',
            'IR_014': 'Schedule bearing inspection within 2 weeks.',
            'IR_021': 'Replace bearing immediately. Inner race failure is imminent.',
            'OR_007': 'Monitor closely. Plan inspection in next maintenance window.',
            'OR_014': 'Schedule bearing inspection within 1 week.',
            'OR_021': 'Replace bearing immediately. Outer race failure is imminent.',
        }
    
    def explain_anomaly(
        self,
        machine_id: str,
        machine_type: str,
        anomaly_score: float,
        is_anomaly: bool
    ) -> Dict:
        """Generate explanation for audio anomaly detection."""
        if is_anomaly:
            summary = f"⚠️ Anomaly detected in {machine_type} ({machine_id})"
            explanation = (
                f"The {machine_type} is producing abnormal sounds. "
                f"The anomaly score of {anomaly_score:.4f} exceeds the normal threshold. "
                f"This could indicate mechanical issues, loose components, or unusual operating conditions."
            )
            recommendation = (
                f"1. Inspect {machine_id} for visible issues\n"
                f"2. Check for loose components or debris\n"
                f"3. Verify operating conditions are within spec\n"
                f"4. Consider pausing operation if sound persists"
            )
            urgency = "high" if anomaly_score > 0.3 else "medium"
        else:
            summary = f"✅ {machine_type} ({machine_id}) operating normally"
            explanation = (
                f"The {machine_type} audio pattern matches normal operation. "
                f"No anomalies detected."
            )
            recommendation = "Continue normal operation. No action required."
            urgency = "low"
        
        return {
            "summary": summary,
            "explanation": explanation,
            "recommendation": recommendation,
            "urgency": urgency,
            "timestamp": datetime.now().isoformat()
        }
    
    def explain_fault(
        self,
        machine_id: str,
        fault_type: str,
        confidence: float
    ) -> Dict:
        """Generate explanation for vibration fault diagnosis."""
        explanation = self.fault_explanations.get(
            fault_type, 
            "Unknown fault type detected."
        )
        action = self.fault_actions.get(
            fault_type,
            "Inspect machine and consult maintenance team."
        )
        
        if fault_type == 'Normal':
            summary = f"✅ Bearing ({machine_id}) is healthy"
            urgency = "low"
        elif '007' in fault_type:
            summary = f"⚠️ Minor fault detected in {machine_id}"
            urgency = "medium"
        elif '014' in fault_type:
            summary = f"🔶 Moderate fault detected in {machine_id}"
            urgency = "high"
        else:
            summary = f"🔴 Severe fault detected in {machine_id}"
            urgency = "critical"
        
        return {
            "summary": summary,
            "fault_type": fault_type,
            "confidence": f"{confidence:.1%}",
            "explanation": explanation,
            "recommendation": action,
            "urgency": urgency,
            "timestamp": datetime.now().isoformat()
        }
    
    def explain_rul(
        self,
        machine_id: str,
        rul_cycles: float,
        health_score: float
    ) -> Dict:
        """Generate explanation for RUL prediction."""
        if rul_cycles > 100:
            summary = f"✅ {machine_id} has substantial remaining life"
            urgency = "low"
            timeline = "No immediate action needed"
        elif rul_cycles > 50:
            summary = f"⚠️ {machine_id} entering maintenance window"
            urgency = "medium"
            timeline = "Plan maintenance within 2-4 weeks"
        elif rul_cycles > 20:
            summary = f"🔶 {machine_id} requires attention soon"
            urgency = "high"
            timeline = "Schedule maintenance within 1 week"
        else:
            summary = f"🔴 {machine_id} at critical health level"
            urgency = "critical"
            timeline = "Immediate maintenance required"
        
        explanation = (
            f"Based on sensor data analysis, this machine has approximately "
            f"{rul_cycles:.0f} operational cycles remaining. "
            f"Current health score: {health_score:.1%}."
        )
        
        return {
            "summary": summary,
            "rul_cycles": round(rul_cycles, 1),
            "health_score": f"{health_score:.1%}",
            "explanation": explanation,
            "timeline": timeline,
            "urgency": urgency,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_alert(
        self,
        machine_id: str,
        alert_type: str,
        details: Dict
    ) -> str:
        """Generate a formatted alert message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if alert_type == "anomaly":
            return (
                f"🔔 ALERT [{timestamp}]\n"
                f"Machine: {machine_id}\n"
                f"Issue: Abnormal sound detected\n"
                f"Score: {details.get('score', 'N/A')}\n"
                f"Action: Inspect immediately"
            )
        elif alert_type == "fault":
            return (
                f"🔔 ALERT [{timestamp}]\n"
                f"Machine: {machine_id}\n"
                f"Fault: {details.get('fault_type', 'Unknown')}\n"
                f"Confidence: {details.get('confidence', 'N/A')}\n"
                f"Action: {details.get('action', 'Inspect')}"
            )
        elif alert_type == "rul":
            return (
                f"🔔 ALERT [{timestamp}]\n"
                f"Machine: {machine_id}\n"
                f"RUL: {details.get('rul_cycles', 'N/A')} cycles\n"
                f"Health: {details.get('health_score', 'N/A')}\n"
                f"Action: Schedule maintenance"
            )
        else:
            return f"🔔 ALERT [{timestamp}] - {machine_id}: {alert_type}"
