"""
Sample Machine Data Generator.
Initializes sample machines with area/zone information and random values.
"""

import random
from datetime import datetime
from typing import Dict, List

# Plant areas/zones
PLANT_AREAS = {
    "production_line_a": {
        "name": "Production Line A",
        "icon": "🏭",
        "description": "Main production floor"
    },
    "production_line_b": {
        "name": "Production Line B", 
        "icon": "🏭",
        "description": "Secondary production"
    },
    "hvac_system": {
        "name": "HVAC System",
        "icon": "❄️",
        "description": "Climate control"
    },
    "utilities": {
        "name": "Utilities",
        "icon": "⚡",
        "description": "Power & water systems"
    }
}

# Sample machine definitions with areas
SAMPLE_MACHINES = [
    # Production Line A
    {"machine_id": "PLA-FAN-001", "machine_type": "fan", "area": "production_line_a"},
    {"machine_id": "PLA-PUMP-001", "machine_type": "pump", "area": "production_line_a"},
    {"machine_id": "PLA-PUMP-002", "machine_type": "pump", "area": "production_line_a"},
    {"machine_id": "PLA-BEARING-001", "machine_type": "bearing", "area": "production_line_a"},
    
    # Production Line B
    {"machine_id": "PLB-FAN-001", "machine_type": "fan", "area": "production_line_b"},
    {"machine_id": "PLB-FAN-002", "machine_type": "fan", "area": "production_line_b"},
    {"machine_id": "PLB-PUMP-001", "machine_type": "pump", "area": "production_line_b"},
    
    # HVAC System
    {"machine_id": "HVAC-FAN-001", "machine_type": "fan", "area": "hvac_system"},
    {"machine_id": "HVAC-FAN-002", "machine_type": "fan", "area": "hvac_system"},
    {"machine_id": "HVAC-VALVE-001", "machine_type": "valve", "area": "hvac_system"},
    
    # Utilities
    {"machine_id": "UTIL-PUMP-001", "machine_type": "pump", "area": "utilities"},
    {"machine_id": "UTIL-PUMP-002", "machine_type": "pump", "area": "utilities"},
    {"machine_id": "UTIL-VALVE-001", "machine_type": "valve", "area": "utilities"},
    {"machine_id": "UTIL-BEARING-001", "machine_type": "bearing", "area": "utilities"},
]

# Possible fault types
FAULT_TYPES = [
    "Normal",
    "Ball_007", "Ball_014", "Ball_021",
    "IR_007", "IR_014", "IR_021",
    "OR_007", "OR_014", "OR_021",
]


def generate_random_machine_status(machine: Dict) -> Dict:
    """Generate random status values for a machine."""
    # Randomly decide if machine has issues (20% chance of issues)
    has_issues = random.random() < 0.2
    
    if has_issues:
        anomaly_score = random.uniform(0.1, 0.4)
        fault_type = random.choice(FAULT_TYPES[1:])  # Exclude "Normal"
        fault_confidence = random.uniform(0.7, 0.95)
        rul_cycles = random.randint(15, 50)
    else:
        anomaly_score = random.uniform(0.0, 0.08)
        fault_type = "Normal"
        fault_confidence = random.uniform(0.9, 0.99)
        rul_cycles = random.randint(80, 150)
    
    return {
        "machine_id": machine["machine_id"],
        "machine_type": machine["machine_type"],
        "area": machine.get("area", "unknown"),
        "anomaly_score": round(anomaly_score, 3),
        "fault_type": fault_type,
        "fault_confidence": round(fault_confidence, 2),
        "rul_cycles": rul_cycles,
    }


def get_all_sample_machines() -> List[Dict]:
    """Get all sample machines with random status values."""
    return [generate_random_machine_status(m) for m in SAMPLE_MACHINES]


def get_sample_machine_ids() -> List[str]:
    """Get list of sample machine IDs."""
    return [m["machine_id"] for m in SAMPLE_MACHINES]


def get_plant_areas() -> Dict:
    """Get plant area definitions."""
    return PLANT_AREAS


if __name__ == "__main__":
    # Test: print sample data
    import json
    machines = get_all_sample_machines()
    print(json.dumps(machines, indent=2))
