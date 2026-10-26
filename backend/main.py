"""
FastAPI Main Application.
Predictive Maintenance System Backend - Separates verified inference from explicitly simulated plant observations.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import random
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.schemas import MachineInput
from backend.routes import predict_router, websocket_router
from backend.services.plant_intelligence import PlantIntelligence
from backend.services.sample_data import SAMPLE_MACHINES, PLANT_AREAS

# Import inference modules
from src.inference.fault_diagnoser import FaultDiagnoser
from src.inference.rul_estimator import RULEstimator

# Paths to trained models
MODEL_DIR = Path(__file__).parent.parent / "trained_models"

# Model paths
VIBRATION_MODEL_PATH = MODEL_DIR / "vibration_classifier.pth"
AUDIO_MODEL_PATHS = {
    "fan": MODEL_DIR / "audio_iforest_v1_fan.joblib",
    "pump": MODEL_DIR / "audio_iforest_v1_pump.joblib",
    "valve": MODEL_DIR / "audio_iforest_v1_valve.joblib",
}
RUL_MODEL_PATHS = {
    "lstm": {
        "FD001": MODEL_DIR / "rul_predictor_FD001.pth",
        "FD002": MODEL_DIR / "rul_predictor_FD002.pth",
        "FD003": MODEL_DIR / "rul_predictor_FD003.pth",
        "FD004": MODEL_DIR / "rul_predictor_FD004.pth",
    },
    "transformer": {
        "FD001": MODEL_DIR / "rul_transformer_FD001.pth",
        "FD002": MODEL_DIR / "rul_transformer_FD002.pth",
        "FD003": MODEL_DIR / "rul_transformer_FD003.pth",
        "FD004": MODEL_DIR / "rul_transformer_FD004.pth",
    }
}

# Global instances
plant_intel = PlantIntelligence()
fault_diagnoser = None
audio_models = {}  # Machine type -> model
rul_models = {}    # Model name -> RULEstimator
simulation_task = None


def load_models():
    """Load ALL trained models."""
    global fault_diagnoser, audio_models, rul_models
    
    print("📂 Loading trained models...")
    
    # 1. Load Vibration Classifier
    try:
        if VIBRATION_MODEL_PATH.exists():
            fault_diagnoser = FaultDiagnoser(str(VIBRATION_MODEL_PATH))
            print(f"  ✅ Vibration classifier loaded (2D CNN)")
        else:
            print(f"  ⚠️ Vibration model not found")
    except Exception as e:
        print(f"  ❌ Vibration model error: {e}")
    
    # 2. Load Audio Anomaly Models (per machine type)
    for machine_type, path in AUDIO_MODEL_PATHS.items():
        try:
            if path.exists():
                from src.inference.anomaly_detector import AudioAnomalyDetector
                audio_models[machine_type] = AudioAnomalyDetector(str(path))
                print(f"  ✅ Audio model loaded: {machine_type}")
            else:
                print(f"  ⚠️ Audio model not found: {machine_type}")
        except Exception as e:
            print(f"  ❌ Audio model error ({machine_type}): {e}")
    
    # 3. Load RUL Models (LSTM - all 4 datasets)
    for dataset, path in RUL_MODEL_PATHS["lstm"].items():
        try:
            if path.exists():
                rul_models[f"lstm_{dataset}"] = RULEstimator(str(path))
                print(f"  ✅ RUL LSTM loaded: {dataset}")
        except Exception as e:
            print(f"  ❌ RUL LSTM error ({dataset}): {e}")
    
    # 4. Load RUL Transformer Models
    for dataset, path in RUL_MODEL_PATHS["transformer"].items():
        try:
            if path.exists():
                rul_models[f"transformer_{dataset}"] = RULEstimator(str(path))
                print(f"  ✅ RUL Transformer loaded: {dataset}")
        except Exception as e:
            print(f"  ❌ RUL Transformer error ({dataset}): {e}")
    
    total_loaded = (1 if fault_diagnoser else 0) + len(audio_models) + len(rul_models)
    print(f"📊 Total models loaded: {total_loaded}/12")


def get_model_predictions(machine_type: str):
    """Legacy dashboard simulation only; never apply turbofan RUL to plant assets.

    Real learned predictions are available through the /predict routes. These
    generated health observations are deliberately isolated from inference.
    """
    level = random.random()
    if level < .7:
        anomaly, rul, fault = random.uniform(.01, .08), random.uniform(90, 130), 'Normal'
    elif level < .9:
        anomaly, rul = random.uniform(.08, .18), random.uniform(50, 90)
        fault = random.choice(['Normal', 'Ball_007', 'IR_007'])
    else:
        anomaly, rul = random.uniform(.18, .4), random.uniform(15, 50)
        fault = random.choice(['Ball_021', 'IR_021', 'OR_021'])
    return dict(anomaly_score=round(anomaly, 3), fault_type=fault,
                fault_confidence=round(random.uniform(.8, .99), 2), rul_cycles=round(rul, 1))


async def simulate_machine_updates():
    """Background task to broadcast explicitly simulated machine observations."""
    from backend.routes.websocket import broadcast_machine_update, broadcast_plant_summary, broadcast_alert
    
    while True:
        await asyncio.sleep(10)  # Update every 10 seconds
        from backend.core.demo import simulate_tick
        await asyncio.to_thread(simulate_tick)
        
        # Update 2-4 random machines
        num_updates = random.randint(2, 4)
        machines_to_update = random.sample(SAMPLE_MACHINES, min(num_updates, len(SAMPLE_MACHINES)))
        
        for machine in machines_to_update:
            predictions = get_model_predictions(machine["machine_type"])
            
            status = plant_intel.update_machine(
                machine_id=machine["machine_id"],
                machine_type=machine["machine_type"],
                area=machine.get("area", "unknown"),
                anomaly_score=predictions["anomaly_score"],
                fault_type=predictions["fault_type"],
                fault_confidence=predictions["fault_confidence"],
                rul_cycles=predictions["rul_cycles"]
            )
            
            # Broadcast update via WebSocket
            try:
                await broadcast_machine_update(status.to_dict())
                
                # Send alert if critical
                if status.health_score < 0.5:
                    await broadcast_alert(
                        machine["machine_id"],
                        "critical" if status.health_score < 0.3 else "warning",
                        f"Health: {status.health_score:.0%}, RUL: {status.rul_cycles:.0f} cycles"
                    )
            except Exception as e:
                pass  # WebSocket errors shouldn't break simulation
        
        # Broadcast updated plant summary
        try:
            summary = plant_intel.get_plant_summary()
            await broadcast_plant_summary(summary)
        except Exception:
            pass


def initialize_sample_machines():
    """Initialize explicitly simulated plant observations."""
    print("📦 Initializing simulated machines (not model predictions)...")
    
    for machine in SAMPLE_MACHINES:
        predictions = get_model_predictions(machine["machine_type"])
        
        plant_intel.update_machine(
            machine_id=machine["machine_id"],
            machine_type=machine["machine_type"],
            area=machine.get("area", "unknown"),
            anomaly_score=predictions["anomaly_score"],
            fault_type=predictions["fault_type"],
            fault_confidence=predictions["fault_confidence"],
            rul_cycles=predictions["rul_cycles"]
        )
    
    print(f"✅ Initialized {len(SAMPLE_MACHINES)} machines")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global simulation_task
    
    print("🚀 Starting Predictive Maintenance API...")
    print("=" * 50)
    
    from backend.core.database import migrate
    from backend.core.demo import seed_demo
    migrate()
    seed_demo()
    from backend.ml.registry import initialize_registry
    initialize_registry()
    from backend.risk.service import seed_risk
    seed_risk()
    from backend.scheduling.demo import seed_planning
    seed_planning()
    from backend.copilot.demo import seed_copilot
    try:
        seed_copilot()
    except (ImportError,OSError,ValueError,RuntimeError,HTTPException) as exc:
        print('Copilot demo evidence unavailable:',type(exc).__name__)

    # Load verified public-dataset models.
    load_models()
    
    print("=" * 50)
    
    # Initialize machines
    initialize_sample_machines()
    
    # Start simulation
    simulation_task = asyncio.create_task(simulate_machine_updates())
    print("🔄 Simulation started (updates every 10s)")
    
    from backend.preventive.service import worker
    preventive_task=asyncio.create_task(worker())
    try:
        yield
    finally:
        preventive_task.cancel()
        try:await preventive_task
        except asyncio.CancelledError:pass
    
    if simulation_task:
        simulation_task.cancel()
        try:
            await simulation_task
        except asyncio.CancelledError:
            pass
    print("👋 Shutting down...")


app = FastAPI(
    title="Predictive Maintenance API",
    description="AI-powered predictive maintenance using 12 trained ML models.",
    version="1.0.0",
    lifespan=lifespan
)

# Validation errors must not echo NaN/Inf back into JSON serialization.
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": list(error['loc']), "msg": error['msg'], "type": error['type']}
        for error in exc.errors()
    ]})


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router)
app.include_router(websocket_router)
from backend.routes.v2 import router as v2_router
app.include_router(v2_router)
from backend.routes.ml import router as ml_router
app.include_router(ml_router)
from backend.routes.risk import router as risk_router
app.include_router(risk_router)
from backend.routes.copilot import router as copilot_router
app.include_router(copilot_router)
from backend.routes.observability import router as observability_router
app.include_router(observability_router)
from backend.routes.scheduling import router as scheduling_router
app.include_router(scheduling_router)
from backend.routes.administration import router as administration_router
app.include_router(administration_router)


# Endpoints
@app.get("/")
async def root():
    return {
        "status": "healthy" if fault_diagnoser and len(audio_models) == 3 and len(rul_models) == 8 else "degraded",
        "plant_data_mode": "simulation",
        "service": "Predictive Maintenance API",
        "models_loaded": {
            "vibration_classifier": fault_diagnoser is not None,
            "audio_models": list(audio_models.keys()),
            "rul_models": list(rul_models.keys()),
            "total": (1 if fault_diagnoser else 0) + len(audio_models) + len(rul_models)
        }
    }


@app.get("/plant/summary")
async def get_plant_summary():
    return plant_intel.get_plant_summary()


@app.get("/plant/areas")
async def get_plant_areas():
    return PLANT_AREAS


@app.get("/plant/machines")
async def get_all_machines():
    return plant_intel.get_all_machines()


@app.get("/plant/machines/{machine_id}")
async def get_machine(machine_id: str):
    machine = plant_intel.get_machine(machine_id)
    if machine is None:
        raise HTTPException(status_code=404, detail="Machine not found")
    return machine


@app.get("/plant/maintenance-queue")
async def get_maintenance_queue():
    queue = plant_intel.get_maintenance_queue()
    return [r.to_dict() for r in queue]


@app.post("/plant/machines/{machine_id}/update")
async def update_machine(
    machine_id: str,
    machine_type: str | None = None,
    area: str | None = None,
    anomaly_score: float | None = Query(None, ge=0, le=1, allow_inf_nan=False),
    fault_type: str | None = None,
    fault_confidence: float | None = Query(None, ge=0, le=1, allow_inf_nan=False),
    rul_cycles: float | None = Query(None, ge=0, le=1000000, allow_inf_nan=False),
):
    existing = plant_intel.get_machine(machine_id)
    values = {k: v for k, v in (existing or {}).items() if k in MachineInput.model_fields}
    values['machine_id'] = machine_id
    updates = dict(machine_type=machine_type, area=area, anomaly_score=anomaly_score,
                   fault_type=fault_type, fault_confidence=fault_confidence, rul_cycles=rul_cycles)
    values.update({k: v for k, v in updates.items() if v is not None})
    try:
        validated = MachineInput.model_validate(values)
    except ValidationError:
        raise HTTPException(status_code=422, detail="Invalid machine fields")
    status = plant_intel.update_machine(**validated.model_dump())
    from backend.routes.websocket import broadcast_machine_update, broadcast_plant_summary
    await broadcast_machine_update(status.to_dict())
    await broadcast_plant_summary(plant_intel.get_plant_summary())
    return status.to_dict()


@app.post("/plant/reset")
async def reset_machines():
    """Reset the explicitly simulated plant state."""
    plant_intel.machines.clear()
    plant_intel.history.clear()
    initialize_sample_machines()
    return {"status": "reset", "machines": len(SAMPLE_MACHINES)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from backend.routes.simulation import router as simulation_router
from backend.routes.machines import router as machines_router
app.include_router(simulation_router)
app.include_router(machines_router)

from backend.routes.visuals import router as visuals_router
app.include_router(visuals_router)

from backend.routes.issues import router as issues_router
app.include_router(issues_router)

from backend.routes.preventive import router as preventive_router
from backend.routes.asset_access import router as asset_access_router
app.include_router(preventive_router)
from backend.routes.meters import router as meters_router
app.include_router(meters_router)
from backend.routes.inspections import router as inspections_router
app.include_router(inspections_router)
app.include_router(asset_access_router)
from backend.routes.coordination import router as coordination_router
app.include_router(coordination_router)
