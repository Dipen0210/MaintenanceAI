"""
FastAPI Main Application.
Predictive Maintenance System Backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.routes import predict_router, websocket_router
from backend.services.plant_intelligence import PlantIntelligence


# Global plant intelligence instance
plant_intel = PlantIntelligence()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 Starting Predictive Maintenance API...")
    yield
    # Shutdown
    print("👋 Shutting down...")


app = FastAPI(
    title="Predictive Maintenance API",
    description="""
    AI-powered predictive maintenance system for industrial machines.
    
    ## Features
    - 🔊 **Audio Anomaly Detection**: Detect abnormal machine sounds
    - ⚙️ **Vibration Fault Diagnosis**: Classify bearing faults
    - 📈 **RUL Prediction**: Predict remaining useful life
    - 🏭 **Plant Intelligence**: Multi-machine health monitoring
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
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


# Plant-level endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Predictive Maintenance API"}


@app.get("/plant/summary")
async def get_plant_summary():
    """Get overall plant health summary."""
    return plant_intel.get_plant_summary()


@app.get("/plant/machines")
async def get_all_machines():
    """Get status of all machines."""
    return plant_intel.get_all_machines()


@app.get("/plant/machines/{machine_id}")
async def get_machine(machine_id: str):
    """Get status of a specific machine."""
    machine = plant_intel.get_machine(machine_id)
    if machine is None:
        return {"error": "Machine not found"}
    return machine


@app.get("/plant/maintenance-queue")
async def get_maintenance_queue():
    """Get prioritized maintenance recommendations."""
    queue = plant_intel.get_maintenance_queue()
    return [r.to_dict() for r in queue]


@app.post("/plant/machines/{machine_id}/update")
async def update_machine(
    machine_id: str,
    machine_type: str,
    anomaly_score: float = 0.0,
    fault_type: str = None,
    fault_confidence: float = 0.0,
    rul_cycles: float = None
):
    """Update machine status (called after predictions)."""
    status = plant_intel.update_machine(
        machine_id=machine_id,
        machine_type=machine_type,
        anomaly_score=anomaly_score,
        fault_type=fault_type,
        fault_confidence=fault_confidence,
        rul_cycles=rul_cycles
    )
    return status.to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
