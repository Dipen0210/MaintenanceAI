"""
Prediction API Routes.
Endpoints for audio anomaly, vibration fault, and RUL predictions.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import tempfile
import os

router = APIRouter(prefix="/predict", tags=["predictions"])

# Lazy-loaded inference modules (loaded when first needed)
_audio_detector = None
_fault_diagnoser = None
_rul_estimator = None


# Request/Response Models
class AudioPredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    label: str


class VibrationPredictionRequest(BaseModel):
    signal: List[float]


class VibrationPredictionResponse(BaseModel):
    fault_type: str
    confidence: float
    description: str
    is_faulty: bool
    severity: Optional[str] = None


class RULPredictionRequest(BaseModel):
    sequence: List[List[float]]  # Shape: [seq_len, n_features]


class RULPredictionResponse(BaseModel):
    rul_cycles: float
    health_score: float
    status: str
    recommendation: str


# Endpoints
@router.post("/audio", response_model=AudioPredictionResponse)
async def predict_audio_anomaly(file: UploadFile = File(...)):
    """
    Detect anomalies in audio file.
    
    Upload a .wav file of machine operating sounds.
    Returns anomaly score and classification.
    """
    global _audio_detector
    
    # Validate file type
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Lazy load detector
        if _audio_detector is None:
            from src.inference.anomaly_detector import AudioAnomalyDetector
            model_path = os.environ.get(
                'AUDIO_MODEL_PATH',
                'trained_models/audio_autoencoder_fan.pth'
            )
            _audio_detector = AudioAnomalyDetector(model_path=model_path)
        
        # Get prediction
        is_anomaly, score, label = _audio_detector.predict(tmp_path)
        
        return AudioPredictionResponse(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            label=label
        )
    
    finally:
        # Clean up
        os.unlink(tmp_path)


@router.post("/vibration", response_model=VibrationPredictionResponse)
async def predict_vibration_fault(request: VibrationPredictionRequest):
    """
    Diagnose fault from vibration signal.
    
    Send a vibration signal array (2048 samples recommended).
    Returns fault type and confidence.
    """
    global _fault_diagnoser
    
    signal = np.array(request.signal, dtype=np.float32)
    
    # Validate signal length
    if len(signal) < 100:
        raise HTTPException(status_code=400, detail="Signal too short (min 100 samples)")
    
    # Lazy load diagnoser
    if _fault_diagnoser is None:
        from src.inference.fault_diagnoser import FaultDiagnoser
        model_path = os.environ.get(
            'VIBRATION_MODEL_PATH',
            'trained_models/vibration_classifier.pth'
        )
        _fault_diagnoser = FaultDiagnoser(model_path=model_path)
    
    # Get prediction
    result = _fault_diagnoser.predict(signal)
    
    return VibrationPredictionResponse(
        fault_type=result['fault_type'],
        confidence=result['confidence'],
        description=result['description'],
        is_faulty=result['is_faulty'],
        severity=_fault_diagnoser.get_severity(result['fault_type'])
    )


@router.post("/rul", response_model=RULPredictionResponse)
async def predict_rul(request: RULPredictionRequest):
    """
    Predict Remaining Useful Life from sensor sequence.
    
    Send a sequence of sensor readings (50 timesteps x N features).
    Returns RUL in cycles and health status.
    """
    global _rul_estimator
    
    sequence = np.array(request.sequence, dtype=np.float32)
    
    # Validate sequence
    if sequence.ndim != 2:
        raise HTTPException(status_code=400, detail="Sequence must be 2D array [timesteps, features]")
    
    # Lazy load estimator
    if _rul_estimator is None:
        from src.inference.rul_estimator import RULEstimator
        model_path = os.environ.get(
            'RUL_MODEL_PATH',
            'trained_models/rul_predictor_FD001.pth'
        )
        _rul_estimator = RULEstimator(model_path=model_path)
    
    # Get prediction
    result = _rul_estimator.predict(sequence)
    
    return RULPredictionResponse(
        rul_cycles=result['rul_cycles'],
        health_score=result['health_score'],
        status=result['status'],
        recommendation=result['recommendation']
    )
