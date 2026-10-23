"""Validated direct inference APIs for the original supported datasets."""
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
import os
import tempfile
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field, FiniteFloat, model_validator

router = APIRouter(prefix='/predict', tags=['predictions'])
MODEL_DIR = Path(__file__).resolve().parents[2] / 'trained_models'


class AudioPredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    label: str


class VibrationPredictionRequest(BaseModel):
    signal: list[FiniteFloat] = Field(min_length=1024, max_length=65536)
    sample_rate: Literal[48000] = 48000


class VibrationPredictionResponse(BaseModel):
    fault_type: str
    confidence: float
    description: str
    is_faulty: bool
    severity: str | None = None


class RULPredictionRequest(BaseModel):
    sequence: list[list[FiniteFloat]] = Field(min_length=1, max_length=10000)

    @model_validator(mode='after')
    def rectangular(self):
        if any(len(row) != 17 for row in self.sequence):
            raise ValueError('C-MAPSS input requires exactly 17 features per timestep')
        return self


class RULPredictionResponse(BaseModel):
    rul_cycles: float
    health_score: float
    status: str
    recommendation: str


@lru_cache(maxsize=3)
def audio_detector(machine_type):
    from src.inference.anomaly_detector import AudioAnomalyDetector
    path = os.environ.get('AUDIO_MODEL_PATH', str(MODEL_DIR / f'audio_iforest_v1_{machine_type}.joblib'))
    detector = AudioAnomalyDetector(path)
    if detector.model_data['machine_type'] != machine_type:
        raise ValueError('Audio artifact is incompatible with requested machine type')
    return detector


@lru_cache(maxsize=1)
def fault_diagnoser():
    from src.inference.fault_diagnoser import FaultDiagnoser
    return FaultDiagnoser(os.environ.get('VIBRATION_MODEL_PATH', str(MODEL_DIR / 'vibration_classifier.pth')))


@lru_cache(maxsize=1)
def rul_estimator():
    from src.inference.rul_estimator import RULEstimator
    return RULEstimator(os.environ.get('RUL_MODEL_PATH', str(MODEL_DIR / 'rul_predictor_FD001.pth')))


def load_model(loader, *args):
    try:
        return loader(*args)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(503, detail='Verified compatible model is unavailable') from exc


@router.post('/audio', response_model=AudioPredictionResponse)
def predict_audio_anomaly(file: UploadFile = File(...),
                          machine_type: Literal['fan','pump','valve'] = 'fan',
                          section: Literal['00','01','02'] = '00'):
    if not (file.filename or '').lower().endswith('.wav'):
        raise HTTPException(400, detail='Only .wav files are supported')
    content = file.file.read(20 * 1024 * 1024 + 1)
    if not content or len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, detail='Audio upload must contain 1 byte to 20 MiB')
    detector = load_model(audio_detector, machine_type)
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
        tmp.write(content); tmp.flush()
        try:
            anomaly, score, label = detector.predict(tmp.name, section)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, detail=str(exc)) from exc
    return AudioPredictionResponse(is_anomaly=anomaly, anomaly_score=score, label=label)


@router.post('/vibration', response_model=VibrationPredictionResponse)
def predict_vibration_fault(request: VibrationPredictionRequest):
    model = load_model(fault_diagnoser)
    try:
        result = model.predict(np.asarray(request.signal, dtype=np.float32))
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return VibrationPredictionResponse(**result, severity=model.get_severity(result['fault_type']))


@router.post('/rul', response_model=RULPredictionResponse)
def predict_rul(request: RULPredictionRequest):
    model = load_model(rul_estimator)
    try:
        result = model.predict(np.asarray(request.sequence, dtype=np.float64))
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return RULPredictionResponse(**result)
