
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "Data"

# Data Paths
MIMII_FAN_DIR = DATA_DIR / "fan"
MIMII_PUMP_DIR = DATA_DIR / "pump"
MIMII_VALVE_DIR = DATA_DIR / "valve"
CWRU_DIR = DATA_DIR / "CWRU"
CMAPSS_DIR = DATA_DIR / "CMaps"

# Audio Hyperparameters (MIMII)
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
AUDIO_FRAME_SIZE = 64  # Number of frames for one input sample

# Vibration Hyperparameters (CWRU)
VIBRATION_WINDOW_SIZE = 2048

# RUL Hyperparameters (C-MAPSS)
RUL_SEQUENCE_LENGTH = 50
MAX_RUL = 125

# Model Save Paths
MODEL_DIR = BASE_DIR / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)
