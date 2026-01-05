# 🏭 Predictive Maintenance AI System

An end-to-end AI-powered predictive maintenance system for industrial machines.

## Features

- 🔊 **Audio Anomaly Detection** - Detect abnormal machine sounds using autoencoders
- ⚙️ **Vibration Fault Diagnosis** - Classify bearing faults from vibration signals
- 📈 **RUL Prediction** - Predict Remaining Useful Life using LSTM
- 🏭 **Plant Intelligence** - Multi-machine health monitoring & maintenance prioritization
- 📊 **Real-time Dashboard** - Next.js frontend with live updates

## Project Structure

```
MaintanenceAI/
├── Data/                          # Datasets
│   ├── CWRU/                      # Bearing vibration data
│   ├── CMaps/                     # NASA C-MAPSS (RUL)
│   ├── fan/, pump/, valve/        # MIMII audio data
│
├── src/
│   ├── models/                    # PyTorch model architectures
│   ├── preprocessing/             # Data processing pipelines
│   ├── inference/                 # Inference modules
│   └── utils/                     # Config, loaders, explainer
│
├── backend/                       # FastAPI backend
│   ├── main.py
│   ├── routes/
│   └── services/
│
├── frontend/                      # Next.js dashboard
│
├── notebooks/                     # Colab training notebooks
│   ├── 01_Colab_Audio_Anomaly_Training.ipynb
│   ├── 02_Colab_Vibration_Classifier_Training.ipynb
│   └── 03_Colab_RUL_Prediction_Training.ipynb
│
└── trained_models/                # Saved model weights
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models (Google Colab)

Upload the notebooks from `notebooks/` to Google Colab:
1. **Audio**: Train on MIMII dataset → `audio_autoencoder_fan.pth`
2. **Vibration**: Train on CWRU dataset → `vibration_classifier.pth`
3. **RUL**: Train on C-MAPSS dataset → `rul_predictor_FD001.pth`

Save trained models to `trained_models/` folder.

### 3. Start Backend

```bash
cd backend
uvicorn main:app --reload
```

API available at: http://localhost:8000

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard at: http://localhost:3000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/audio` | POST | Detect audio anomalies |
| `/predict/vibration` | POST | Diagnose vibration faults |
| `/predict/rul` | POST | Predict RUL |
| `/plant/summary` | GET | Plant health summary |
| `/plant/machines` | GET | All machine statuses |
| `/plant/maintenance-queue` | GET | Prioritized maintenance |

## Datasets

- **CWRU Bearing**: Vibration fault classification (10 classes)
- **MIMII DUE**: Audio anomaly detection (fan, pump, valve)
- **NASA C-MAPSS**: Turbofan RUL prediction

## Tech Stack

- **ML**: PyTorch
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Next.js + Tailwind CSS
- **Audio**: librosa, torchaudio
- **Data**: pandas, numpy, scipy
