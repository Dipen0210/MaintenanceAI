# MaintenanceAI: Multimodal Equipment Health Monitoring

An end-to-end AI-powered predictive maintenance system using **audio, vibration, and sensor data** for industrial machine health monitoring. Features CNN-based fault diagnosis, LSTM/Transformer RUL prediction, a simulated plant dashboard, and optional direct inference APIs.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## Key Features

| Feature | Description |
|---------|-------------|
| Audio Anomaly Detection | Ensemble ML on MIMII dataset (fan, pump, valve) |
| Vibration Fault Diagnosis | 2D CNN classifier, 10 bearing fault classes (~98% accuracy) |
| RUL Prediction | LSTM + Transformer ensemble on NASA C-MAPSS (RMSE ~14–15) |
| Plant Intelligence | Multi-machine health scoring and risk-based maintenance queue |
| Simulated Live Plant | 14 sample machines across 4 areas, updated every 10 seconds |
| Modern Dashboard | Next.js UI with area grouping, machine detail, and WebSocket status |

---

## How It Works

This repo has three layers. Training is real. The dashboard is a **plant simulation** that uses the trained models. Direct `/predict` APIs are available if you want to score your own files or signals.

```
Train (Colab notebooks)
        │
        ▼
trained_models/  (12 weights)
        │
        ├──────────────────────────────┐
        ▼                              ▼
Demo plant (dashboard)          Direct inference APIs
FastAPI loads models            POST /predict/audio
Synthetic signals every 10s     POST /predict/vibration
Health mix ~70/20/10            POST /predict/rul
Plant scoring + maintenance
REST poll (5s) + WebSocket
        │
        ▼
Next.js dashboard
```

### 1. Train models (offline)

Notebooks in `notebooks/` train on public datasets and save weights to `trained_models/`:

1. Audio ensemble (fan, pump, valve) → `.pkl`
2. Vibration 2D CNN (10 CWRU bearing classes) → `.pth`
3. RUL LSTM + Transformer (C-MAPSS FD001–FD004) → `.pth`

### 2. Demo plant (what the dashboard shows)

On backend startup:

1. Loads up to 12 trained models from `trained_models/`
2. Initializes **14 sample machines** in **4 areas** (Production Line A/B, HVAC, Utilities)
3. A background task updates 2–4 random machines every **10 seconds**
4. Each update generates **synthetic** vibration spectrograms and sensor sequences (not live shop-floor sensors)
5. Health is sampled from a demo mix: ~70% healthy, ~20% warning, ~10% critical
6. The vibration CNN and RUL ensemble may be consulted, then blended with those health targets so the demo stays readable
7. Plant intelligence computes a health score, risk score, and prioritized maintenance queue
8. The frontend **polls REST every 5 seconds**. WebSocket broadcasts exist (`/ws/updates`) and the dashboard shows a connection badge

This is a demonstration of the monitoring UX, not a live industrial ingest pipeline.

### 3. Direct model inference (optional)

Use these endpoints to score real inputs without the plant simulation:

- `POST /predict/audio` — upload a `.wav` file
- `POST /predict/vibration` — send a vibration signal array
- `POST /predict/rul` — send a sensor sequence `[timesteps, features]`

---

## Model Performance

| Model | Dataset | Metric | Result |
|-------|---------|--------|--------|
| Audio Anomaly | MIMII DUE | AUC | > 0.96 |
| Vibration CNN | CWRU Bearing | Accuracy | ~98% |
| RUL LSTM | C-MAPSS FD001 | RMSE | ~15 cycles |
| RUL Transformer | C-MAPSS FD001 | RMSE | ~14 cycles |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js 16)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Dashboard   │  │ Area Pages   │  │ Machine Detail Pages   │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│                            │                                     │
│              HTTP REST (poll 5s) + WebSocket badge               │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ REST /plant  │  │ REST /predict│  │ WebSocket /ws/updates  │ │
│  │ + simulation │  │ audio/vib/RUL│  │ Plant Intelligence     │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│                            │                                     │
│                     Inference Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Audio        │  │ Vibration    │  │ RUL Estimator          │ │
│  │ Detector     │  │ Diagnoser    │  │ (LSTM x4 + Trans x4)   │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                    Trained Models (12 total)                     │
│  • audio_advanced_v2_{fan,pump,valve}.pkl                       │
│  • vibration_classifier.pth (2D CNN, 10 classes)                │
│  • rul_predictor_FD00{1,2,3,4}.pth (LSTM)                       │
│  • rul_transformer_FD00{1,2,3,4}.pth (Transformer)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Visuals

<img width="1470" height="684" alt="Dashboard overview" src="https://github.com/user-attachments/assets/3d761ef5-8b52-4113-b5dc-666433c4610e" />

<img width="1470" height="493" alt="Plant areas and machine grouping" src="https://github.com/user-attachments/assets/e9bb600b-20ed-4249-8df3-d1d49b1262bd" />

<img width="962" height="799" alt="Area detail" src="https://github.com/user-attachments/assets/9f099c64-9745-4632-b4e7-5bf78f73d541" />

<img width="962" height="800" alt="Area machines" src="https://github.com/user-attachments/assets/b9876320-54bd-4e8d-a289-eb71a3be9d44" />

<img width="783" height="797" alt="Machine health gauge" src="https://github.com/user-attachments/assets/e1d77826-d2c2-46b7-a9d3-d76926f39d59" />

<img width="777" height="781" alt="Machine detail" src="https://github.com/user-attachments/assets/8be1b595-93cc-4ed8-af65-a6752254ce1e" />

<img width="778" height="729" alt="Fault and RUL metrics" src="https://github.com/user-attachments/assets/3ed13ec4-390c-4a11-8b54-b456ee5227bb" />

<img width="848" height="749" alt="Maintenance recommendations" src="https://github.com/user-attachments/assets/40c16ee7-fb45-4320-bedc-31dc6e706988" />

<img width="765" height="676" alt="Alerts and status" src="https://github.com/user-attachments/assets/ad87d70e-9d31-4ee4-9062-c952b6380b70" />

---

## Project Structure

```
MaintanenceAI/
├── backend/
│   ├── main.py                    # FastAPI app, model load, plant simulation
│   ├── routes/
│   │   ├── predict.py             # POST /predict/audio, /vibration, /rul
│   │   └── websocket.py           # WebSocket /ws/updates
│   └── services/
│       ├── plant_intelligence.py  # Health scoring, risk, maintenance queue
│       └── sample_data.py         # 14 machines, 4 plant areas
├── frontend/
│   └── src/
│       ├── app/
│       │   ├── page.tsx           # Main dashboard (REST poll + WS badge)
│       │   ├── area/[id]/page.tsx # Area detail
│       │   └── machine/[id]/page.tsx
│       ├── components/Dashboard.tsx
│       ├── hooks/useWebSocket.ts
│       └── lib/api.ts             # REST + predict client
├── src/
│   ├── models/                    # PyTorch architectures
│   ├── inference/                 # Runtime wrappers for trained weights
│   ├── preprocessing/             # Audio, vibration, RUL pipelines
│   └── utils/                     # Config, data loader, explainer
├── notebooks/                     # Colab training notebooks
├── trained_models/                # 12 saved model weights
├── Data/                          # Local audio samples (gitignored)
├── LICENSE
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Trained models in `trained_models/` (included for the demo)

### 1. Clone and install backend

```bash
git clone https://github.com/Dipen0210/MaintenanceAI.git
cd MaintenanceAI

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Start backend

```bash
source venv/bin/activate
uvicorn backend.main:app --reload
```

- API: http://localhost:8000
- API docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws/updates

### 3. Start frontend

```bash
cd frontend
npm install
npm run dev
```

- Dashboard: http://localhost:3000

If the backend is down, the UI falls back to static demo data.

---

## API Endpoints

### Plant (dashboard)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service health and which models loaded |
| `/plant/summary` | GET | Overall plant health stats |
| `/plant/machines` | GET | All machine statuses |
| `/plant/machines/{id}` | GET | Single machine detail |
| `/plant/areas` | GET | Plant area definitions |
| `/plant/maintenance-queue` | GET | Prioritized maintenance tasks |
| `/plant/machines/{id}/update` | POST | Manually update a machine |
| `/plant/reset` | POST | Re-initialize all machines from the simulator |
| `/ws/updates` | WebSocket | Broadcasts machine updates, plant summary, and alerts |

### Direct inference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/audio` | POST | Anomaly score from a `.wav` upload |
| `/predict/vibration` | POST | Bearing fault class from a vibration signal |
| `/predict/rul` | POST | Remaining useful life from a sensor sequence |

---

## Dataset Downloads

### 1. MIMII DUE (Audio Anomaly)

[Zenodo - MIMII DUE](https://zenodo.org/record/4740355)

### 2. CWRU Bearing (Vibration)

[CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) or [Kaggle](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets)

### 3. NASA C-MAPSS (RUL)

[NASA Prognostics Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) or [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

---

## Training Models

Training notebooks in `notebooks/`:

1. `01_Colab_Audio_Anomaly_Training.ipynb` — ensemble anomaly detection
2. `02_Colab_Vibration_Classifier_Training.ipynb` — 2D CNN fault classification
3. `03_Colab_RUL_Prediction_Training.ipynb` — LSTM + Transformer RUL

Save trained models to `trained_models/`.

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| ML/DL | PyTorch, Transformer, LSTM, CNN, Autoencoder |
| Audio | librosa, SciPy |
| Backend | FastAPI, Uvicorn, WebSocket |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS v4 |
| Data | pandas, numpy, scikit-learn |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author — Dipen Prajapati

Built as a demonstration of industrial AI predictive maintenance capabilities.
