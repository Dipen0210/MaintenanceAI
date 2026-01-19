# 🏭 MaintenanceAI: Multimodal Equipment Health Monitoring

An end-to-end AI-powered predictive maintenance system using **audio, vibration, and sensor data** for industrial machine health monitoring. Features real-time WebSocket streaming, CNN-based fault diagnosis, and LSTM/Transformer RUL prediction.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔊 **Audio Anomaly Detection** | Ensemble ML on MIMII dataset (fan, pump, valve) |
| ⚙️ **Vibration Fault Diagnosis** | 2D CNN classifier, 10 bearing fault classes (98% accuracy) |
| 📈 **RUL Prediction** | LSTM + Transformer ensemble on NASA C-MAPSS (RMSE < 15) |
| 🏭 **Plant Intelligence** | Multi-machine health scoring & risk-based prioritization |
| � **WebSocket Streaming** | Real-time machine status updates via WebSocket |
| 📊 **Modern Dashboard** | Glassmorphism UI with area-based machine grouping |

---

## 🎯 Model Performance

| Model | Dataset | Metric | Result |
|-------|---------|--------|--------|
| Audio Anomaly | MIMII DUE | AUC | > 0.96 |
| Vibration CNN | CWRU Bearing | Accuracy | ~98% |
| RUL LSTM | C-MAPSS FD001 | RMSE | ~15 cycles |
| RUL Transformer | C-MAPSS FD001 | RMSE | ~14 cycles |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Dashboard   │  │ Area Pages   │  │ Machine Detail Pages   │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│                            │                                     │
│                    HTTP REST + WebSocket                         │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ REST API     │  │ WebSocket    │  │ Plant Intelligence     │ │
│  │ /plant/*     │  │ /ws/updates  │  │ Health Scoring         │ │
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

## 📁 Project Structure

```
MaintanenceAI/
├── backend/
│   ├── main.py                 # FastAPI app + simulation
│   ├── routes/
│   │   ├── predict.py          # Prediction endpoints
│   │   └── websocket.py        # WebSocket real-time streaming
│   └── services/
│       ├── plant_intelligence.py  # Health scoring & maintenance queue
│       └── sample_data.py         # Machine definitions (14 machines, 4 areas)
├── frontend/
│   └── src/
│       ├── app/
│       │   ├── page.tsx           # Main dashboard
│       │   ├── area/[id]/page.tsx # Area detail pages
│       │   └── machine/[id]/page.tsx # Machine detail pages
│       ├── components/Dashboard.tsx  # UI components
│       ├── hooks/useWebSocket.ts     # WebSocket hook
│       └── lib/api.ts                # API client
├── src/
│   ├── models/                 # PyTorch architectures
│   │   ├── audio_autoencoder.py
│   │   ├── vibration_classifier.py
│   │   └── rul_predictor.py
│   ├── inference/              # Inference modules
│   │   ├── anomaly_detector.py
│   │   ├── fault_diagnoser.py
│   │   └── rul_estimator.py
│   └── preprocessing/          # Data pipelines
├── trained_models/             # Saved model weights (12 models)
├── notebooks/                  # Google Colab training notebooks
└── requirements.txt
```

---

## � Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Trained models in `trained_models/`

### 1. Clone & Install Backend
```bash
git clone https://github.com/yourusername/MaintanenceAI.git
cd MaintanenceAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Backend
```bash
source venv/bin/activate
uvicorn backend.main:app --reload
```
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws/updates

### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
```
- Dashboard: http://localhost:3000

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/plant/summary` | GET | Overall plant health stats |
| `/plant/machines` | GET | All machine statuses |
| `/plant/machines/{id}` | GET | Single machine detail |
| `/plant/areas` | GET | Plant area definitions |
| `/plant/maintenance-queue` | GET | Prioritized maintenance tasks |
| `/ws/updates` | WebSocket | Real-time machine updates |

---

## 📥 Dataset Downloads

### 1. MIMII DUE (Audio Anomaly)
[Zenodo - MIMII DUE](https://zenodo.org/record/4740355)

### 2. CWRU Bearing (Vibration)
[CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) or [Kaggle](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets)

### 3. NASA C-MAPSS (RUL)
[NASA Prognostics Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) or [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

---

## 🧠 Training Models

Training notebooks in `notebooks/`:
1. `01_Colab_Audio_Anomaly_Training.ipynb` - Ensemble anomaly detection
2. `02_Colab_Vibration_Classifier_Training.ipynb` - 2D CNN fault classification
3. `03_Colab_RUL_Prediction_Training.ipynb` - LSTM + Transformer RUL

Save trained models to `trained_models/`

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| **ML/DL** | PyTorch, Transformer, LSTM, CNN, Autoencoder |
| **Audio** | librosa, SciPy |
| **Backend** | FastAPI, Uvicorn, WebSocket |
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS |
| **Data** | pandas, numpy, scikit-learn |

---

## � Screenshots

### Dashboard Overview
- Real-time plant health statistics
- Critical/warning alerts banner
- Area-based machine grouping
- WebSocket connection indicator

### Area Pages
- Area-specific health metrics
- Machine cards sorted by health
- Maintenance queue for area

### Machine Detail
- Circular health gauge
- Anomaly score, RUL, fault type
- AI-generated recommendations

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author - Dipen Prajapati

Built as a demonstration of industrial AI predictive maintenance capabilities.
