# 🏭 Predictive Maintenance AI System

An end-to-end AI-powered predictive maintenance system for industrial machines.

## Features

- 🔊 **Audio Anomaly Detection** - Detect abnormal machine sounds using autoencoders
- ⚙️ **Vibration Fault Diagnosis** - Classify bearing faults from vibration signals (10 classes)
- 📈 **RUL Prediction** - Predict Remaining Useful Life using Transformer/LSTM
- 🏭 **Plant Intelligence** - Multi-machine health monitoring & maintenance prioritization
- 📊 **Real-time Dashboard** - Next.js frontend with live updates

## Project Structure

```
MaintanenceAI/
├── src/
│   ├── models/            # PyTorch model architectures
│   ├── preprocessing/     # Data processing pipelines
│   ├── inference/         # Inference modules
│   └── utils/             # Config, loaders, explainer
├── backend/               # FastAPI backend
├── frontend/              # Next.js dashboard
├── notebooks/             # Colab training notebooks
└── trained_models/        # Saved model weights (.pth)
```

---

## 📥 Dataset Download Links

### 1. MIMII DUE (Audio Anomaly Detection)
| Machine | Download |
|---------|----------|
| Fan | [Zenodo - MIMII DUE Fan](https://zenodo.org/record/4740355) |
| Pump | [Zenodo - MIMII DUE Pump](https://zenodo.org/record/4740355) |
| Valve | [Zenodo - MIMII DUE Valve](https://zenodo.org/record/4740355) |

**Structure after download:**
```
Data/
├── fan/
│   ├── train/         # Normal sounds only
│   ├── source_test/   # Normal + Anomaly
│   └── target_test/   # Domain shift test
├── pump/
└── valve/
```

### 2. CWRU Bearing (Vibration Fault Classification)
**Download:** [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file)

**Required files (48k Drive End):**
- Normal: `97.mat`, `98.mat`, `99.mat`, `100.mat`
- Ball Fault: `B007`, `B014`, `B021`
- Inner Race: `IR007`, `IR014`, `IR021`  
- Outer Race: `OR007`, `OR014`, `OR021`

**Alternative:** [Kaggle - CWRU](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets)

### 3. NASA C-MAPSS (RUL Prediction)
**Download:** [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) or [Kaggle - C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

**Files needed:**
```
Data/CMaps/
├── train_FD001.txt, train_FD002.txt, train_FD003.txt, train_FD004.txt
├── test_FD001.txt, test_FD002.txt, test_FD003.txt, test_FD004.txt
└── RUL_FD001.txt, RUL_FD002.txt, RUL_FD003.txt, RUL_FD004.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data
Download datasets from links above and place in `Data/` folder.

### 3. Train Models (Google Colab)
Upload notebooks from `notebooks/` to Colab:
1. `01_Colab_Audio_Anomaly_Training.ipynb`
2. `02_Colab_Vibration_Classifier_Training.ipynb`
3. `03_Colab_RUL_Prediction_Training.ipynb`

Save trained `.pth` files to `trained_models/`

### 4. Start Backend
```bash
cd backend
uvicorn main:app --reload
```
API: http://localhost:8000

### 5. Start Frontend
```bash
cd frontend
npm install
npm run dev
```
Dashboard: http://localhost:3000

---

## 📊 Model Performance

| Model | Dataset | Metric | Result |
|-------|---------|--------|--------|
| Audio Anomaly | MIMII | AUC | >0.75 |
| Vibration Classifier | CWRU | Accuracy | ~98% |
| RUL Transformer | C-MAPSS FD001 | RMSE | ~15 cycles |
| RUL Transformer | C-MAPSS FD003 | RMSE | ~14 cycles |

---

## Tech Stack

- **ML**: PyTorch, Transformer, LSTM, Autoencoder
- **Backend**: FastAPI, Uvicorn, WebSocket
- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **Audio**: librosa
- **Data**: pandas, numpy, scipy, scikit-learn
