# Local Setup Guide

> Step-by-step instructions to build and run the Pump Fault Risk Prediction Service on your machine.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|:------------|:--------|:------------|
| **OS** | Windows 10+, macOS 12+, Ubuntu 20.04+ | Any |
| **Python** | 3.11 | 3.11 (tested with 3.11 and 3.14.2) |
| **RAM** | 4 GB free | 8 GB |
| **Disk** | 5 GB free | 10 GB |
| **Docker** (optional) | Docker Desktop 4.x | Latest |
| **Git** | 2.30+ | Latest |

---

## Option A: Run with Docker (Recommended)

The fastest way to get everything running. One command starts both the API and the Streamlit UI.

### Step 1: Clone the Repository

```bash
git clone https://github.com/akashvignesh/multimodal-pump-fault-prediction.git
cd multimodal-pump-fault-prediction
```

### Step 2: Download Data

📥 Download the `data/` folder from **[Google Drive](https://drive.google.com/drive/folders/19V-kQsAaLnxQI_4dxVCp-VjhPwJq3U4r?usp=sharing)**

Place the downloaded `data/` folder in the project root:

```
multimodal-pump-fault-prediction/
├── data/                          ← Downloaded from Google Drive
│   ├── baseline_model/
│   │   └── sensor_data/
│   │       └── sensor.csv
│   └── multimodal_model/
│       ├── sensor_data.csv
│       ├── image_mapping.csv
│       └── images/
│           ├── pump_001.png
│           ├── pump_002.png
│           └── ... (241 images)
├── artifacts/
├── src/
├── app/
├── Dockerfile
├── docker-compose.yml
└── ...
```

### Step 3: Build and Run

```bash
docker compose up --build -d
```

This will:
- Build the Docker image (~3-5 min first time)
- Start the FastAPI server on **port 8000**
- Start the Streamlit UI on **port 8501**
- Run a health check to verify everything is working

### Step 4: Verify

Wait ~60 seconds for the CLIP model to load, then:

```bash
# Check container status
docker ps

# You should see:
# NAMES              STATUS                    PORTS
# pump-fault-risk    Up X minutes (healthy)    0.0.0.0:8000->8000, 0.0.0.0:8501->8501
```

### Step 5: Access the Application

| Service | URL |
|:--------|:----|
| **Streamlit UI** | http://localhost:8501 |
| **FastAPI Docs** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |

### Docker Commands Cheat Sheet

```bash
# Start
docker compose up -d

# Stop
docker compose down

# Rebuild after code changes
docker compose up --build -d

# View logs
docker compose logs -f

# Check status
docker ps
```

---

## Option B: Run Locally (Without Docker)

### Step 1: Clone the Repository

```bash
git clone https://github.com/akashvignesh/multimodal-pump-fault-prediction.git
cd multimodal-pump-fault-prediction
```

### Step 2: Download Data

📥 Download the `data/` folder from **[Google Drive](https://drive.google.com/drive/folders/19V-kQsAaLnxQI_4dxVCp-VjhPwJq3U4r?usp=sharing)**

Place it in the project root (same structure as shown in Option A, Step 2).

### Step 3: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
# Production dependencies (API + Streamlit)
pip install --upgrade pip
pip install -r requirements.txt

# Development dependencies (training, testing, debugging)
pip install -r requirements-dev.txt
```

> **Note:** PyTorch CPU-only is installed automatically via `--extra-index-url` in `requirements.txt`. No GPU/CUDA required.

### Step 5: Verify Installation

```bash
python -c "import torch; import lightgbm; import transformers; print('All dependencies OK')"
```

### Step 6: Start the API Server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **First startup takes ~45 seconds** — the CLIP model (~600 MB) downloads from HuggingFace on first run and is cached in `~/.cache/huggingface/` for subsequent runs.

Wait for this log message before sending requests:
```
INFO:     Orchestrator initialized at startup
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 7: Start the Streamlit UI (New Terminal)

Open a **second terminal**, activate the same virtual environment, and run:

**Windows:**
```powershell
.venv\Scripts\Activate.ps1
streamlit run app/Home.py --server.port 8501
```

**macOS / Linux:**
```bash
source .venv/bin/activate
streamlit run app/Home.py --server.port 8501
```

### Step 8: Access the Application

| Service | URL |
|:--------|:----|
| **Streamlit UI** | http://localhost:8501 |
| **FastAPI Docs** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |

---

## Quick Test After Setup

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Expected:
```json
{"status": "ok", "model_version": "v1.0.0", "uptime_s": 120.5}
```

### 2. Sample Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "pump_017",
    "timestamp": "2026-02-17T10:30:00Z",
    "sensor_window": [
      {
        "sensor_00": 2.44, "sensor_01": 46.31, "sensor_02": 52.34,
        "sensor_03": 44.66, "sensor_04": 628.59, "sensor_05": 79.70
      }
    ]
  }'
```

Expected response (values may vary slightly):
```json
{
  "asset_id": "pump_017",
  "failure_probability": 0.0045,
  "fault_confidence": 0.7964,
  "top_signals": ["flow_rate_anomaly", "pressure_drop", "temperature_rise"],
  "explanation": "Minimal failure risk (0%) with high confidence (80%)...",
  "inference_ms": 2,
  "model_version": "v1.0.0"
}
```

### 3. Run Tests

```bash
pytest tests/ -v
```

Expected: 10 tests, all passing.

---

## Training Models (Optional)

Pre-trained model weights are included in `artifacts/`. You only need to train if you want to retrain from scratch.

### Train All Models

```bash
python scripts/train.py
```

### Train Individually

```bash
# Sensor-only LightGBM (~2 min)
python scripts/train.py --model baseline

# Joint sensor+image model + transformer fusion (~5 min)
python scripts/train.py --model multimodal
```

### CLI Inference (No Server Needed)

```bash
# Test with built-in samples
python scripts/infer.py --sample normal
python scripts/infer.py --sample at-risk
```

---

## Common Issues

| Problem | Solution |
|:--------|:---------|
| `ModuleNotFoundError: No module named 'src'` | Run all commands from the project root directory, not from `scripts/` or `src/` |
| CLIP model takes very long on first run | Normal — it downloads ~600 MB from HuggingFace. Cached after first download. |
| `Port 8000 already in use` | **Windows:** `Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess \| Stop-Process -Force` **Linux/Mac:** `lsof -ti:8000 \| xargs kill` |
| `torch not found` or CUDA errors | PyTorch CPU is installed via requirements.txt. If issues persist: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `sensor_baseline.pkl not found` | Run `python scripts/train_baseline.py` or ensure `artifacts/` has the pre-trained weights |
| Out of memory (OOM) | CLIP model needs ~730 MB RAM. Close other applications, or use a machine with ≥4 GB RAM |
| Docker: `port is already allocated` | Stop the conflicting container: `docker stop $(docker ps -q)` then retry |
| Docker: container exits immediately | Check logs: `docker logs pump-fault-risk` — usually a missing data file |
| `libgomp.so.1 not found` (Docker) | Already fixed in Dockerfile. If building custom image, add `libgomp1` to apt-get install |
| Streamlit can't connect to API | Make sure the API is running on port 8000 first. Streamlit expects API at `http://localhost:8000` |

---

## Project Structure (Key Files)

```
multimodal-pump-fault-prediction/
│
├── src/                           # Application source code
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings (ports, paths, model config)
│   ├── api/routes/                # API endpoints (health, prediction)
│   ├── models/                    # ML models (LightGBM, CLIP, Transformer)
│   └── services/                  # Orchestrator, preprocessing
│
├── app/                           # Streamlit multi-page UI
│   ├── Home.py                    # Entry point (streamlit run app/Home.py)
│   └── pages/                     # Sub-pages (prediction, evaluation, etc.)
│
├── scripts/                       # Training, evaluation, benchmarking
│   ├── train.py                   # Unified training entry point
│   ├── infer.py                   # CLI inference tool
│   └── load_test.py               # Load testing
│
├── data/                          # ⚠️ Not in Git — download from Google Drive
│   ├── baseline_model/            # 220K-row sensor data (optional)
│   └── multimodal_model/          # 241 sensor+image samples (required)
│
├── artifacts/                     # Pre-trained model weights
│   ├── sensor_baseline.pkl        # LightGBM sensor model
│   ├── joint_sensor_image.pkl     # LightGBM joint model
│   └── transformer_fusion_trained.pt  # Trained transformer weights
│
├── tests/                         # pytest test suite (10 tests)
├── docs/                          # Architecture, evaluation, demo script
│
├── Dockerfile                     # Container image definition
├── docker-compose.yml             # Single-container orchestration
├── start.sh                       # Runs both API + Streamlit in container
├── requirements.txt               # Production dependencies
└── requirements-dev.txt           # Dev dependencies (training, testing)
```

---

## Environment Variables (Optional)

These can be set in your shell or in a `.env` file:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `MODEL_VERSION` | `v1.0.0` | Version string returned in API responses |
| `DEBUG` | `false` | Enable debug logging |
| `API_URL` | `http://localhost:8000` | URL Streamlit uses to call the API |

---

## Next Steps After Setup

1. **Explore the UI** — Open http://localhost:8501 and try the Live Prediction page
2. **Read the demo script** — See `docs/DEMO_SCRIPT.md` for a guided walkthrough
3. **Review architecture** — See `docs/ARCHITECTURE.md` for system design
4. **Run tests** — `pytest tests/ -v` to verify everything works
5. **Check API docs** — Open http://localhost:8000/docs for interactive Swagger UI
