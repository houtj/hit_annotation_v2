# Active Learning Image Annotation Tool

## Quick Start

### Initialize Session (First Time)
```bash
cd backend
uv run python init_session.py --image-dir /path/to/images --formats jpg,png
uv run python init_session.py --image-dir /Users/houtj/projects/active_annotation/data/img_samples --format jpg
```

### Frontend (Development)
```bash
cd frontend
npm run dev
```
Access at: http://localhost:5173

### Backend (FastAPI)
```bash
cd backend
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Access at: http://localhost:8000

### ML Service (Training/Inference)
```bash
cd ml_service
uv run python train.py
# or
uv run python inference.py
```

## Project Structure
```
active_annotation/
├── frontend/        # TypeScript + Vite
├── backend/         # FastAPI server
├── ml_service/      # DINOv2 ML worker
├── data/            # Database & images
└── models/          # Saved model checkpoints
```

