# Active Learning Image Annotation Tool

## Quick Start

### Initialize Session (First Time)

**Basic usage (images only):**
```bash
cd backend
uv run python init_session.py --image-dir /path/to/images --formats jpg,png
```

**With labels import:**
```bash
cd backend
uv run python init_session.py \
  --image-dir ../data/hitl_data/condabri_north_349/images \
  --formats png \
  --labels-dir ../data/hitl_data/condabri_north_349_point_labels
```

**With file limit (useful for testing with large datasets):**
```bash
cd backend
uv run python init_session.py \
  --image-dir ../data/hitl_data/condabri_north_349/images \
  --formats png \
  --labels-dir ../data/hitl_data/condabri_north_349_point_labels \
  --max-files 50
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
```
The ML service runs continuously, polling the database for training triggers. When you click "Save" in the labeling interface, it automatically starts training.

## Project Structure
```
active_annotation/
├── frontend/           # TypeScript + Vite frontend
├── backend/            # FastAPI server
│   ├── routes/         # API endpoints
│   ├── db/             # Database models
│   └── utils/          # Helper functions
├── ml_service/         # DINOv3 ML training service
│   ├── model.py        # Binary segmentation head
│   ├── data_loader.py  # Data loading utilities
│   ├── training_loop.py # Training with suspension
│   ├── inference.py    # Prediction generation
│   └── train.py        # Main training script
└── session/            # Generated session data (gitignored)
    ├── annotations.db  # SQLite database
    ├── storage/        # Images and features
    └── checkpoints/    # Model checkpoints
```

## How It Works

1. **Initialization**: Run `init_session.py` to create database, convert images to `.npy`, and extract DINOv3 features
2. **Annotation**: Use the web interface to annotate images with point labels (foreground/background)
3. **Training**: Click "Save" to trigger ML training. The service trains a binary segmentation head on DINOv3 features
4. **Prediction**: During training, the model periodically generates predictions on the current image
5. **Iteration**: View predictions overlaid on images, refine labels, and retrain

