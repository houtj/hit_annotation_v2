# Active Learning Image Annotation Tool

A flexible active learning annotation tool supporting both **segmentation** and **classification** tasks.

## Supported Tasks

### 1. Segmentation (Dip Picking)
- Point-based annotation for binary segmentation
- Automatic point extraction from predictions
- Suitable for geological dip picking, line detection, etc.

### 2. Classification
- Image-level class labels
- Multi-class support (e.g., animal, car, human, other)
- Simple click-to-classify interface

## Quick Start

### Initialize Session (First Time)

**Segmentation Task (Dip Picking):**
```bash
cd initialization
uv run python init_session.py --config dip_picking_config.yaml
```

**Classification Task:**
```bash
cd initialization
uv run python init_session.py --config classification_config.yaml
```

**Custom configuration:**
Edit the YAML config files to specify:
- Task type (`task: "segmentation"` or `task: "classification"`)
- Image directory and formats
- Classes and colors
- Training parameters

### Frontend (Development)
```bash
cd frontend
npm run dev
```
Access at: http://localhost:5173

### Backend (FastAPI)
```bash
cd backend
PYTHONPATH=.. uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
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
├── db/                 # Shared database infrastructure
│   ├── models.py       # SQLAlchemy models
│   └── database.py     # Database connection
├── models/             # Shared ML models
│   ├── dinov3/         # DINOv3 feature extractor
│   └── weights/        # Pre-trained weights
├── initialization/     # Setup tooling
│   └── init_session.py # Session initialization script
├── frontend/           # TypeScript + Vite frontend
├── backend/            # FastAPI server
│   ├── routes/         # API endpoints
│   └── utils/          # Helper functions
├── ml_service/         # ML training service
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

### Segmentation Workflow
1. **Initialization**: Run `init_session.py` with segmentation config
2. **Annotation**: Use the web interface to annotate images with point labels (foreground/background)
3. **Training**: Click "Save" to trigger ML training on point-based labels
4. **Prediction**: Model generates segmentation masks periodically during training
5. **Point Extraction**: Extract points from predictions for active learning
6. **Iteration**: Refine labels and retrain

### Classification Workflow
1. **Initialization**: Run `init_session.py` with classification config
2. **Annotation**: Select a class for each image using the class selector
3. **Training**: Click "Save" to trigger ML training on image-level labels
4. **Prediction**: Model predicts class and confidence for images
5. **Iteration**: Review predictions, correct labels, and retrain

## Configuration Files

- `initialization/dip_picking_config.yaml` - Segmentation task configuration
- `initialization/classification_config.yaml` - Classification task configuration

Key parameters:
- `task`: Task type ("segmentation" or "classification")
- `image_dir`: Path to images
- `classes`: List of classes with names and colors
- `prediction_interval`, `early_stop_patience`: Training parameters

