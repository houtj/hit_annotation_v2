# Active Learning Image Annotation Tool

A flexible active learning platform supporting both **image segmentation** and **image classification** tasks. The application uses DINOv3 features with task-specific heads trained via human-in-the-loop annotation.

## Supported Tasks

### 1. Segmentation (Dip Picking)
- Point-based annotation for foreground/background labeling
- Binary segmentation with prediction overlay
- Point extraction from model predictions
- Red-to-green confidence heatmap visualization

### 2. Classification (Multiclass)
- Single-click class assignment per image
- Confidence bars and probability distributions
- Support for any number of custom classes
- Real-time prediction feedback

## Quick Start

### Initialize Session (First Time)

The application uses **config files** for initialization. Choose the appropriate config for your task:

**For Segmentation (Dip Picking):**
```bash
cd initialization
PYTHONPATH=.. uv run python init_session.py --config dip_picking_config.yaml
```

**For Classification:**
```bash
cd initialization
PYTHONPATH=.. uv run python init_session.py --config classification_config.yaml
```

**Override settings (e.g., limit files for testing):**
```bash
cd initialization
PYTHONPATH=.. uv run python init_session.py --config classification_config.yaml --max-files 50
```

See `initialization/dip_picking_config.yaml` and `initialization/classification_config.yaml` for configuration examples.

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
PYTHONPATH=.. uv run python train.py
```
The ML service runs continuously, polling the database for training triggers. When you click "Save" in the labeling interface, it automatically starts training with the appropriate task-specific model head.

## Project Structure
```
active_annotation/
├── db/                     # Shared database infrastructure
│   ├── models.py           # SQLAlchemy models (supports both tasks)
│   └── database.py         # Database connection
├── models/                 # Shared ML models
│   ├── dinov3/             # DINOv3 feature extractor
│   └── weights/            # Pre-trained weights
├── initialization/         # Setup tooling
│   ├── init_session.py     # Session initialization script
│   ├── dip_picking_config.yaml      # Segmentation task config
│   └── classification_config.yaml   # Classification task config
├── frontend/               # TypeScript + Vite frontend
│   └── src/
│       └── pages/labeling/
│           ├── LabelingView.ts           # Main orchestrator
│           └── components/
│               ├── SegmentationUI.ts     # Segmentation-specific UI
│               └── ClassificationUI.ts   # Classification-specific UI
├── backend/                # FastAPI server
│   ├── main.py             # Config caching, /api/config endpoint
│   ├── routes/
│   │   ├── labels.py       # Task-aware label/prediction endpoints
│   │   ├── files.py        # File management
│   │   └── task_handlers/  # Task-specific logic
│   │       ├── segmentation.py
│   │       └── classification.py
│   └── utils/              # Helper functions
├── ml_service/             # ML training service (task-agnostic)
│   ├── models/             # Task-specific model heads
│   │   ├── base_head.py    # Abstract base class
│   │   ├── segmentation_head.py
│   │   └── classification_head.py
│   ├── data_loaders/       # Task-specific data loaders
│   │   ├── base.py
│   │   ├── segmentation.py  # Point-based batching
│   │   └── classification.py # Image-based batching
│   ├── losses/             # Task-specific loss functions
│   │   ├── segmentation.py  # BCE with logits
│   │   └── classification.py # CrossEntropy
│   ├── task_factory.py     # Factory pattern for task components
│   ├── train.py            # Main training script
│   ├── training_loop.py    # Training with suspension
│   └── inference.py        # Task-aware prediction generation
└── session/                # Generated session data (gitignored)
    ├── annotations.db      # SQLite database
    ├── storage/            # Images, features, and predictions
    └── checkpoints/        # Task-specific model checkpoints
```

## How It Works

### Workflow (Both Tasks)
1. **Initialization**: Run `init_session.py` with a config file to:
   - Create database with task type and class definitions
   - Convert images to `.npy` format
   - Extract DINOv3 features (384-dim, 96×96 spatial resolution)
   
2. **Annotation**: Use the web interface to label images:
   - **Segmentation**: Click to add point labels (foreground/background)
   - **Classification**: Click to assign a class to the entire image
   
3. **Training**: Click "Save" to trigger ML training:
   - The ML service detects the task type and instantiates the appropriate:
     - Model head (segmentation or classification)
     - Data loader (point-based or image-based batching)
     - Loss function (BCE or CrossEntropy)
   - Training uses early stopping with smart suspension for real-time annotation
   
4. **Prediction**: During training, the model periodically generates predictions:
   - **Segmentation**: Probability heatmap saved as PNG (red=background, green=foreground)
   - **Classification**: Class probabilities stored as JSON
   
5. **Iteration**: 
   - **Segmentation**: View prediction overlay, extract points, refine labels, retrain
   - **Classification**: View confidence bars, correct mislabeled images, retrain

### Architecture Patterns

- **ML Service**: Factory pattern for task-agnostic training loop
- **Backend**: Conditional dispatch to task-specific handlers
- **Frontend**: Component composition with task-based UI selection
- **Config**: Cached at startup (app.state) to avoid repeated DB queries

## Creating a Config File

Configuration files define the task type, data paths, classes, and training parameters. Key fields:

```yaml
# Global task type (required)
task: "segmentation"  # or "classification"

# Data paths (required)
image_dir: "/path/to/images"
formats: "jpg,jpeg,png"
labels_dir: null  # Optional: path to pre-existing labels

# Session paths (required)
session_dir: "../session"
dinov3_repo: "../models/dinov3"
weights_path: "../models/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# Feature extraction (required)
resize: 1536  # Max image dimension for DINOv3

# Class definitions (required)
classes:
  - name: "foreground"
    color: "#00FF00"
  - name: "background"
    color: "#FF0000"

# ML training parameters (required)
prediction_interval: 20      # Generate prediction every N epochs
early_stop_patience: 5       # Stop if no improvement for N epochs
early_stop_threshold: 0.001  # Minimum improvement threshold

# Segmentation-specific (optional, only for segmentation task)
max_points: 500              # Max points to extract from predictions
confidence_threshold: 0.15   # Min confidence for point extraction
min_distance: 3.0            # Min pixel distance between points
gradient_weight: 2.0         # Weight for gradient-based point selection
```

**Key Differences:**
- **Segmentation**: Requires point extraction parameters; uses 2 classes (foreground/background)
- **Classification**: No point extraction; supports any number of classes

See `initialization/dip_picking_config.yaml` and `initialization/classification_config.yaml` for complete examples.

## Notes & Tips

### PYTHONPATH Requirements
The `PYTHONPATH=..` prefix is required when running scripts from subdirectories to ensure Python can import shared modules (`db`, `models`, etc.) from the project root.

### Session Management
- Each initialization creates/overwrites the `session/` directory
- To switch tasks, re-run `init_session.py` with a different config file
- Previous checkpoints and annotations will be lost when re-initializing

### Task Type Detection
The frontend and backend automatically detect the task type from the database at startup. No code changes are needed when switching between tasks.

### Training Behavior
- Training suspends at `prediction_interval` epochs to generate predictions for the current file
- The model resumes training after prediction completes
- Early stopping triggers when test loss stops improving for `early_stop_patience` epochs
- Click "Stop Training" to manually halt training

### Segmentation-Specific
- **Point Mode**: Toggle on/off to enable/disable point annotation
- **Extract Points**: Converts prediction heatmap to point labels (removes old auto-extracted points)
- Human-created points are preserved when extracting

### Classification-Specific
- One label per image (clicking a class overwrites the previous selection)
- Predictions show confidence bars for all classes, sorted by probability
- The top prediction is highlighted with a colored badge

### Performance
- DINOv3 feature extraction runs once during initialization (cached as `.npy` files)
- Training operates on pre-extracted features (384-dim × 96×96), making it very fast
- Image-based batching (classification) is faster than point-based batching (segmentation)

