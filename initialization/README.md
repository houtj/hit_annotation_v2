# Initialization Tooling

This folder contains tooling for initializing new annotation sessions.

## Setup

This folder has its own virtual environment managed by `uv`:

```bash
cd initialization
uv sync
```

## Usage

Initialize a new annotation session with images:

```bash
cd initialization
PYTHONPATH=/Users/houtj/projects/active_annotation uv run python init_session.py \
  --image-dir /path/to/images \
  --formats png,jpg,jpeg
```

### With existing labels:

```bash
cd initialization
PYTHONPATH=/Users/houtj/projects/active_annotation uv run python init_session.py \
  --image-dir /path/to/images \
  --formats png \
  --labels-dir /path/to/labels
```

### With file limit (for testing):

```bash
cd initialization
PYTHONPATH=/Users/houtj/projects/active_annotation uv run python init_session.py \
  --image-dir /path/to/images \
  --formats png \
  --max-files 10
```

## What it does

1. Creates session directory structure (`../session/`)
2. Loads DINOv3 model from `../models/dinov3/`
3. Processes images:
   - Converts images to `.npy` format
   - Extracts DINOv3 features for each image
   - Stores features in `session/storage/features/`
4. Populates database (`session/annotations.db`) with:
   - File records
   - Default classes (foreground, background)
   - Configuration parameters
   - Optional: imported labels if `--labels-dir` is provided

## Dependencies

- PyTorch (with MPS support for Apple Silicon)
- DINOv3 (from `../models/dinov3/`)
- SQLAlchemy (async)
- Pillow, NumPy

## Note

This is a one-time setup tool. After initialization, use:
- `backend/` for the API service
- `ml_service/` for training
- `frontend/` for the UI
