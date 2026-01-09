# Initialization Tooling

This folder contains tooling for initializing new annotation sessions.

## Setup

This folder has its own virtual environment managed by `uv`:

```bash
cd initialization
uv sync
```

## Usage

All initialization parameters are managed through a YAML configuration file. This makes it easier to manage multiple parameters and ensures reproducibility.

### 1. Copy and edit the configuration template:

```bash
cd initialization
cp dip_picking_config.yaml my_project_config.yaml
# Edit my_project_config.yaml with your settings
```

### 2. Run initialization with config file:

```bash
cd initialization
PYTHONPATH=/Users/houtj/projects/active_annotation uv run python init_session.py \
  --config my_project_config.yaml
```

### 3. Override max_files for testing (optional):

```bash
cd initialization
PYTHONPATH=/Users/houtj/projects/active_annotation uv run python init_session.py \
  --config my_project_config.yaml \
  --max-files 10
```

## Configuration File

The `dip_picking_config.yaml` file contains all initialization parameters with detailed documentation:

- **Data Paths**: Image directory, formats, labels directory
- **Feature Extraction**: DINOv3 resize parameter
- **ML Training**: Prediction interval, early stopping parameters
- **Point Extraction**: Max points, confidence threshold, spatial distribution
- **Classes**: Binary segmentation class definitions with colors

See `dip_picking_config.yaml` for detailed documentation on each parameter.

## What it does

1. Creates session directory structure (`../session/`)
2. Loads DINOv3 model from `../models/dinov3/`
3. Processes images:
   - Converts images to `.npy` format
   - Extracts DINOv3 features for each image
   - Stores features in `session/storage/features/`
4. Populates database (`session/annotations.db`) with:
   - File records
   - Classes (from config or defaults: foreground, background)
   - Configuration parameters (from config or defaults)
   - Optional: imported labels if labels directory is provided

## Configuration Parameters

All configuration parameters from the YAML file are stored in the database's `Config` table and can be accessed by the backend and ML service during annotation and training.

## Dependencies

- PyTorch (with MPS support for Apple Silicon)
- DINOv3 (from `../models/dinov3/`)
- SQLAlchemy (async)
- Pillow, NumPy
- PyYAML (for config file parsing)

## Note

This is a one-time setup tool. After initialization, use:
- `backend/` for the API service
- `ml_service/` for training
- `frontend/` for the UI
