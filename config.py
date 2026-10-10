"""Project-wide defaults for the full RADAR model."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATASET = "QaTa-COV19"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4
DEFAULT_SEED = 42

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_BIOMEDCLIP_ROOT = Path("pretrained/BiomedCLIP")
DEFAULT_MEDSAM2_CHECKPOINT = Path("pretrained/MedSAM2_latest.pt")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MEDSAM2_CONFIG = "configs/sam2.1_hiera_t512.yaml"

LEARNING_RATE = 3e-4
EPOCHS = 500
EARLY_STOPPING_PATIENCE = 50

SUPPORTED_DATASETS = ("QaTa-COV19", "MosMedData+", "Kvasir")

