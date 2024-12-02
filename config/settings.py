"""Configuration settings for protein visualization and classification project."""

from pathlib import Path
import os

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = Path("/home/cain/alphafold_predicted_proteins/humanv4")
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR = ROOT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, LOG_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset settings
DATASET = {
    "train_split": 0.8,
    "validation_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42
}

# Visualization settings
VISUALIZATION = {
    "width": 1024,
    "height": 1024,
    "background_color": "white",
    "image_format": "tiff",
    "quality": 100,
    "styles": [
        "spheres",
        "surface",
        "mesh",
        "sticks",
        "ribbon",
        "cartoon",
        "wireframe",
        "spacefill"
    ],
    "orientations": {
        "front": {"rotation": [0, 0, 0]},
        "back": {"rotation": [0, 180, 0]},
        "top": {"rotation": [-90, 0, 0]},
        "bottom": {"rotation": [90, 0, 0]},
        "left": {"rotation": [0, 90, 0]},
        "right": {"rotation": [0, -90, 0]}
    }
}

# Model settings
MODEL = {
    "name": "vit_base_patch16_224",
    "pretrained": True,
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-6,
    "early_stopping_patience": 10,
}

# Style settings for different representation types
STYLE_SETTINGS = {
    "spheres": {"radius": 1.0},
    "surface": {"opacity": 0.8},
    "mesh": {"color": "grey"},
    "sticks": {},
    "ribbon": {"style": "ribbon"},
    "cartoon": {},
    "wireframe": {},
    "spacefill": {"radius": 1.6}
}

# Label mapping settings
LABEL_SOURCES = {
    "uniprot": "https://www.uniprot.org/uniprot",
    "pfam": "https://pfam.xfam.org",
    "go": "http://geneontology.org"
}

# Logging settings
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOG_DIR / "app.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}