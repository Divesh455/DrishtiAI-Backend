from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TRAIN_IMAGE_DIR = RAW_DATA_DIR / "train_images"

WEIGHTS_DIR = PROJECT_ROOT / "weights"


# ============================================================
# MODEL PATHS
# ============================================================

# Existing production/default DR model
MODEL_PATH = WEIGHTS_DIR / "dr_model.pth"

# New focal-loss experiment model.
# This does NOT overwrite the existing dr_model.pth.
MODEL_SAVE_PATH = WEIGHTS_DIR / "dr_model_focal.pth"

# Image quality / fundus suitability model
QUALITY_MODEL_PATH = WEIGHTS_DIR / "quality_model.pth"

# Lesion detection model
LESION_MODEL_PATH = WEIGHTS_DIR / "lesion_model.pth"


# ============================================================
# DATASET
# ============================================================

NUM_CLASSES = 5

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


# ============================================================
# IMAGE
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# TRAINING
# ============================================================

NUM_EPOCHS = 20

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2

RANDOM_SEED = 42


# ============================================================
# FOCAL LOSS
# ============================================================

FOCAL_GAMMA = 2.0


# ============================================================
# LABEL SMOOTHING
# ============================================================

LABEL_SMOOTHING = 0.0


# ============================================================
# SCHEDULER
# ============================================================

SCHEDULER_FACTOR = 0.5

SCHEDULER_PATIENCE = 2

MIN_LR = 1e-7


# ============================================================
# EARLY STOPPING
# ============================================================

EARLY_STOPPING_PATIENCE = 5


# ============================================================
# GRADIENT CLIPPING
# ============================================================

GRADIENT_CLIP_NORM = 1.0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATA SPLIT
# ============================================================

VALIDATION_SIZE = 0.20

SPLIT_RANDOM_STATE = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def print_config():
    print("=" * 60)
    print("DrishtiAI Training Configuration")
    print("=" * 60)

    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Training CSV        : {TRAIN_CSV}")
    print(f"Training images     : {TRAIN_IMAGE_DIR}")

    print()
    print("Model paths")
    print("-" * 60)
    print(f"DR model            : {MODEL_PATH}")
    print(f"Focal DR model      : {MODEL_SAVE_PATH}")
    print(f"Quality model       : {QUALITY_MODEL_PATH}")
    print(f"Lesion model        : {LESION_MODEL_PATH}")

    print()
    print("Dataset")
    print("-" * 60)
    print(f"Number of classes   : {NUM_CLASSES}")
    print(f"Class names         : {CLASS_NAMES}")

    print()
    print("Image")
    print("-" * 60)
    print(f"Image size          : {IMAGE_SIZE}")

    print()
    print("Training")
    print("-" * 60)
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Epochs              : {NUM_EPOCHS}")
    print(f"Learning rate       : {LEARNING_RATE}")
    print(f"Weight decay        : {WEIGHT_DECAY}")
    print(f"Workers             : {NUM_WORKERS}")

    print()
    print("Focal Loss")
    print("-" * 60)
    print(f"Focal gamma         : {FOCAL_GAMMA}")

    print()
    print("Scheduler")
    print("-" * 60)
    print(f"Factor              : {SCHEDULER_FACTOR}")
    print(f"Patience            : {SCHEDULER_PATIENCE}")
    print(f"Minimum LR          : {MIN_LR}")

    print()
    print("Early Stopping")
    print("-" * 60)
    print(f"Patience            : {EARLY_STOPPING_PATIENCE}")

    print()
    print("Device")
    print("-" * 60)
    print(f"Device              : {DEVICE}")

    print()
    print("Data Split")
    print("-" * 60)
    print(f"Validation size     : {VALIDATION_SIZE}")
    print(f"Random state        : {SPLIT_RANDOM_STATE}")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print_config()