# graphcast_lite/config.py
import os
import torch
import numpy as np

# =========================
# PATHS
# =========================
DATA_DIR = r"C:\Users\user\Desktop\Graphcast_Project\ERA5_np_float32_1986_2026"
STATS_DIR = r"C:\Users\user\Desktop\Graphcast_Project\ERA5_np_stats"
CHECKPOINT_DIR = r"C:\Users\user\Desktop\Graphcast_Project\checkpoints_graphcast_lite"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================
# GRID
# =========================
LAT = np.linspace(46.0, 20.0, 64).astype(np.float32)
LON = np.linspace(-15.0, 6.0, 64).astype(np.float32)

HEIGHT = 64
WIDTH = 64

# =========================
# DATASET
# =========================
INPUT_STEPS = 2
TARGET_LEAD_TIMES = "6h"

INPUT_CHANNELS = 189
FORCING_CHANNELS = 7
OUTPUT_CHANNELS = 189

BATCH_SIZE = 1
NUM_WORKERS = 0
PIN_MEMORY = True

# =========================
# MODEL LITE
# =========================
MESH_SIZE = 2
LATENT_SIZE = 64
GNN_MSG_STEPS = 4
HIDDEN_LAYERS = 1
RADIUS_QUERY_FRACTION_EDGE_LENGTH = 0.6

# =========================
# TRAINING
# =========================
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0

USE_AMP = True
SAVE_EVERY = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"