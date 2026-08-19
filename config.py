"""
Shared configuration for the Lichess data pipeline.

Model hyperparameters live in model.py, not here -- this file is data
pipeline only (download/parse/filter settings).
"""

import os

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_raw")
SHARD_DIR = os.path.join(PROJECT_ROOT, "data_shards")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SHARD_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Lichess database -----------------------------------------------------
LICHESS_BASE_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{year_month}.pgn.zst"

# --- Position filtering -----------------------------------------------------
MIN_PLY = 10          # skip the first N half-moves (opening theory / book moves)
MAX_PLY = 300         # sanity cap, skip absurdly long games past this
MIN_GAME_PLIES = 20   # skip aborted/very short games entirely
MIN_ELO = 1600        # skip games below this average rating; None to disable
POSITION_SAMPLE_RATE = 1.0  # 1.0 = every eligible position, 0.1 = ~1 in 10

# --- Feature encoding ---------------------------------------------------------
NUM_FEATURES = 768  # 6 piece types x 2 colors x 64 squares -- see features.py

# --- Output shard settings -----------------------------------------------------
POSITIONS_PER_SHARD = 1_000_000
