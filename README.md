```markdown
# NNUE Chess Engine & End-to-End Training Pipeline

A custom neural network chess evaluation engine powered by an Efficiently Updatable Neural Network (NNUE) architecture[cite: 8]. The repository contains a complete pipeline that streams and processes compressed Lichess archives[cite: 5, 9], downloads Chess.com player games[cite: 4], exports user gameplay from Supabase[cite: 6], annotates positions via parallel Stockfish instances[cite: 1], and exports weight matrices structured for incremental accumulator evaluation in TypeScript[cite: 7, 8].

---

## Architecture Overview


```

[ Lichess .pgn.zst / Chess.com API / Supabase DB ]
│
▼
[ Streaming PGN Parsing (parser.py) ]
│
▼
[ JSONL Shards (data_shards/*.jsonl) ]
│
├──────────────────────────┐
▼                          ▼
[ Stockfish Teacher Annotation ]    [ Raw Game Outcome ]
(Parallel UCI / depth 12)          (Result Header)
│                          │
└─────────────┬────────────┘
▼
[ StreamingChessDataset ]
- 768-dim Bitboard Features
- Shuffle Buffer (200k)
- Deterministic FEN Split
│
▼
[ NNUE Architecture ]
- Accumulator: 768 -> 256
- Dense Trunk: 256 -> 64 -> 32 -> 1
│
▼
[ Weight Export (.json format) ]
- Transposed w1 for Fast Acc
- Row-Major Dense Trunks
│
▼
[ TypeScript Engine Evaluation ]

```

---

## Key Features

* **Zero-Disk Decompression Streaming:** Reads directly from `.pgn.zst` files using `zstandard` stream readers and `python-chess` generators, keeping RAM usage flat regardless of archive size[cite: 9].
* **Multi-Source Data Ingestion:**
  * **Lichess Archives:** Automated monthly dump download with HTTP Range resumption[cite: 5].
  * **Chess.com Public API:** Ingests monthly game archives by player username[cite: 4].
  * **Supabase Production Data:** Streams finished games from the `chess_games` database table[cite: 6].
* **Stockfish Teacher Distillation:** Annotates position shards across multi-core CPU workers with persistent UCI Stockfish engine instances[cite: 1].
* **Lazy Streaming Dataset:** `StreamingChessDataset` avoids loading multi-gigabyte shards into memory by utilizing a rolling shuffle buffer and deterministic MD5 FEN-hashing for train/validation splits[cite: 3].
* **NNUE Accumulator & Trunk Design:** Implements a 768-dimension feature input[cite: 2, 7, 8], a 256-width accumulator layer (transposed for incremental addition/subtraction in search)[cite: 8], and a deep feedforward trunk ($256 \to 64 \to 32 \to 1$)[cite: 8].

---

## Tech Stack

* **Language & Runtime:** Python 3.10+[cite: 1, 4, 5, 6, 9]
* **Machine Learning & Math:** PyTorch, NumPy[cite: 3, 7, 8]
* **Chess Parsing & Analysis:** `python-chess`, Stockfish UCI engine[cite: 1, 9]
* **Data Streams & Storage:** `zstandard` (zstd), Supabase Client (`supabase-py`), Requests[cite: 4, 5, 6, 9]

---

## Project Structure

```bash
├── config.py                 # Central pipeline paths, filter thresholds, and constants[cite: 2]
├── features.py               # 768-dim board feature mapping and phase calculation[cite: 7]
├── model.py                  # NNUE PyTorch architecture and JSON weight exporter[cite: 8]
├── dataset.py                # Streaming iterable dataset with shuffle buffer[cite: 3]
├── pgn_utils.py              # PGN text parser and result header mapping[cite: 10]
├── download_lichess.py       # Lichess .pgn.zst stream downloader with resume support[cite: 5]
├── download_chesscom.py      # Chess.com public API player game downloader[cite: 4]
├── export_own_games.py       # Supabase chess_games database exporter[cite: 6]
├── parser.py                 # Direct .pgn.zst stream parser to JSONL position shards[cite: 9]
├── annotate.py               # Multi-process Stockfish position evaluator[cite: 1]
├── data_raw/                 # Raw downloaded compressed archives (.pgn.zst)[cite: 2]
├── data_shards/              # Tokenized JSONL position shards (1M positions/shard)[cite: 2, 9]
├── data_shards_annotated/    # Stockfish-annotated position shards[cite: 1]
└── checkpoints/              # Model weights and exported JSON parameters[cite: 2, 8]

```

---

## Setup & Installation

### 1. System Requirements

* **Python:** 3.10 or higher
* **Stockfish Binary:**
* **Ubuntu/Debian:** `sudo apt-get install -y stockfish`

* **macOS:** `brew install stockfish`



### 2. Python Dependencies

```bash
pip install torch numpy python-chess zstandard requests supabase

```

---

## Execution Workflow

### Step 1: Download Game Data

**From Lichess:**

```bash
# Auto-resolve and download the most recent complete month[cite: 5]
python download_lichess.py --latest

# Or specify a specific month[cite: 5]
python download_lichess.py 2026-07

```

**From Chess.com:**

```bash
python download_chesscom.py hikaru MagnusCarlsen --out data_shards/chesscom.jsonl[cite: 4]

```

**From Supabase Database:**

```bash
export SUPABASE_URL="[https://your-project.supabase.co](https://your-project.supabase.co)"[cite: 6]
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"[cite: 6]
python export_own_games.py --out data_shards/own_games.jsonl[cite: 6]

```

---

### Step 2: Stream & Parse into JSONL Shards

Stream compressed Lichess archives into 1,000,000-position `.jsonl` shards:

```bash
python parser.py data_raw/lichess_db_standard_rated_2026-07.pgn.zst --min-elo 1600 --sample-rate 1.0[cite: 2, 9]

```

---

### Step 3: Annotate Positions with Stockfish

Run multi-process evaluation using Stockfish to generate centipawn/mate score targets:

```bash
python annotate.py data_shards/lichess*.shard00000.jsonl \
    --out data_shards_annotated \
    --stockfish-path /usr/games/stockfish \
    --depth 12 \
    --workers 4[cite: 1]

```

---

### Step 4: Model Training

Load shards dynamically using `StreamingChessDataset`. If positions contain Stockfish evaluations, targets are softly blended ($0.8 \times P_{\text{eval}} + 0.2 \times \text{Result}$); otherwise, training falls back directly to game outcome ($1.0, 0.5, 0.0$).

```python
import glob
from torch.utils.data import DataLoader
from dataset import StreamingChessDataset
from model import NNUE

# Load annotated and raw shards seamlessly[cite: 1, 3]
shard_files = glob.glob("data_shards_annotated/*.jsonl") or glob.glob("data_shards/*.jsonl")

train_data = StreamingChessDataset(shard_files, val_split=0.05, want_val=False, shuffle_buffer_size=200_000)[cite: 3]
train_loader = DataLoader(train_data, batch_size=4096)

model = NNUE(num_features=768, acc_size=256, trunk1_size=64, trunk2_size=32)[cite: 8]
# Execute standard PyTorch training loop

```

---

### Step 5: Export Weights for TypeScript Runtime

Export the trained model to JSON with accumulator weights transposed (`w1`) for incremental feature additions/subtractions:

```python
from model import export_weights

export_weights(model, "checkpoints/nnue_weights.json")[cite: 8]

```

---

## Model Architecture Details

* **Feature Vector (768):** Encodes piece placement across 64 squares for 6 piece types and 2 colors via `(color_offset + piece_type) * 64 + square`.


* **Accumulator Layer:** Single `nn.Linear(768, 256)` with ReLU activation. Allows the TypeScript search engine to update board scores incrementally per move rather than recomputing the full board.


* **Dense Evaluation Trunk:**
* Layer 1: `Linear(256, 64)` + ReLU


* Layer 2: `Linear(64, 32)` + ReLU


* Output: `Linear(32, 1)` (Centipawn perspective score)





```

```
