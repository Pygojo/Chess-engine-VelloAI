```markdown
# NNUE Chess Engine & Streaming Training Pipeline

An end-to-end data processing, distillation, and training pipeline for a custom Efficiently Updatable Neural Network (NNUE) chess evaluation engine[cite: 8]. The system is built around memory-bounded streaming data pipelines that parse multi-gigabyte compressed PGN archives without disk decompression[cite: 9], aggregate games across multiple sources[cite: 4, 5, 6], distill evaluations from Stockfish via multi-process workers[cite: 1], and train a PyTorch NNUE architecture optimized for incremental accumulator updates in a TypeScript chess engine runtime[cite: 7, 8, 13].

---

## System Architecture & Data Flow


```

+-------------------------------------------------------------------------------+
|                             DATA INGESTION                                    |
|                                                                               |
|  +------------------------+  +----------------------+  +--------------------+ |
|  |   Lichess .pgn.zst     |  |   Chess.com API      |  | Supabase Postgres  | |
|  |  (download_lichess.py) |  | (download_chesscom.py)| |(export_own_games.py)| |
|  +-----------+------------+  +----------+-----------+  +---------+----------+ |
+--------------|--------------------------|------------------------|------------+
|                          |                        |
v                          |                        |
+------------------------------+          |                        |
| Stream Decompression Parser  |          |                        |
| (zstd stream -> parser.py)   |          |                        |
+--------------+---------------+          |                        |
|                          |                        |
+-------------------> + <--+------------------------+
|
v
+-------------------------------------------------------------------------------+
|                       POSITION SHARDING & ANNOTATION                          |
|                                                                               |
|                   data_shards/*.shardXXXXX.jsonl                              |
|           {"fen": "...", "result": 1.0, "ply": 24, "avg_elo": 1950}           |
|                                    |                                          |
|                                    v                                          |
|                  +-----------------------------------+                        |
|                  | (Optional) Stockfish Distillation |                        |
|                  |      (annotate.py - Multi-CPU)    |                        |
|                  +-----------------+-----------------+                        |
|                                    |                                          |
|                                    v                                          |
|                     data_shards_annotated/*.jsonl                             |
|          {"fen": "...", "result": 1.0, "eval": 145, ...}                      |
+------------------------------------|------------------------------------------+
|
v
+-------------------------------------------------------------------------------+
|                        STREAMING TRAINING PIPELINE                            |
|                                                                               |
|  +-------------------------------------------------------------------------+  |
|  | StreamingChessDataset (dataset.py)                                      |  |
|  |  - Feature Extraction: 768-dim one-hot vector (features.py)             |  |
|  |  - Train/Val Split: Deterministic FEN MD5 hashing (no full scan)        |  |
|  |  - Shuffling: In-memory rolling buffer (200k records ~= 600MB RAM)      |  |
|  |  - Target Blending: 0.8 * sigmoid(eval/400) + 0.2 * result              |  |
|  +-------------------------------------+-----------------------------------+  |
|                                        |                                      |
|                                        v                                      |
|  +-------------------------------------------------------------------------+  |
|  | NNUE Model (model.py / train.py)                                        |  |
|  |  - Input Layer: Linear(768, 256) + ReLU (Incremental Accumulator)       |  |
|  |  - Dense Trunk: Linear(256, 64) -> Linear(64, 32) -> Linear(32, 1)      |  |
|  |  - Optimizer: Adam + CosineAnnealingLR + Gradient Norm Clipping        |  |
|  |  - Loss: Mean Squared Error on Sigmoid Win Probabilities                |  |
|  |  - Checkpoints: Atomic step-level and epoch-level persistence           |  |
|  +-------------------------------------+-----------------------------------+  |
+----------------------------------------|--------------------------------------+
|
v
+-------------------------------------------------------------------------------+
|                         ENGINE WEIGHT SERIALIZATION                           |
|                                                                               |
|  export_weights() -> nnue_weights.json                                        |
|   * w1 transposed to (768, 256) for contiguous feature-major row slicing      |
|   * Dense trunk weights preserved in row-major layout                         |
|   * Ready for TypeScript evaluateNNUE / updateAcc incremental execution       |
+-------------------------------------------------------------------------------+

```

---

## File Structure

```bash
.
├── config.py                 # Central configurations, filtering thresholds, and path declarations[cite: 2]
├── features.py               # 768-dim board state encoding and game phase heuristics[cite: 7]
├── model.py                  # PyTorch NNUE architecture and JSON weight export serialization[cite: 8]
├── dataset.py                # Streaming IterableDataset with MD5 FEN partitioning & shuffle buffer[cite: 3]
├── pgn_utils.py              # PGN movetext extraction and score header parsing[cite: 10]
├── download_lichess.py       # Lichess monthly database dump downloader with HTTP Range resumption[cite: 5]
├── download_chesscom.py      # Chess.com public API player archive scraper[cite: 4]
├── export_own_games.py       # Supabase service role database client for user games[cite: 6]
├── parser.py                 # Memory-constant zstandard PGN stream parser to JSONL shards[cite: 9]
├── annotate.py               # Parallel Stockfish UCI engine position annotator[cite: 1]
├── train.py                  # Training pipeline with step/epoch atomic checkpointing[cite: 13]
├── run_pipeline.py           # Automated end-to-end CLI orchestrator[cite: 12]
├── requirements.txt          # Pinned Python package dependencies[cite: 11]
├── data_raw/                 # Storage for compressed .pgn.zst downloads[cite: 2]
├── data_shards/              # Tokenized 1M-record JSONL shards[cite: 2, 9]
├── data_shards_annotated/    # Shards annotated with Stockfish centipawn evaluations[cite: 1]
└── checkpoints/              # PyTorch training states and optimizer checkpoints[cite: 2, 13]

```

---

## Environment Setup

### 1. Python Virtual Environment

```bash
# Clone the repository
git clone [https://github.com/your-username/nnue-chess-engine.git](https://github.com/your-username/nnue-chess-engine.git)
cd nnue-chess-engine

# Initialize and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 2. Install Dependencies

Install all pinned packages via `requirements.txt`:

```bash
pip install -r requirements.txt

```

Required packages: `torch>=2.0`, `numpy>=1.24`, `requests>=2.31`, `chess>=1.10`, `zstandard>=0.22`, `supabase>=2.0`.

### 3. Install Stockfish Binary (Optional)

Required only if running position distillation via `annotate.py`:

* **Debian/Ubuntu:** `sudo apt-get install -y stockfish` (installs to `/usr/games/stockfish`)


* **macOS:** `brew install stockfish` (installs to `/opt/homebrew/bin/stockfish` or `/usr/local/bin/stockfish`)

---

## Execution Pipeline (Step-by-Step)

### Option A: Fully Automated Run

Run the end-to-end pipeline (download latest Lichess month $\to$ stream parse $\to$ train $\to$ export weights) via `run_pipeline.py`:

```bash
# Standard pipeline execution (Lichess only, 50 epochs)[cite: 12]
python run_pipeline.py

# Include production games exported from Supabase[cite: 6, 12]
export SUPABASE_URL="[https://your-project.supabase.co](https://your-project.supabase.co)"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
python run_pipeline.py --include-own-games[cite: 6, 12]

# Skip downloading if raw data is already present[cite: 12]
python run_pipeline.py --skip-download --epochs 30[cite: 12]

```

---

### Option B: Granular Manual Execution

#### Step 1: Ingest Game Data

Collect PGN datasets from one or more supported providers:

* **Source 1: Lichess Database Dumps**
Streams `.pgn.zst` archives with automatic resumption support:


```bash
# Resolve and download the latest available monthly dump[cite: 5]
python download_lichess.py --latest

# Or download a specific target archive[cite: 5]
python download_lichess.py 2026-07 --out data_raw/lichess_2026-07.pgn.zst

```


* **Source 2: Chess.com Public API**
Pulls full game archives for specified player accounts:


```bash
python download_chesscom.py MagnusCarlsen Hikaru --out data_shards/chesscom.jsonl --min-ply 4[cite: 4]

```


* **Source 3: Supabase Production Match Data**
Queries completed matches directly from the application's PostgreSQL instance:


```bash
export SUPABASE_URL="[https://your-project.supabase.co](https://your-project.supabase.co)"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
python export_own_games.py --out data_shards/own_games.jsonl --min-ply 4[cite: 6]

```



#### Step 2: Parse Compressed Streams into JSONL Shards

Stream `.pgn.zst` files into fixed 1,000,000-position `.jsonl` shards. The parser applies opening book filtering (`MIN_PLY=10`), length boundaries (`MAX_PLY=300`), game duration filtering (`MIN_GAME_PLIES=20`), and ELO filtering (`MIN_ELO=1600`) on the fly:

```bash
python parser.py data_raw/lichess_db_standard_rated_2026-07.pgn.zst \
    --out-dir data_shards \
    --positions-per-shard 1000000 \
    --min-elo 1600 \
    --sample-rate 1.0[cite: 2, 9]

```

#### Step 3: (Optional) Distill Teacher Evaluations with Stockfish

Enrich raw outcome shards with deep engine evaluations. Stockfish processes remain persistent across worker threads to avoid subprocess startup overhead:

```bash
python annotate.py data_shards/lichess_db_standard_rated_2026-07.shard00000.jsonl \
    --out data_shards_annotated \
    --stockfish-path /usr/games/stockfish \
    --depth 12 \
    --workers 4[cite: 1]

```

#### Step 4: Model Training

Train the NNUE architecture using `train.py`.

The training script reads shards dynamically using `StreamingChessDataset` with an in-memory shuffle buffer (default 200,000 positions $\approx 600\text{ MB}$ RAM), avoiding out-of-memory errors on large datasets. Train/validation splits are computed deterministically via MD5 hashes of position FEN strings.

```bash
python train.py \
    --data data_shards/*.jsonl \
    --output nnue_weights.json \
    --epochs 50 \
    --batch-size 1024 \
    --lr 1e-3 \
    --val-split 0.05 \
    --shuffle-buffer 200000 \
    --checkpoint checkpoints/latest.pt \
    --checkpoint-every-epochs 1 \
    --checkpoint-every-steps 5000 \
    --resume[cite: 13]

```

Note on Fault Tolerance: Checkpointing writes to `.tmp` files and swaps atomically to prevent corrupted saves if Google Colab or compute instances disconnect mid-training.

---

## Technical Specifications

### 1. Board Feature Encoding (768-dim)

Positions are converted into 768-dimensional sparse binary vectors via `features.fen_to_features()`:

$$\text{Index} = (\text{Color Offset} + \text{Piece Type}) \times 64 + \text{Square Index}$$

* **Color Offset:** White = $0$, Black = $6$

* **Piece Types:** $\text{P}=0, \text{N}=1, \text{B}=2, \text{R}=3, \text{Q}=4, \text{K}=5$

* **Square Index:** $0 \dots 63$ corresponding to $a8 \dots h1$


### 2. Neural Architecture

Defined in `model.py`:

* **Accumulator Layer (`input_layer`):** `nn.Linear(768, 256)` with unclipped ReLU. Evaluated incrementally in search: when a piece moves, only the activated weight rows corresponding to added/removed features are added/subtracted.


* **Trunk Layer 1 (`trunk1`):** `nn.Linear(256, 64)` + ReLU.


* **Trunk Layer 2 (`trunk2`):** `nn.Linear(64, 32)` + ReLU.


* **Output Layer (`output_layer`):** `nn.Linear(32, 1)` yielding centipawn evaluation score from White's perspective.



### 3. Loss Formulation & Target Blending

During training, output evaluations are mapped to a winning probability via a standard sigmoid scaling factor:

$$P_{\text{win}} = \sigma\left(\frac{\text{eval}}{400.0}\right) = \frac{1}{1 + e^{-\text{eval}/400.0}}$$

The model optimizes Mean Squared Error ($MSE$) against target $y$:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (P_{\text{win}, i} - y_i)^2$$

* **When Stockfish `eval` exists:** $y = 0.8 \cdot \sigma\left(\frac{\text{eval}_{\text{Stockfish}}}{400.0}\right) + 0.2 \cdot \text{Result}$

* **When `eval` is missing (raw game outcomes):** $y = \text{Result} \in \{1.0, 0.5, 0.0\}$


### 4. Weight Export Schema (`nnue_weights.json`)

`export_weights()` serializes model parameters into a JSON schema structured for direct consumption by the TypeScript engine:

```json
{
  "w1": [ ... ],         // (768 * 256) Transposed accumulator weights, flattened feature-major[cite: 8]
  "b1": [ ... ],         // (256) Accumulator bias vector[cite: 8]
  "w_trunk1": [ ... ],   // (64 * 256) Row-major dense matrix[cite: 8]
  "b_trunk1": [ ... ],   // (64) Dense bias vector[cite: 8]
  "w_trunk2": [ ... ],   // (32 * 64) Row-major dense matrix[cite: 8]
  "b_trunk2": [ ... ],   // (32) Dense bias vector[cite: 8]
  "w_out": [ ... ],      // (1 * 32) Output weight vector[cite: 8]
  "b_out": 0.042,        // Scalar output bias[cite: 8]
  "acc_size": 256,       // Dimension validation on engine load[cite: 8]
  "trunk1_size": 64,     // Dimension validation on engine load[cite: 8]
  "trunk2_size": 32      // Dimension validation on engine load[cite: 8]
}

```

```

```
