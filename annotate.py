"""
Annotates position shards with real Stockfish evaluations.

Uses actual Stockfish, not our own engine -- annotating with our own
engine's evals would just reinforce whatever mistakes it currently makes,
no new signal gained. That's the whole point of this step (it's the
"Phase 3/4" from your roadmap doc: train on a stronger teacher's
evaluations instead of game outcome alone).

Requires a Stockfish binary:
    Colab/Ubuntu:  !apt-get install -y stockfish
                   (binary ends up at /usr/games/stockfish)
    Elsewhere:     https://stockfishchess.org/download/

Runs multiple Stockfish processes in parallel (--workers), each one kept
alive across many positions rather than respawned per position -- process
startup cost would otherwise dominate runtime.

READ THIS BEFORE RUNNING AT SCALE:
Stockfish analysis is slow per position compared to everything else in
this pipeline. Realistic throughput per worker at depth 12 is roughly
5-20 positions/sec depending on hardware and position complexity (rough
estimate, not a guarantee -- test on one shard first and look at the
printed rate before committing to annotating everything). At, say, 10
workers and 10/sec/worker = 100/sec, a single 1M-position shard takes
~2.75 hours. Annotating all 16 shards from your Lichess run at that rate
is multiple days of continuous compute. Practical options:
  - Lower --depth (8-10 is still meaningfully better than no eval at all)
  - Annotate a subset, not every shard (a few hundred thousand
    well-distributed positions still helps a lot more than zero)
  - Accept it's a background job that runs for a while, not a quick step

Usage:
    python annotate.py data_shards/lichess*.shard00000.jsonl \\
        --out data_shards_annotated \\
        --stockfish-path /usr/games/stockfish \\
        --depth 12 \\
        --workers 4

Requires: pip install python-chess
"""

import argparse
import json
import multiprocessing
import os
import time
from typing import Optional

import chess
import chess.engine

# Set per worker process by _init_worker -- not shared across processes,
# each worker gets its own Stockfish instance and its own limit.
_worker_engine: Optional["chess.engine.SimpleEngine"] = None
_worker_limit: Optional["chess.engine.Limit"] = None


def _init_worker(stockfish_path: str, depth: Optional[int], movetime: Optional[float]):
    global _worker_engine, _worker_limit
    _worker_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _worker_limit = chess.engine.Limit(depth=depth) if depth is not None else chess.engine.Limit(time=movetime)


def _annotate_one(rec: dict) -> Optional[dict]:
    global _worker_engine, _worker_limit
    fen = rec.get("fen") if isinstance(rec, dict) else None
    if not fen:
        return None
    try:
        board = chess.Board(fen)
        if board.is_game_over():
            return None
        info = _worker_engine.analyse(board, _worker_limit)
        pov_score = info["score"].white()  # White's-perspective score, matches result_to_white_score convention
        cp = pov_score.score(mate_score=100000)
        if cp is None:
            return None
    except Exception as e:
        print(f"[annotate] skipping a position due to error: {e}")
        return None

    annotated = dict(rec)
    annotated["eval"] = cp
    return annotated


def annotate_shard(
    in_path: str,
    out_path: str,
    stockfish_path: str,
    depth: Optional[int] = 12,
    movetime: Optional[float] = None,
    workers: int = 4,
    chunk_size: int = 200,
) -> None:
    if os.path.exists(out_path):
        print(f"[annotate] {out_path} already exists -- skipping (delete it first to re-annotate)")
        return

    with open(in_path, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[annotate] {in_path}: {len(records)} positions, depth={depth}, movetime={movetime}, workers={workers}")

    out_tmp = out_path + ".tmp"
    written = 0
    start = time.time()

    with multiprocessing.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(stockfish_path, depth, movetime),
    ) as pool, open(out_tmp, "w") as out_f:
        for i, result in enumerate(pool.imap(_annotate_one, records, chunksize=chunk_size), 1):
            if result is not None:
                out_f.write(json.dumps(result) + "\n")
                written += 1
            if i % 5000 == 0:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                eta_min = (len(records) - i) / rate / 60 if rate > 0 else float("inf")
                print(f"[annotate] {i}/{len(records)} ({rate:.1f}/sec, ~{eta_min:.0f} min left), {written} written")

    os.replace(out_tmp, out_path)
    elapsed = time.time() - start
    print(f"[annotate] done: {written}/{len(records)} annotated in {elapsed:.1f}s -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Annotate position shards with Stockfish evaluations.")
    p.add_argument("shards", nargs="+", help="Input .jsonl shard file(s)")
    p.add_argument("--out", default="data_shards_annotated", help="Output directory")
    p.add_argument("--stockfish-path", required=True, help="Path to the stockfish binary")
    p.add_argument("--depth", type=int, default=12, help="Stockfish search depth per position")
    p.add_argument("--movetime", type=float, default=None,
                    help="Use a per-position time budget (seconds) instead of --depth")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--chunk-size", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    for shard_path in args.shards:
        out_path = os.path.join(args.out, os.path.basename(shard_path))
        annotate_shard(
            shard_path,
            out_path,
            stockfish_path=args.stockfish_path,
            depth=None if args.movetime else args.depth,
            movetime=args.movetime,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
