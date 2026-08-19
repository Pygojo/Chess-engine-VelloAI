"""
Streams positions directly out of a Lichess .pgn.zst dump, one game at a
time, without ever decompressing the file to disk or holding more than one
game in memory at once. Memory usage stays effectively constant whether the
file has a million games or a billion.

How: zstandard's stream_reader wraps the compressed file as a normal
readable stream; python-chess's read_game() then reads exactly one game at
a time off of that stream (it's a generator-style parser itself, not a
load-everything-then-split one). Neither library buffers the whole file.

Usage as a library:
    from parser import iter_positions
    for rec in iter_positions("data_raw/lichess_db_standard_rated_2026-07.pgn.zst"):
        ...  # {"fen", "result", "ply", "avg_elo"}

Usage as a CLI (writes sharded JSONL files that train.py can already read):
    python parser.py data_raw/lichess_db_standard_rated_2026-07.pgn.zst

Requires: pip install python-chess zstandard
"""

import argparse
import io
import json
import os
import random
from typing import Iterator, Optional

import chess
import chess.pgn
import zstandard as zstd

from config import (
    MIN_PLY,
    MAX_PLY,
    MIN_GAME_PLIES,
    MIN_ELO,
    POSITION_SAMPLE_RATE,
    POSITIONS_PER_SHARD,
    SHARD_DIR,
)
from features import result_to_white_score


def _open_pgn_stream(path: str) -> io.TextIOWrapper:
    fh = open(path, "rb")
    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(fh)
    return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")


def _average_elo(headers: "chess.pgn.Headers") -> Optional[int]:
    try:
        w = int(headers.get("WhiteElo", ""))
        b = int(headers.get("BlackElo", ""))
        return (w + b) // 2
    except (TypeError, ValueError):
        return None


def iter_positions(
    pgn_zst_path: str,
    min_ply: int = MIN_PLY,
    max_ply: int = MAX_PLY,
    min_game_plies: int = MIN_GAME_PLIES,
    min_elo: Optional[int] = MIN_ELO,
    sample_rate: float = POSITION_SAMPLE_RATE,
    seed: int = 0,
) -> Iterator[dict]:
    """
    Yields {"fen", "result", "ply", "avg_elo"} dicts, one per sampled
    position. No "eval" key -- these are raw, unannotated positions;
    train.py already treats a missing eval as "train on game outcome only"
    for exactly this case. Add a Stockfish annotation pass later if you
    want search-eval-based targets too.
    """
    rng = random.Random(seed)
    text_stream = _open_pgn_stream(pgn_zst_path)

    games_seen = 0
    games_kept = 0
    positions_yielded = 0

    try:
        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break  # end of file
            games_seen += 1

            headers = game.headers
            result_score = result_to_white_score(headers.get("Result", ""))
            if result_score is None:
                continue  # ongoing/aborted/unknown result

            avg_elo = _average_elo(headers)
            if min_elo is not None and (avg_elo is None or avg_elo < min_elo):
                continue

            moves = list(game.mainline_moves())
            if len(moves) < min_game_plies:
                continue

            games_kept += 1
            board = game.board()
            for ply, move in enumerate(moves, start=1):
                board.push(move)

                if ply < min_ply or ply > max_ply:
                    continue
                if sample_rate < 1.0 and rng.random() > sample_rate:
                    continue

                yield {
                    "fen": board.fen(),
                    "result": result_score,
                    "ply": ply,
                    "avg_elo": avg_elo,
                }
                positions_yielded += 1

            if games_seen % 5000 == 0:
                print(
                    f"[parser] {games_seen} games read, {games_kept} kept, "
                    f"{positions_yielded} positions yielded"
                )
    finally:
        text_stream.close()

    print(
        f"[parser] done: {games_seen} games read, {games_kept} kept, "
        f"{positions_yielded} positions yielded"
    )


def write_shards(
    pgn_zst_path: str,
    out_dir: str = SHARD_DIR,
    positions_per_shard: int = POSITIONS_PER_SHARD,
    **iter_kwargs,
) -> list:
    """
    Streams positions from a .pgn.zst file straight into sharded JSONL
    files -- the same format train.py's iter_records() already reads via
    its .jsonl branch. Each shard is flushed and closed as soon as it's
    full, so peak memory is one shard's worth of buffering at most, not
    the whole dataset.
    """
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pgn_zst_path))[0]
    if base_name.endswith(".pgn"):
        base_name = base_name[: -len(".pgn")]

    shard_paths = []
    shard_idx = 0
    count_in_shard = 0
    out_f = None

    def _open_next_shard():
        nonlocal out_f, shard_idx, count_in_shard
        if out_f is not None:
            out_f.close()
        shard_path = os.path.join(out_dir, f"{base_name}.shard{shard_idx:05d}.jsonl")
        out_f = open(shard_path, "w")
        shard_paths.append(shard_path)
        shard_idx += 1
        count_in_shard = 0

    _open_next_shard()
    for rec in iter_positions(pgn_zst_path, **iter_kwargs):
        out_f.write(json.dumps(rec) + "\n")
        count_in_shard += 1
        if count_in_shard >= positions_per_shard:
            _open_next_shard()

    if out_f is not None:
        out_f.close()

    print(f"[parser] wrote {len(shard_paths)} shard(s) to {out_dir}")
    return shard_paths


def parse_args():
    p = argparse.ArgumentParser(description="Stream positions out of a Lichess .pgn.zst file into JSONL shards.")
    p.add_argument("pgn_zst_path")
    p.add_argument("--out-dir", default=SHARD_DIR)
    p.add_argument("--positions-per-shard", type=int, default=POSITIONS_PER_SHARD)
    p.add_argument("--min-elo", type=int, default=MIN_ELO)
    p.add_argument("--sample-rate", type=float, default=POSITION_SAMPLE_RATE)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_shards(
        args.pgn_zst_path,
        out_dir=args.out_dir,
        positions_per_shard=args.positions_per_shard,
        min_elo=args.min_elo,
        sample_rate=args.sample_rate,
    )
