"""
Chains the pipeline into one command: fetch latest Lichess month -> parse
into shards -> (optionally) pull your own app's finished games -> train ->
export. Meant to be run by hand or dropped into a cron job / scheduled
task -- see the bottom of this file's docstring for that.

Usage:
    python run_pipeline.py                       # Lichess only
    python run_pipeline.py --include-own-games    # + your app's chess_games
    python run_pipeline.py --skip-download        # reuse existing data_raw file

Scheduling this (pick whichever fits your machine):

  Linux/Mac cron, weekly on Sunday 3am, logging output:
      0 3 * * 0 cd /path/to/trainer && /path/to/venv/bin/python run_pipeline.py --include-own-games >> pipeline.log 2>&1

  Windows Task Scheduler: create a Basic Task, trigger "Weekly", action
  "Start a program" -> point it at your venv's python.exe with
  run_pipeline.py's full path as the argument, "Start in" set to the
  trainer/ folder.

This is a real GPU/CPU-bound job (the train step especially) -- don't put
it on a machine that also needs to be responsive for other things while
it runs, and don't expect a typical CI runner (e.g. free GitHub Actions
minutes) to have the GPU or the multi-hour time budget this needs at any
real dataset size.
"""

import argparse
import glob
import os

from config import DATA_DIR, SHARD_DIR, CHECKPOINT_DIR
from download_lichess import download, get_latest_available_month
from parser import write_shards
from export_own_games import export_own_games
from train import train_and_export


def run(
    include_own_games: bool = False,
    skip_download: bool = False,
    epochs: int = 50,
    output_weights: str = None,
):
    if not skip_download:
        month = get_latest_available_month()
        print(f"[pipeline] downloading {month}...")
        pgn_path = download(month)
        print(f"[pipeline] parsing into shards...")
        write_shards(pgn_path)
    else:
        print("[pipeline] --skip-download set, reusing existing data_raw/*.pgn.zst if already parsed")

    data_paths = glob.glob(os.path.join(SHARD_DIR, "*.jsonl"))

    if include_own_games:
        own_games_path = os.path.join(SHARD_DIR, "own_games.jsonl")
        print("[pipeline] exporting your app's finished games...")
        export_own_games(own_games_path)
        if own_games_path not in data_paths:
            data_paths.append(own_games_path)

    if not data_paths:
        raise SystemExit("[pipeline] no shard files found -- nothing to train on")

    print(f"[pipeline] training on {len(data_paths)} shard file(s)...")
    train_and_export(
        shard_paths=data_paths,
        output_weights_path=output_weights or os.path.join(os.path.dirname(__file__), "nnue_weights.json"),
        epochs=epochs,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, "latest.pt"),
        resume=True,
    )
    print("[pipeline] done.")


def parse_args():
    p = argparse.ArgumentParser(description="Run the full download -> parse -> train pipeline.")
    p.add_argument("--include-own-games", action="store_true")
    p.add_argument("--skip-download", action="store_true", help="Skip fetching a new Lichess dump")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        include_own_games=args.include_own_games,
        skip_download=args.skip_download,
        epochs=args.epochs,
        output_weights=args.output,
    )
