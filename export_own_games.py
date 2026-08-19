"""
Exports finished games from your app's own `chess_games` Supabase table
into the same {"fen", "result", "ply"} JSONL schema parser.py produces --
so games actually played against your engine (by real users, at whatever
difficulty they picked) can be trained on directly, blended with or on top
of Lichess data.

Needs env vars:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY   (service role, not anon -- this reads
                                  across ALL users; RLS on the anon/user
                                  key would only return your own games)

Requires: pip install supabase python-chess

Usage:
    export SUPABASE_URL=...
    export SUPABASE_SERVICE_ROLE_KEY=...
    python export_own_games.py --out data_shards/own_games.jsonl
"""

import argparse
import json
import os
from typing import Optional

from config import SHARD_DIR
from pgn_utils import positions_from_pgn

try:
    from supabase import create_client
except ImportError as e:
    raise SystemExit("Missing dependency: pip install supabase") from e


# (status, player_color) -> White's-perspective result score. player_color
# is which color the HUMAN played; "won"/"lost" are from the human's side.
STATUS_TO_RESULT = {
    ("won", "w"): 1.0,
    ("won", "b"): 0.0,
    ("lost", "w"): 0.0,
    ("lost", "b"): 1.0,
    ("draw", "w"): 0.5,
    ("draw", "b"): 0.5,
}


def game_result_score(status: str, player_color: str) -> Optional[float]:
    return STATUS_TO_RESULT.get((status, player_color))


def export_own_games(out_path: str, min_ply: int = 4, page_size: int = 500) -> None:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    written = 0
    skipped_games = 0
    offset = 0

    with open(out_path, "w") as out_f:
        while True:
            resp = (
                client.table("chess_games")
                .select("pgn, player_color, status")
                .in_("status", ["won", "lost", "draw"])
                .range(offset, offset + page_size - 1)
                .execute()
            )
            rows = resp.data or []
            if not rows:
                break

            for row in rows:
                result_score = game_result_score(row["status"], row["player_color"])
                if result_score is None:
                    skipped_games += 1
                    continue
                for rec in positions_from_pgn(row["pgn"], result_score, min_ply=min_ply):
                    out_f.write(json.dumps(rec) + "\n")
                    written += 1

            offset += page_size
            print(f"[export_own_games] processed {offset} rows, {written} positions so far")

    print(
        f"[export_own_games] done: {written} positions written to {out_path} "
        f"({skipped_games} games skipped -- unresolved status or missing color)"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Export finished chess_games rows into a training JSONL shard.")
    p.add_argument("--out", default=os.path.join(SHARD_DIR, "own_games.jsonl"))
    p.add_argument("--min-ply", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_own_games(args.out, min_ply=args.min_ply)
