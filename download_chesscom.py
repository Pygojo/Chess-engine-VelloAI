"""
Downloads games from chess.com's public API for one or more usernames and
writes them straight into the same JSONL shard schema parser.py produces.

Unlike Lichess, chess.com does NOT publish a single bulk "every game on the
site" dump -- their public API is per-player only. This script has to be
pointed at specific usernames (e.g. a list of strong players, or your own
users if they've connected a chess.com account -- that's a separate feature
this doesn't build).

No API key needed for public data, but chess.com's API requires a
descriptive User-Agent identifying the calling application or requests get
rate-limited/blocked -- set CHESSCOM_CONTACT below to something real.

Usage:
    python download_chesscom.py hikaru MagnusCarlsen --out data_shards/chesscom.jsonl

Requires: pip install requests python-chess
"""

import argparse
import json
import os
import time
from typing import Iterator, List

import requests

from config import SHARD_DIR
from pgn_utils import positions_from_pgn, pgn_result_header_to_score

# chess.com asks that requests identify the calling app/contact -- replace
# with something real before running this at any real volume.
CHESSCOM_CONTACT = "your-app-name (contact: you@example.com)"
HEADERS = {"User-Agent": CHESSCOM_CONTACT}

API_BASE = "https://api.chess.com/pub"


def get_archive_urls(username: str) -> List[str]:
    resp = requests.get(f"{API_BASE}/player/{username}/games/archives", headers=HEADERS, timeout=20)
    if resp.status_code == 404:
        print(f"[chesscom] no such user: {username}")
        return []
    resp.raise_for_status()
    return resp.json().get("archives", [])


def iter_games_for_archive(archive_url: str) -> Iterator[dict]:
    resp = requests.get(archive_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    yield from resp.json().get("games", [])


def download_user(username: str, out_f, min_ply: int = 4, sleep_between_months: float = 0.5) -> int:
    written = 0
    archive_urls = get_archive_urls(username)
    print(f"[chesscom] {username}: {len(archive_urls)} monthly archives")

    for i, archive_url in enumerate(archive_urls, 1):
        try:
            for game in iter_games_for_archive(archive_url):
                pgn_text = game.get("pgn")
                if not pgn_text:
                    continue
                # chess.com's PGN blob has a real [Result "..."] header,
                # unlike our own chess_games.pgn which has none.
                result_line = next(
                    (line for line in pgn_text.splitlines() if line.startswith("[Result ")),
                    None,
                )
                if not result_line:
                    continue
                result_header = result_line.split('"')[1] if '"' in result_line else None
                result_score = pgn_result_header_to_score(result_header) if result_header else None
                if result_score is None:
                    continue

                for rec in positions_from_pgn(pgn_text, result_score, min_ply=min_ply):
                    out_f.write(json.dumps(rec) + "\n")
                    written += 1
        except requests.HTTPError as e:
            print(f"[chesscom] skipping archive {archive_url}: {e}")
            continue

        if i % 10 == 0:
            print(f"[chesscom] {username}: {i}/{len(archive_urls)} months processed, {written} positions so far")
        time.sleep(sleep_between_months)  # be polite to the API

    return written


def download_chesscom(usernames: List[str], out_path: str, min_ply: int = 4) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    total_written = 0
    with open(out_path, "w") as out_f:
        for username in usernames:
            total_written += download_user(username, out_f, min_ply=min_ply)
    print(f"[chesscom] done: {total_written} positions written to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Download chess.com game archives for one or more usernames.")
    p.add_argument("usernames", nargs="+", help="chess.com usernames")
    p.add_argument("--out", default=os.path.join(SHARD_DIR, "chesscom.jsonl"))
    p.add_argument("--min-ply", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    if CHESSCOM_CONTACT.startswith("your-app-name"):
        print(
            "[chesscom] warning: CHESSCOM_CONTACT is still the placeholder -- "
            "edit it at the top of this file before running at real volume."
        )
    args = parse_args()
    download_chesscom(args.usernames, args.out, min_ply=args.min_ply)
