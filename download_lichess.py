"""
Downloads a monthly Lichess standard-rated PGN dump.

Streams to disk in fixed-size chunks -- never holds the whole file in
memory -- and resumes an interrupted download via HTTP Range if a partial
file already exists at the destination. The .pgn.zst file is NOT
decompressed here; parser.py reads directly from the compressed stream.

Usage:
    python download_lichess.py 2026-07
    python download_lichess.py 2026-07 --out data_raw/custom_name.pgn.zst

Requires: pip install requests
"""

import argparse
import datetime
import os
import sys
from typing import Optional

import requests

from config import LICHESS_BASE_URL, DATA_DIR

CHUNK_SIZE = 1024 * 1024  # 1 MB


def _month_exists(year_month: str) -> bool:
    url = LICHESS_BASE_URL.format(year_month=year_month)
    resp = requests.head(url, timeout=15, allow_redirects=True)
    return resp.status_code == 200


def get_latest_available_month(start_months_back: int = 1, search_limit: int = 6) -> str:
    """
    Lichess publishes each month's dump a few days into the following
    month, so "this month's" file usually doesn't exist yet. Starts one
    month back and walks further back until it finds one that actually
    responds 200, instead of you having to guess/hardcode an offset.
    """
    today = datetime.date.today().replace(day=1)
    for i in range(start_months_back, start_months_back + search_limit):
        # subtract i months
        month = today.month - i
        year = today.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        candidate = f"{year:04d}-{month:02d}"
        print(f"[download_lichess] checking {candidate}...")
        if _month_exists(candidate):
            return candidate
    raise RuntimeError(
        f"No available Lichess dump found in the last {search_limit} months -- "
        f"check https://database.lichess.org/ manually."
    )


def download(year_month: str, out_path: Optional[str] = None) -> str:
    url = LICHESS_BASE_URL.format(year_month=year_month)
    if out_path is None:
        out_path = os.path.join(DATA_DIR, f"lichess_db_standard_rated_{year_month}.pgn.zst")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    resume_from = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(url, headers=headers, stream=True, timeout=30) as resp:
        if resume_from and resp.status_code == 200:
            # Server ignored the Range request (doesn't support resuming) --
            # this response is the full file from byte 0, so start over
            # rather than appending a second copy onto the partial file.
            resume_from = 0
        elif resp.status_code not in (200, 206):
            raise RuntimeError(f"Download failed: HTTP {resp.status_code} for {url}")

        content_length = resp.headers.get("Content-Length")
        total_bytes = (int(content_length) + resume_from) if content_length else None

        mode = "ab" if resume_from else "wb"
        written = resume_from
        with open(out_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                _report_progress(written, total_bytes)

    print(f"\nSaved to {out_path}")
    return out_path


def _report_progress(written: int, total: Optional[int]):
    mb = written / (1024 * 1024)
    if total:
        pct = 100 * written / total
        sys.stdout.write(f"\r{mb:,.1f} MB downloaded ({pct:.1f}%)")
    else:
        sys.stdout.write(f"\r{mb:,.1f} MB downloaded")
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description="Download a monthly Lichess standard-rated PGN dump.")
    p.add_argument("year_month", nargs="?", default=None, help="e.g. 2026-07. Omit if using --latest")
    p.add_argument("--latest", action="store_true", help="Auto-resolve the most recent available month")
    p.add_argument("--out", default=None, help="Output path (default: data_raw/<name>.pgn.zst)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.latest:
        target_month = get_latest_available_month()
        print(f"[download_lichess] using latest available: {target_month}")
    elif args.year_month:
        target_month = args.year_month
    else:
        raise SystemExit("Provide a year_month (e.g. 2026-07) or pass --latest")
    download(target_month, args.out)
