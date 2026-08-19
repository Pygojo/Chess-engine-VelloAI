"""
Feature encoding shared by parser.py, train.py, and model.py.

768-dim board encoding: 6 piece types x 2 colors x 64 squares, matching the
TypeScript engine's getFeatureIdx exactly:

    idx = (color_offset + piece_type) * 64 + square
    color_offset = 0 for white, 6 for black
    piece_type   = p=0, n=1, b=2, r=3, q=4, k=5

Do not change this encoding without also updating that TS function --
it's what makes exported weights meaningful to the engine at all.
"""

from typing import Optional

import numpy as np

PIECE_MAP = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}
PIECE_VALUES = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 0}

_MAX_PHASE_MATERIAL = 2 * (
    8 * PIECE_VALUES["p"]
    + 2 * PIECE_VALUES["n"]
    + 2 * PIECE_VALUES["b"]
    + 2 * PIECE_VALUES["r"]
    + PIECE_VALUES["q"]
)


def fen_to_features(fen: str) -> np.ndarray:
    board_str = fen.split()[0]
    features = np.zeros(768, dtype=np.float32)

    sq = 0
    for char in board_str:
        if char == "/":
            continue
        if char.isdigit():
            sq += int(char)
            continue
        if sq > 63:
            raise ValueError(f"FEN board field overflows 64 squares: {fen!r}")
        color_offset = 0 if char.isupper() else 6
        piece_type = PIECE_MAP.get(char.lower())
        if piece_type is None:
            raise ValueError(f"Unrecognized piece character {char!r} in FEN: {fen!r}")
        idx = (color_offset + piece_type) * 64 + sq
        features[idx] = 1.0
        sq += 1

    if sq != 64:
        raise ValueError(f"FEN board field covers {sq} squares, expected 64: {fen!r}")

    return features


def result_to_white_score(result_header: str) -> Optional[float]:
    """Maps a PGN Result header to White's-perspective score. None if unknown/ongoing."""
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result_header)


def game_phase(fen: str) -> float:
    """
    Rough material-based phase in [0, 1]: 1.0 = full material (opening),
    0.0 = bare kings (endgame). A cheap heuristic -- useful later as an
    auxiliary training signal, not required for the base evaluator.
    """
    board_str = fen.split()[0]
    total = 0
    for char in board_str:
        if char.isalpha():
            total += PIECE_VALUES.get(char.lower(), 0)
    return max(0.0, min(1.0, total / _MAX_PHASE_MATERIAL))
