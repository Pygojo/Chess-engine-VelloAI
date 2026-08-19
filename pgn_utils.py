"""
Shared helper for turning a movetext PGN string into position records.
Used by anything that has whole games in PGN form already (export_own_games.py,
download_chesscom.py) rather than a raw .pgn.zst stream (that's parser.py,
which has its own walk because it's reading one game at a time off a huge
file instead of a string already in memory).
"""

import io
from typing import Iterator, Optional

import chess.pgn


def positions_from_pgn(pgn_text: str, result_score: float, min_ply: int = 4) -> Iterator[dict]:
    """
    Replays a movetext PGN (headers optional -- python-chess tolerates
    their absence) and yields {"fen", "result", "ply"} for each position
    reached at or past min_ply.
    """
    if not pgn_text or not pgn_text.strip():
        return
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return
    board = game.board()
    for ply, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        if ply < min_ply:
            continue
        yield {"fen": board.fen(), "result": result_score, "ply": ply}


def pgn_result_header_to_score(result_header: str) -> Optional[float]:
    """Maps a PGN Result header ('1-0'/'0-1'/'1/2-1/2') to White's-perspective score."""
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result_header)
