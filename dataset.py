"""
Streaming dataset for JSONL position shards. Reads records lazily in
batches instead of loading everything into RAM upfront -- this is what
removes the "how many shards fit in memory" question you've been hitting
repeatedly, regardless of whether you point it at 2 shards or 200.

Trade-off: no true global shuffle (that needs everything resident in
memory, or pre-shuffled on disk). Uses a shuffle buffer instead -- reads
ahead into a buffer of `shuffle_buffer_size` records, yields a random one
from it, refills from the stream. Same approach TensorFlow's
tf.data.Dataset.shuffle() uses for exactly this reason. A buffer of a few
hundred thousand records gives good-enough shuffling without needing the
whole dataset resident at once (200k records ~= 600MB, not 35GB).

Train/val split is per-record, via a deterministic hash of the FEN -- not
by holding out whole shard files. That means both splits see the same mix
of games, and you don't need to know the total position count upfront
(which you can't, in a streaming setup) to do the split.
"""

import hashlib
import json
import math
import random
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from features import fen_to_features


def _is_val_record(fen: str, val_split: float) -> bool:
    if val_split <= 0:
        return False
    h = int(hashlib.md5(fen.encode("utf-8")).hexdigest(), 16)
    return (h % 10_000) < int(val_split * 10_000)


def _iter_shard_records(path: str) -> Iterator[dict]:
    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Skip a truncated/corrupt line (e.g. the shard that
                    # was mid-write when a runtime got interrupted)
                    # instead of crashing the whole run over one record.
                    continue
    else:
        # Plain JSON array -- loaded fully into memory. Only sane for
        # small files (e.g. a hand-written toy dataset); shards from
        # parser.py are always .jsonl and never hit this branch.
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON array of position records")
        yield from data


class StreamingChessDataset(IterableDataset):
    def __init__(
        self,
        shard_paths: List[str],
        val_split: float = 0.05,
        want_val: bool = False,
        shuffle_buffer_size: int = 200_000,
        seed: int = 0,
    ):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.val_split = val_split
        self.want_val = want_val
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.epoch = 0  # bumped via set_epoch() so shuffle order varies per epoch

    def set_epoch(self, epoch: int) -> None:
        """
        Call once per epoch, before iterating, so the shuffle buffer
        produces a different order each time -- without this, __iter__
        rebuilding its RNG from the same fixed seed every epoch means
        every epoch sees positions in the exact same order, which drags
        on how well SGD converges (same mirrors PyTorch's
        DistributedSampler.set_epoch() pattern for the same reason).
        """
        self.epoch = epoch

    def _my_shard_paths(self) -> List[str]:
        """Splits shard files across DataLoader workers so each worker
        reads a disjoint subset instead of every worker re-reading
        everything from scratch."""
        info = get_worker_info()
        if info is None:
            return self.shard_paths
        return self.shard_paths[info.id :: info.num_workers]

    def _filtered_records(self) -> Iterator[dict]:
        for path in self._my_shard_paths():
            for rec in _iter_shard_records(path):
                fen = rec.get("fen") if isinstance(rec, dict) else None
                if not fen:
                    continue
                if _is_val_record(fen, self.val_split) != self.want_val:
                    continue
                yield rec

    @staticmethod
    def _to_tensor_pair(rec: dict) -> Optional[tuple]:
        try:
            feats = fen_to_features(rec["fen"])
        except (ValueError, KeyError):
            return None

        result = rec.get("result")
        if result is None:
            result = 0.5

        eval_score = rec.get("eval")
        if eval_score is None:
            # No search eval for this position -- train on outcome alone
            # rather than diluting it through the blended formula below.
            target = float(result)
        else:
            eval_prob = 1.0 / (1.0 + math.exp(-float(eval_score) / 400.0))
            target = 0.8 * eval_prob + 0.2 * float(result)

        x = torch.from_numpy(feats)
        y = torch.tensor([target], dtype=torch.float32)
        return x, y

    def __iter__(self):
        info = get_worker_info()
        worker_seed = self.seed + self.epoch * 1_000_003 + (info.id if info else 0)
        rng = random.Random(worker_seed)

        buffer = []
        for rec in self._filtered_records():
            pair = self._to_tensor_pair(rec)
            if pair is None:
                continue

            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(pair)
                continue

            idx = rng.randrange(len(buffer))
            yield buffer[idx]
            buffer[idx] = pair

        rng.shuffle(buffer)
        for pair in buffer:
            yield pair
