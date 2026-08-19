"""
Trains the NNUE model (see model.py) and exports weights for the TS engine.

Reads shard files via the streaming dataset in dataset.py -- RAM usage
stays flat whether you point --data at 2 shards or 200. There is no more
"does this fit in memory" question to answer before running this.

Checkpointing happens two ways now:
  --checkpoint-every-epochs : save at the end of every N epochs (as before)
  --checkpoint-every-steps  : ALSO save every N batches within an epoch

The step-based checkpoint exists specifically so a mid-epoch disconnect
(Colab losing connection, etc.) loses at most a few minutes of progress
instead of the whole epoch -- one full pass over a large shard set can
take a long time, and epoch boundaries alone aren't frequent enough to
checkpoint against on an unreliable connection.

Same record schema as before, still JSONL or a single JSON array:
    {"fen": "<FEN>", "result": <0|0.5|1>, "eval": <centipawns, optional>}
"""

import argparse
import os
import random
from typing import List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR
from dataset import StreamingChessDataset
from model import NNUE, export_weights


# --- CHECKPOINTING -----------------------------------------------------------
def save_checkpoint(path: str, model: NNUE, optimizer, scheduler, epoch: int,
                     step_in_epoch: int, best_val: float):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": best_val,
        },
        tmp_path,
    )
    # Atomic-ish replace: if the process dies mid-write (the exact failure
    # mode this checkpointing is meant to protect against), you're left
    # with a stale-but-intact previous checkpoint, not a half-written
    # corrupt one that torch.load() can't even read.
    os.replace(tmp_path, path)


def load_checkpoint(path: str, model: NNUE, optimizer, scheduler) -> tuple:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt.get("step_in_epoch", 0), ckpt["best_val"]


# --- TRAINING LOOP -------------------------------------------------------------
def train_and_export(
    shard_paths: List[str],
    output_weights_path: str,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,
    val_split: float = 0.05,
    shuffle_buffer_size: int = 200_000,
    seed: int = 0,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    checkpoint_every_epochs: int = 1,
    checkpoint_every_steps: int = 5000,
    log_every_steps: int = 200,
    num_workers: int = 0,
):
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    for p in shard_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Shard file missing: {p}")
    print(f"Streaming from {len(shard_paths)} shard file(s), val_split={val_split}")

    train_ds = StreamingChessDataset(
        shard_paths, val_split=val_split, want_val=False,
        shuffle_buffer_size=shuffle_buffer_size, seed=seed,
    )
    val_ds = (
        StreamingChessDataset(
            shard_paths, val_split=val_split, want_val=True,
            shuffle_buffer_size=max(1000, shuffle_buffer_size // 10), seed=seed,
        )
        if val_split > 0 else None
    )

    # shuffle=True is not valid for IterableDataset -- shuffling already
    # happens inside StreamingChessDataset via the shuffle buffer.
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers) if val_ds else None

    model = NNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    start_epoch = 0
    best_val = float("inf")
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, _resumed_step, best_val = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        print(f"Resumed from checkpoint at epoch {start_epoch}, best_val={best_val:.6f}")
        # Mid-epoch resume (picking up at the exact batch) isn't
        # implemented -- IterableDataset doesn't support seeking into the
        # middle of the stream. Resuming restarts the epoch it was
        # interrupted in from the beginning, which still saves everything
        # up through the last COMPLETED epoch.

    print(f"Beginning training for {epochs} epochs...")

    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_count = 0
        step = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            raw_eval = model(bx)
            win_prob = torch.sigmoid(raw_eval / 400.0)
            loss = torch.mean((win_prob - by) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_count += 1
            step += 1

            if step % log_every_steps == 0:
                avg = running_loss / running_count
                print(f"  epoch {epoch + 1}, step {step} ({step * batch_size:,} positions) - avg loss: {avg:.6f}")

            if checkpoint_path and checkpoint_every_steps and step % checkpoint_every_steps == 0:
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, step, best_val)
                print(f"  [checkpoint] saved at epoch {epoch + 1}, step {step}")

        scheduler.step()
        avg_train_loss = running_loss / max(1, running_count)
        current_lr = scheduler.get_last_lr()[0]

        val_msg = ""
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    win_prob = torch.sigmoid(model(bx) / 400.0)
                    val_loss += torch.mean((win_prob - by) ** 2).item()
                    val_batches += 1
            if val_batches > 0:
                val_loss /= val_batches
                val_msg = f" - Val: {val_loss:.6f}"
                if val_loss < best_val:
                    best_val = val_loss

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_train_loss:.6f}{val_msg} - LR: {current_lr:.6f}")

        if checkpoint_path and (epoch + 1) % checkpoint_every_epochs == 0:
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch + 1, 0, best_val)
            print(f"Checkpoint saved at end of epoch {epoch + 1}")

    if checkpoint_path:
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epochs, 0, best_val)

    export_weights(model, output_weights_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and export NNUE weights.")
    parser.add_argument(
        "--data", nargs="+", default=None,
        help="One or more shard paths (.jsonl, or a small .json array)",
    )
    parser.add_argument("--output", default=None, help="Path to write nnue_weights.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.05, help="0 to disable validation")
    parser.add_argument("--shuffle-buffer", type=int, default=200_000,
                         help="Records held in memory for shuffling -- ~3KB each, 200k ~= 600MB")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="Path to save/resume checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint if it exists")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=5000,
                         help="Also checkpoint mid-epoch every N batches, 0 to disable")
    parser.add_argument("--log-every-steps", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0,
                         help="DataLoader workers; each reads a disjoint subset of shard files")
    return parser.parse_args()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args = parse_args()

    data_paths = args.data or [os.path.join(script_dir, "data.json")]
    weights_file = args.output or os.path.join(script_dir, "nnue_weights.json")
    checkpoint_file = args.checkpoint or os.path.join(CHECKPOINT_DIR, "latest.pt")

    train_and_export(
        shard_paths=data_paths,
        output_weights_path=weights_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        shuffle_buffer_size=args.shuffle_buffer,
        seed=args.seed,
        checkpoint_path=checkpoint_file,
        resume=args.resume,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        checkpoint_every_steps=args.checkpoint_every_steps,
        log_every_steps=args.log_every_steps,
        num_workers=args.num_workers,
    )
