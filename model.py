"""
NNUE model architecture.

Two parts, deliberately treated differently:

1. Accumulator layer (768 -> ACC_SIZE): this is the part the TS engine
   updates INCREMENTALLY -- one row of w1 added/subtracted per piece move,
   never recomputed from scratch. That's the entire reason NNUE search is
   fast. This has to stay a single plain Linear layer with no batch norm,
   because incremental updates only make sense for a linear transform.

2. Trunk (ACC_SIZE -> TRUNK1 -> TRUNK2 -> 1): this runs in full on every
   leaf node the search evaluates, not incrementally. That's fine -- it's
   small (a few thousand multiply-adds), so the cost is trivial next to
   the cost of the search itself. This is where the extra depth from a
   "make it a bigger network" request should go, not into the accumulator.

This mirrors how Stockfish's own NNUE is structured (big sparse
incremental input, small dense trunk) rather than making the whole
network deep, which would kill incremental-update search performance.

Both stages use plain (unclipped) ReLU to match the existing TS
evaluateNNUE exactly -- see updateAcc/evaluateNNUE in the engine.
"""

import json
import os

import torch
import torch.nn as nn

ACC_SIZE = 256    # accumulator width -- unchanged, this is the incremental part
TRUNK1_SIZE = 64   # widened from 32 -- more capacity, ~2x trunk compute, not ~4.6x
TRUNK2_SIZE = 32
NUM_FEATURES = 768


class NNUE(nn.Module):
    def __init__(self, num_features: int = NUM_FEATURES, acc_size: int = ACC_SIZE,
                 trunk1_size: int = TRUNK1_SIZE, trunk2_size: int = TRUNK2_SIZE):
        super().__init__()
        self.input_layer = nn.Linear(num_features, acc_size)   # incremental in TS
        self.trunk1 = nn.Linear(acc_size, trunk1_size)          # full eval per node
        self.trunk2 = nn.Linear(trunk1_size, trunk2_size)
        self.output_layer = nn.Linear(trunk2_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = torch.relu(self.input_layer(x))
        h1 = torch.relu(self.trunk1(acc))
        h2 = torch.relu(self.trunk2(h1))
        return self.output_layer(h2)


def export_weights(model: NNUE, output_path: str):
    """
    Exports weights in the layout the TS engine expects (see the matching
    patch to evaluateNNUE/NNUEWeights in the engine):

    - w1: the accumulator layer, TRANSPOSED to (num_features, acc_size) and
      flattened feature-major, so updateAcc can add/subtract one
      contiguous acc_size-length row per feature. This layout is what
      makes incremental updates possible at all -- do not change it.
    - b1: accumulator bias, as-is.
    - w_trunk1/w_trunk2/w_out: NOT transposed -- these run as a normal
      full forward pass per node, so they're kept in PyTorch's native
      (out_features, in_features) row-major layout, i.e.
      w[out_idx * in_size + in_idx]. The TS forward pass indexes them the
      same way.
    - b_trunk1/b_trunk2/b_out: biases, as-is. b_out is a scalar.
    """
    model = model.eval().cpu()

    w1_transposed = model.input_layer.weight.t().contiguous().view(-1).tolist()
    b1_flat = model.input_layer.bias.tolist()

    w_trunk1 = model.trunk1.weight.contiguous().view(-1).tolist()
    b_trunk1 = model.trunk1.bias.tolist()

    w_trunk2 = model.trunk2.weight.contiguous().view(-1).tolist()
    b_trunk2 = model.trunk2.bias.tolist()

    w_out = model.output_layer.weight.view(-1).tolist()
    b_out = model.output_layer.bias.item()

    payload = {
        "w1": w1_transposed,
        "b1": b1_flat,
        "w_trunk1": w_trunk1,
        "b_trunk1": b_trunk1,
        "w_trunk2": w_trunk2,
        "b_trunk2": b_trunk2,
        "w_out": w_out,
        "b_out": b_out,
        # Shapes included so the TS side can validate on load instead of
        # silently misreading a mismatched file -- see validateNNUEWeights.
        "acc_size": model.input_layer.out_features,
        "trunk1_size": model.trunk1.out_features,
        "trunk2_size": model.trunk2.out_features,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f)
    print(f"NNUE weights exported to: {output_path}")
