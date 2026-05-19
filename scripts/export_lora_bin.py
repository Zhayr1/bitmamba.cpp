#!/usr/bin/env python3
"""
Convert a BitMamba-2 LoRA checkpoint (.lora.pth) into the compact .lora.bin
format consumed by the C++ inference engine.

The .pth is a PyTorch state dict produced by BitMambaFactory training, with
keys of the form:

    layers.{i}.lora_W_in.A   shape [rank, in_features]   fp32
    layers.{i}.lora_W_in.B   shape [out_features, rank]  fp32
    layers.{i}.lora_W_out.A  shape [rank, in_features]   fp32
    layers.{i}.lora_W_out.B  shape [out_features, rank]  fp32

The binary format (little-endian, IEEE-754 fp32):

    Header (24 bytes):
        u32 magic       = 0x4C4F5232  ("LOR2")
        u32 version     = 1
        u32 rank
        f32 alpha
        u32 n_layers
        u32 n_targets   = 2 (fixed: target 0 = W_in, target 1 = W_out)

    For layer_idx in [0, n_layers):
      For target_idx in [0, n_targets):
        u32 layer_idx           (redundant — sanity check)
        u32 target_idx          (0=W_in, 1=W_out)
        u32 out_features        (rows of B, rows of W_base)
        u32 in_features         (cols of A, cols of W_base)
        u32 rank                (matches header)
        f32[rank * in_features]    A   row-major
        f32[out_features * rank]   B   row-major

The scale = alpha / rank is recomputed on the C++ side from the header.
"""

import argparse
import struct
import sys
from pathlib import Path

import torch


MAGIC_LOR2 = 0x4C4F5232
VERSION    = 1
TARGETS    = ("W_in", "W_out")   # target_idx == index in this tuple


def _layer_keys(sd, layer_idx, target):
    return (
        f"layers.{layer_idx}.lora_{target}.A",
        f"layers.{layer_idx}.lora_{target}.B",
    )


def collect_layer_indices(sd):
    import re
    ixs = set()
    pat = re.compile(r"layers\.(\d+)\.lora_")
    for k in sd:
        m = pat.search(k)
        if m:
            ixs.add(int(m.group(1)))
    return sorted(ixs)


def infer_rank(sd, layer_indices):
    """Take rank from the first available A tensor."""
    for li in layer_indices:
        for tgt in TARGETS:
            k_a, _ = _layer_keys(sd, li, tgt)
            if k_a in sd:
                return int(sd[k_a].shape[0])
    raise RuntimeError("No lora_*.A tensors found in checkpoint")


def export(pth_path: Path, out_path: Path, alpha: float):
    print(f"Loading {pth_path}")
    sd = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    layer_indices = collect_layer_indices(sd)
    if not layer_indices:
        sys.exit("ERROR: no LoRA keys found in checkpoint")

    n_layers = layer_indices[-1] + 1
    if layer_indices != list(range(n_layers)):
        sys.exit(f"ERROR: non-contiguous layer indices found: {layer_indices}")

    rank = infer_rank(sd, layer_indices)
    print(f"  n_layers={n_layers}  rank={rank}  alpha={alpha}  scale={alpha/rank:.4f}")

    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack(
            "<IIIfII",
            MAGIC_LOR2,
            VERSION,
            rank,
            float(alpha),
            n_layers,
            len(TARGETS),
        ))

        total_floats = 0
        for li in range(n_layers):
            for tgt_idx, tgt in enumerate(TARGETS):
                k_a, k_b = _layer_keys(sd, li, tgt)
                if k_a not in sd or k_b not in sd:
                    sys.exit(f"ERROR: missing keys for layer {li} target {tgt}")
                A = sd[k_a].detach().to(torch.float32).contiguous()
                B = sd[k_b].detach().to(torch.float32).contiguous()

                # A: [rank, in_features]
                # B: [out_features, rank]
                if A.shape[0] != rank or B.shape[1] != rank:
                    sys.exit(f"ERROR: rank mismatch at layer {li}/{tgt}: "
                             f"A={tuple(A.shape)} B={tuple(B.shape)} expected rank={rank}")
                in_features  = int(A.shape[1])
                out_features = int(B.shape[0])

                # Slot header
                f.write(struct.pack(
                    "<IIIII",
                    li,
                    tgt_idx,
                    out_features,
                    in_features,
                    rank,
                ))
                # Tensor payload
                f.write(A.numpy().tobytes())
                f.write(B.numpy().tobytes())
                total_floats += A.numel() + B.numel()

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path}  size={size_mb:.1f} MB  fp32_count={total_floats}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pth",    required=True, type=Path, help="Input .lora.pth checkpoint")
    p.add_argument("--output", required=True, type=Path, help="Output .lora.bin path")
    p.add_argument("--alpha",  type=float, default=32.0,
                   help="LoRA alpha hyperparameter (default: 32.0, matches BitMambaFactory recipe)")
    args = p.parse_args()
    export(args.pth, args.output, args.alpha)


if __name__ == "__main__":
    main()
