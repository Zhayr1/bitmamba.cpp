import struct
import numpy as np
import os
import sys
from loader import load_checkpoint
from tqdm import tqdm
from config import CONFIG_1B, CONFIG_255M
from torch_model import BitMambaLM

def pack_ternary_weights(weights, layer_name):
    """
    Packs ternary weights (-1, 0, 1) into 2-bit format.
    weights: shape [rows, cols]
    """
    rows, cols = weights.shape
    
    # --- SAFETY CHECK 1: Alignment ---
    # The C++ kernel iterates by rows and skips memory based on (cols/4).
    # If cols is not divisible by 4, Python flattens everything and misaligns subsequent rows.
    if cols % 4 != 0:
        raise ValueError(f"❌ Fatal error in {layer_name}: 'cols' ({cols}) must be divisible by 4 for BitNet packing.")

    # Flatten
    flat = weights.flatten()
    
    # Mapping: -1->0, 0->1, 1->2
    # Add 1 to shift from [-1,0,1] to [0,1,2]
    flat_shifted = (flat + 1).astype(np.uint8)
    
    # Pack 4 values into 1 byte
    # Reshape to (N, 4) to vectorize
    reshaped = flat_shifted.reshape(-1, 4)
    
    # Bitwise operation:
    # byte = p0 | (p1 << 2) | (p2 << 4) | (p3 << 6)
    packed = (reshaped[:, 0] | 
              (reshaped[:, 1] << 2) | 
              (reshaped[:, 2] << 4) | 
              (reshaped[:, 3] << 6))
    
    return packed.astype(np.uint8)

def export_packed(model, output_path):
    print(f"📦 Exporting PACKED (2-bit) to {output_path}...")
    
    total_params = 0
    compressed_bytes = 0
    
    with open(output_path, "wb") as f:
        # Header: Magic 0xBIT2
        header = struct.pack("iiiii", 0x42495432, model.vocab_size, model.d_model, model.n_layers, model.n_heads)
        f.write(header)
        
        state_dict = model.state_dict()
        
        # Sorted list of keys to traverse
        # It is vital to keep the exact order expected by C++
        keys_to_write = ["embed.weight"]
        for i in range(model.n_layers):
            base = f"layers.{i}."
            keys_to_write.extend([
                base + "in_proj.norm.weight", base + "in_proj.weight",
                base + "conv1d.weight", base + "conv1d.bias",
                base + "dt_bias", base + "A_log", base + "D",
                base + "out_proj.norm.weight", base + "out_proj.weight"
            ])
        keys_to_write.extend(["norm_f.weight", "lm_head.norm.weight", "lm_head.weight"])

        progress = tqdm(keys_to_write, unit="layer")
        
        for name in progress:
            if name not in state_dict:
                # Special handling for embed/head if they have different names in state_dict
                # (Sometimes happens if the model has no prefixes)
                print(f"⚠️ Warning: {name} not found in state_dict, looking for fallback...")
                continue
                
            tensor = state_dict[name]
            data = tensor.detach().cpu().float().numpy()
            
            # --- SAFETY CHECK 2: NaNs ---
            if np.isnan(data).any():
                print(f"\n❌ CRITICAL: NaNs found in {name}. Model is corrupt.")
                sys.exit(1)

            # Detect if it is a BitLinear layer (Weights W, no bias, no norm)
            is_bitlinear_weight = "weight" in name and ("in_proj" in name or "out_proj" in name or "lm_head" in name) and "norm" not in name and "embed" not in name

            if is_bitlinear_weight:
                # Quantization
                # Use absolute mean as scale (BitNet b1.58 standard)
                mean_abs = np.mean(np.abs(data))
                if mean_abs == 0:
                     print(f"\n⚠️ Warning: Dead layer (all zeros): {name}")
                     scale = 1.0
                else:
                    scale = 1.0 / mean_abs
                
                # Round to nearest & Clip
                quant = np.clip(np.round(data * scale), -1, 1).astype(np.int8)
                
                # Stats for debug
                unique = np.unique(quant)
                if len(unique) == 1 and unique[0] == 0:
                     progress.write(f"⚠️ {name} collapsed to only ZEROS after quantization.")

                # Pack
                packed_data = pack_ternary_weights(quant, name)
                
                # Write Binary (ID 2 = BitNet)
                f.write(struct.pack("i", 2)) 
                f.write(struct.pack("ii", data.shape[0], data.shape[1])) # Rows, Cols
                f.write(struct.pack("f", float(scale)))
                f.write(packed_data.tobytes())
                
                compressed_bytes += packed_data.nbytes
                
            else:
                # FP32 Normal (ID 0)
                f.write(struct.pack("i", 0))
                dims = data.shape
                f.write(struct.pack("i", len(dims)))
                for d in dims: f.write(struct.pack("i", d))
                f.write(data.tobytes())
                
                compressed_bytes += data.nbytes

            total_params += data.size

    print(f"\n✅ Export completed.")
    print(f"   Final size: {compressed_bytes / 1024 / 1024:.2f} MB")
    print(f"   Total parameters: {total_params / 1e9:.2f} B")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BitMamba model to binary format.")
    parser.add_argument("--version", type=str, choices=["1b", "250m"], required=True, help="Model version: '1b' or '250m'")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file (msgpack)")
    parser.add_argument("--output_name", type=str, required=True, help="Output binary filename")

    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    output_bin = args.output_name

    if not os.path.exists(ckpt_path):
        print(f"❌ Not found: {ckpt_path}")
        sys.exit(1)

    print(f"🏗️  Loading PyTorch model ({args.version})...")
    
    if args.version == "1b":
        config = CONFIG_1B
    else:
        config = CONFIG_255M
        
    model = BitMambaLM(**config)
    load_checkpoint(model, ckpt_path) 

    export_packed(model, output_bin)
    print(f"\n👉 Done. Use: ./bitmamba {output_bin}")
