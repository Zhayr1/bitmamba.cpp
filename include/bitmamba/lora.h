#pragma once
#include <array>
#include <string>
#include <vector>

namespace bitmamba {

    // ---------------------------------------------------------------------------
    // LoRA target convention — fixed for the BitMamba-2 recipe.
    //   slot index 0  ↔  W_in    (in-projection BitLinear of each Mamba block)
    //   slot index 1  ↔  W_out   (out-projection BitLinear of each Mamba block)
    // ---------------------------------------------------------------------------
    enum LoraTarget {
        LORA_TARGET_IN  = 0,
        LORA_TARGET_OUT = 1,
        LORA_N_TARGETS  = 2
    };

    // A single LoRA (A, B) pair attached to one base projection.
    //   A is [rank, in_features],  row-major
    //   B is [out_features, rank], row-major
    // The base projection has the same out_features × in_features shape.
    struct LoraSlot {
        int rank          = 0;
        int in_features   = 0;
        int out_features  = 0;
        std::vector<float> A;     // [rank * in_features]
        std::vector<float> B;     // [out_features * rank]

        bool empty() const { return rank == 0; }
    };

    // Whole LoRA adapter: one slot per (layer, target).
    // slots[layer_idx][target_idx] gives the LoraSlot for that projection.
    struct LoraAdapter {
        int   rank      = 0;
        float alpha     = 0.0f;
        float scale     = 0.0f;     // alpha / rank
        int   n_layers  = 0;
        std::vector<std::array<LoraSlot, LORA_N_TARGETS>> slots;

        bool empty() const { return n_layers == 0; }

        const LoraSlot* get(int layer_idx, LoraTarget t) const {
            if (empty()) return nullptr;
            if (layer_idx < 0 || layer_idx >= n_layers) return nullptr;
            const LoraSlot& s = slots[layer_idx][t];
            return s.empty() ? nullptr : &s;
        }
    };

    // Load a .lora.bin file (produced by scripts/export_lora_bin.py).
    // On error, prints to stderr and exits(1).
    void load_lora_bin(const std::string& path, LoraAdapter& adapter);

    // Add scale · (B · A · x_norm) to `out`.
    //   x_norm:  pointer to in_features floats (pre-RMSNormed input)
    //   slot:    the LoRA pair (A, B, rank)
    //   scale:   adapter.scale (alpha / rank)
    //   out:     pointer to out_features floats (already holds base matmul result)
    void apply_lora_delta(const float* x_norm,
                          const LoraSlot& slot,
                          float scale,
                          float* out);

    // Batched LoRA delta: applies scale·(B·A·X_norm[t]) to OUT[t] for all t in [0,T).
    //   X_norm: [T × in_features]   row-major
    //   OUT:    [T × out_features]  row-major (must already hold base matmul result)
    void apply_lora_delta_batched(const float* X_norm,
                                  int T,
                                  const LoraSlot& slot,
                                  float scale,
                                  float* OUT);

} // namespace bitmamba
