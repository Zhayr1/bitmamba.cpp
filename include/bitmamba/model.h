#pragma once
#include <string>
#include <vector>
#include "bitmamba/config.h"
#include "bitmamba/tensor.h"
#include "bitmamba/block.h"  // brings in MambaState and BitMambaBlock
#include "bitmamba/lora.h"

namespace bitmamba {

    class BitMambaModel {
    public:
        Config config;
        Tensor embed, norm_f, lm_head_norm, lm_head;

        // Physical layers — 32 entries for a 1B model, each holding quantized weights.
        std::vector<BitMambaBlock> layers;

        // --- LLM Neuroanatomy / RYS ---
        // execution_path: ordered list of layer indices to execute.
        //   Normal:  {0, 1, 2, ..., 31}  (size == n_layers)
        //   RYS:     {0,..,15, 10,..,15, 16,..,31}  (size == n_layers + repeat_range)
        // layer_states: one independent MambaState per execution_path slot.
        //   Identical layer index → independent state → no state corruption.
        std::vector<int>        execution_path;
        std::vector<MambaState> layer_states;

        // Working buffers
        std::vector<float> current_x, next_x;

        // Optional LoRA adapter. Empty by default; populated by load_lora().
        // Applied to W_in and W_out of every Mamba block at forward time.
        LoraAdapter lora;

        // Constructor.
        //   repeat_start / repeat_end: inclusive layer range to repeat.
        //   Pass -1 (default) for normal inference without repetition.
        BitMambaModel(const std::string& path,
                      int repeat_start = -1,
                      int repeat_end   = -1,
                      int repeat_count =  1);

        void load_from_bin(const std::string& path);

        // Load a LoRA adapter (.lora.bin produced by scripts/export_lora_bin.py).
        // Verifies the adapter's n_layers matches the model and its per-slot
        // shapes match in_proj.rows/cols and out_proj.rows/cols of each block.
        // Aborts with a clear error on mismatch.
        void load_lora(const std::string& path);

        int forward_step(int token,
                         const std::vector<int>& history,
                         float penalty,
                         float temp,
                         float min_p,
                         float top_p,
                         int top_k);

        // Returns {rank (0-indexed, lower=better), log-probability} of
        // target_token after a forward pass on input_token.
        // Runs on current layer_states (call after prefill).
        std::pair<int, float> compute_log_prob(int input_token, int target_token);

        // Resets all layer states (conv_state, ssm_state) to zero.
        // Call before evaluating a new, independent sequence.
        void reset_states();

    private:
        // Builds execution_path and allocates layer_states.
        // Called once after load_from_bin.
        void build_execution_path(int repeat_start, int repeat_end, int repeat_count);
    };

} // namespace bitmamba
