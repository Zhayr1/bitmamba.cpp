#pragma once
#include <vector>
#include "bitmamba/tensor.h"

namespace bitmamba {

    // ---------------------------------------------------------------------------
    // MambaState — runtime recurrent memory, separated from weights.
    //
    // A single BitMambaBlock (holding quantized weights) can be executed with
    // multiple independent MambaState objects, enabling virtual layer repetition
    // (LLM Neuroanatomy / RYS technique) at zero extra weight memory cost.
    // ---------------------------------------------------------------------------
    struct MambaState {
        std::vector<float> conv_state; // size: d_inner * (d_conv - 1)
        std::vector<float> ssm_state;  // size: d_inner

        // Allocate and zero-initialize both state buffers.
        void init(int d_inner, int d_conv) {
            conv_state.assign(d_inner * (d_conv - 1), 0.0f);
            ssm_state.assign(d_inner, 0.0f);
        }
    };

    // ---------------------------------------------------------------------------
    // BitMambaBlock — weights only (no runtime state).
    // ---------------------------------------------------------------------------
    class BitMambaBlock {
    public:
        Tensor in_proj_norm, in_proj, conv1d_w, conv1d_b, dt_bias, A_log, D, out_proj_norm, out_proj;
        int d_model, d_inner, n_heads, head_dim, d_conv;

        // Store dimensional info so callers can size MambaState correctly.
        // Does NOT allocate any state buffers.
        void init_cache(int d_m, int heads, int expand = 2, int conv_k = 4);

        // Execute one autoregressive step.
        // `state` holds the recurrent memory for this specific execution slot.
        void step(const std::vector<float>& u,
                  std::vector<float>& out_buffer,
                  MambaState& state);
    };

} // namespace bitmamba
