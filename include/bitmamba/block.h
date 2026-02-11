#pragma once
#include <vector>
#include "bitmamba/tensor.h"

namespace bitmamba {
    class BitMambaBlock {
    public:
        Tensor in_proj_norm, in_proj, conv1d_w, conv1d_b, dt_bias, A_log, D, out_proj_norm, out_proj;
        std::vector<float> conv_state, ssm_state;
        int d_model, d_inner, n_heads, head_dim, d_conv; 

        void init_cache(int d_m, int heads, int expand=2, int conv_k=4);
        void step(const std::vector<float>& u, std::vector<float>& out_buffer);
    };
}
