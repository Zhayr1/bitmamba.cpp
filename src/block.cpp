#include "bitmamba/block.h"
#include "bitmamba/kernels.h"
#include <cmath>
#include <omp.h>

namespace bitmamba {

    void BitMambaBlock::init_cache(int d_m, int heads, int expand, int conv_k) {
        // Store dimensional info only — state allocation is handled by MambaState::init().
        d_model  = d_m;
        n_heads  = heads;
        d_inner  = d_model * expand;
        head_dim = d_inner / n_heads;
        d_conv   = conv_k;
    }

    // Execute one autoregressive step using the provided recurrent state.
    // The same BitMambaBlock (weights) can be called with different `state`
    // objects to implement virtual layer repetition without extra weight memory.
    void BitMambaBlock::step(const std::vector<float>& u,
                             std::vector<float>& out_buffer,
                             MambaState& state,
                             const LoraSlot* lora_in_proj,
                             const LoraSlot* lora_out_proj,
                             float lora_scale) {
        std::vector<float> proj_out(in_proj.rows);
        bitlinear_forward(u, in_proj, in_proj_norm, proj_out,
                          lora_in_proj, lora_scale);

        float* ptr_z  = proj_out.data();
        float* ptr_x  = proj_out.data() + d_inner;
        float* ptr_B  = proj_out.data() + 2 * d_inner;
        float* ptr_C  = proj_out.data() + 2 * d_inner + n_heads;
        float* ptr_dt = proj_out.data() + 2 * d_inner + 2 * n_heads;

        std::vector<float> y(d_inner);

        #pragma omp parallel for
        for (int h = 0; h < n_heads; ++h) {
            float dt_val = softplus(ptr_dt[h] + dt_bias.data[h]);
            float A_val  = -expf(A_log.data[h]);
            float decay  = expf(A_val * dt_val);
            float B_val  = ptr_B[h];
            float C_val  = ptr_C[h];

            for (int d = 0; d < head_dim; ++d) {
                int idx = h * head_dim + d;

                // --- Convolution step (uses state.conv_state) ---
                float input_x  = ptr_x[idx];
                float conv_res = conv1d_b.data[idx];
                for (int k = 0; k < d_conv - 1; ++k)
                    conv_res += state.conv_state[idx * (d_conv - 1) + k]
                                * conv1d_w.data[idx * d_conv + k];
                conv_res += input_x * conv1d_w.data[idx * d_conv + (d_conv - 1)];

                // Shift conv state buffer
                for (int k = 0; k < d_conv - 2; ++k)
                    state.conv_state[idx * (d_conv - 1) + k] =
                        state.conv_state[idx * (d_conv - 1) + k + 1];
                state.conv_state[idx * (d_conv - 1) + (d_conv - 2)] = input_x;

                // --- SSM step (uses state.ssm_state) ---
                float x_act  = silu(conv_res);
                float u_ssm  = x_act * B_val * dt_val;
                state.ssm_state[idx] = state.ssm_state[idx] * decay + u_ssm;

                y[idx] = (state.ssm_state[idx] * C_val
                          + x_act * D.data[idx]) * silu(ptr_z[idx]);
            }
        }

        bitlinear_forward(y, out_proj, out_proj_norm, out_buffer,
                          lora_out_proj, lora_scale);
    }

} // namespace bitmamba
