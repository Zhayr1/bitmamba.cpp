#pragma once
#include <vector>
#include <cmath>
#include "bitmamba/tensor.h"
#include "bitmamba/lora.h"

namespace bitmamba {
    inline float silu(float x) { return x / (1.0f + expf(-x)); }
    inline float softplus(float x) { return logf(1.0f + expf(x)); }

    void rms_norm(const std::vector<float>& x, const Tensor& weight, std::vector<float>& out);

    // BitLinear matmul. If `lora` is non-null, also adds lora_scale · (B · A · x_norm)
    // to the output, where x_norm is the pre-quantization RMSNormed input (re-used
    // so RMSNorm is computed once per call).
    void bitlinear_forward(const std::vector<float>& x,
                           const Tensor& w,
                           const Tensor& norm_w,
                           std::vector<float>& out,
                           const LoraSlot* lora = nullptr,
                           float lora_scale = 0.0f);
}
