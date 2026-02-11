#pragma once
#include <vector>
#include <cmath>
#include "bitmamba/tensor.h"

namespace bitmamba {
    inline float silu(float x) { return x / (1.0f + expf(-x)); }
    inline float softplus(float x) { return logf(1.0f + expf(x)); }

    void rms_norm(const std::vector<float>& x, const Tensor& weight, std::vector<float>& out);
    void bitlinear_forward(const std::vector<float>& x, const Tensor& w, const Tensor& norm_w, std::vector<float>& out);
}
