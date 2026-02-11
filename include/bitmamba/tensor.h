#pragma once
#include <vector>
#include <cstdint>

namespace bitmamba {
    struct Tensor {
        std::vector<float> data;
        std::vector<uint8_t> packed_data; 
        int rows, cols;
        bool is_bitnet;
        float scale;
        Tensor() : rows(0), cols(0), is_bitnet(false), scale(1.0f) {}
    };
}
