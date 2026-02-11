#pragma once
#include <string>
#include <vector>
#include "bitmamba/config.h"
#include "bitmamba/tensor.h"
#include "bitmamba/block.h"

namespace bitmamba {
    class BitMambaModel {
    public:
        Config config;
        Tensor embed, norm_f, lm_head_norm, lm_head;
        std::vector<BitMambaBlock> layers;
        std::vector<float> current_x, next_x;

        BitMambaModel(const std::string& path);
        void load_from_bin(const std::string& path);
        int forward_step(int token, const std::vector<int>& history, float penalty, 
                         float temp, float min_p, float top_p, int top_k);
    };
}
