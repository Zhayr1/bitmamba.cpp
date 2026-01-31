#include "bitmamba/model.h"
#include "bitmamba/kernels.h"
#include "bitmamba/quantization.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

namespace bitmamba {

    // Helper structs for sampler
    struct TokenProb { int id; float val; };

    // Private sampler function
    static int sample_advanced(std::vector<float>& logits, float temp, float min_p, float top_p, int top_k) {
        int vocab_size = logits.size();
        if (temp < 0.05f) { // Greedy
            int best = 0; float max_l = -1e9;
            for(int i=0; i<vocab_size; ++i) if(logits[i] > max_l) { max_l = logits[i]; best = i; }
            return best;
        }
        for (int i = 0; i < vocab_size; ++i) logits[i] /= temp;
        float max_logit = -1e9; for (float l : logits) if (l > max_logit) max_logit = l;
        double sum_exp = 0.0;
        std::vector<TokenProb> probs(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            probs[i].id = i; probs[i].val = expf(logits[i] - max_logit); sum_exp += probs[i].val;
        }
        for (int i = 0; i < vocab_size; ++i) probs[i].val /= sum_exp;
        std::sort(probs.begin(), probs.end(), [](const TokenProb& a, const TokenProb& b){ return a.val > b.val; });

        int n_probs = vocab_size;
        if (min_p > 0.0f) {
            float thr = probs[0].val * min_p;
            for (int i = 1; i < n_probs; ++i) if (probs[i].val < thr) { n_probs = i; break; }
        }
        if (top_p < 1.0f) {
            double cs = 0.0;
            for (int i = 0; i < n_probs; ++i) { cs += probs[i].val; if (cs >= top_p) { n_probs = i + 1; break; } }
        }
        if (top_k > 0 && top_k < n_probs) n_probs = top_k;

        double new_sum = 0.0; for(int i=0; i<n_probs; ++i) new_sum += probs[i].val;
        float r = ((float)rand() / (float)RAND_MAX) * new_sum;
        double cdf = 0.0;
        for (int i = 0; i < n_probs; ++i) { cdf += probs[i].val; if (r < cdf) return probs[i].id; }
        return probs[n_probs - 1].id;
    }

    BitMambaModel::BitMambaModel(const std::string& path) {
        init_lut(); 
        load_from_bin(path);
        current_x.resize(config.d_model);
        next_x.resize(config.d_model);
    }
    
    void BitMambaModel::load_from_bin(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) { std::cerr << "❌ Error: Cannot open file.\n"; exit(1); }
        int magic; f.read((char*)&magic, sizeof(int));
        if (magic != 0x42495432) { std::cerr << "❌ Error: Wrong format (need Packed .bin)\n"; exit(1); }
        f.read((char*)&config.vocab_size, sizeof(int));
        f.read((char*)&config.d_model, sizeof(int));
        f.read((char*)&config.n_layers, sizeof(int));
        f.read((char*)&config.n_heads, sizeof(int));
        
        auto read_tensor = [&](Tensor& t) {
            int type; f.read((char*)&type, sizeof(int));
            if (type == 2) { 
                t.is_bitnet = true; 
                f.read((char*)&t.rows, sizeof(int)); f.read((char*)&t.cols, sizeof(int)); f.read((char*)&t.scale, sizeof(float));
                int packed_cols = (t.cols + 3) / 4;
                int total_bytes = t.rows * packed_cols;
                t.packed_data.resize(total_bytes + 64); // PADDING
                f.read((char*)t.packed_data.data(), total_bytes);
            } else { 
                t.is_bitnet = false; int ndim; f.read((char*)&ndim, sizeof(int));
                int total_size = 1; t.rows=0; t.cols=0;
                for(int i=0; i<ndim; i++) { int d; f.read((char*)&d, sizeof(int)); total_size*=d; if(i==0)t.rows=d; if(i==1)t.cols=d; }
                t.data.resize(total_size); f.read((char*)t.data.data(), total_size * sizeof(float));
            }
        };
        read_tensor(embed);
        layers.resize(config.n_layers);
        for(int i=0; i<config.n_layers; i++) {
            layers[i].init_cache(config.d_model, config.n_heads);
            read_tensor(layers[i].in_proj_norm); read_tensor(layers[i].in_proj);
            read_tensor(layers[i].conv1d_w); read_tensor(layers[i].conv1d_b);
            read_tensor(layers[i].dt_bias); read_tensor(layers[i].A_log); read_tensor(layers[i].D);
            read_tensor(layers[i].out_proj_norm); read_tensor(layers[i].out_proj);
        }
        read_tensor(norm_f); read_tensor(lm_head_norm); read_tensor(lm_head);
        f.close();
    }

    int BitMambaModel::forward_step(int token, const std::vector<int>& history, float penalty, 
                     float temp, float min_p, float top_p, int top_k) {
        std::memcpy(current_x.data(), &embed.data[token * config.d_model], config.d_model * sizeof(float));
        for (int i = 0; i < config.n_layers; ++i) {
            layers[i].step(current_x, next_x);
            for(int j=0; j<config.d_model; ++j) current_x[j] += next_x[j];
        }
        std::vector<float> final_feat(config.d_model);
        rms_norm(current_x, norm_f, final_feat);
        std::vector<float> logits(config.vocab_size);
        bitlinear_forward(final_feat, lm_head, lm_head_norm, logits);

        std::vector<int> protected_tokens = {50256, 198, 628, 13}; 
        for (int past_token : history) {
            bool is_prot = false; for(int p : protected_tokens) if(p==past_token) {is_prot=true; break;}
            if (!is_prot && past_token < config.vocab_size) {
                if (logits[past_token] > 0) logits[past_token] /= penalty; else logits[past_token] *= penalty;
            }
        }
        return sample_advanced(logits, temp, min_p, top_p, top_k);
    }

}
