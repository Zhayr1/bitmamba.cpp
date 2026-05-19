#include "bitmamba/model.h"
#include "bitmamba/kernels.h"
#include "bitmamba/quantization.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <immintrin.h>

namespace bitmamba {

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    struct TokenProb { int id; float val; };

    static int sample_advanced(std::vector<float>& logits, float temp,
                               float min_p, float top_p, int top_k) {
        int vocab_size = logits.size();
        if (temp < 0.05f) { // Greedy
            int best = 0; float max_l = -1e9;
            for (int i = 0; i < vocab_size; ++i)
                if (logits[i] > max_l) { max_l = logits[i]; best = i; }
            return best;
        }
        for (int i = 0; i < vocab_size; ++i) logits[i] /= temp;
        float max_logit = -1e9;
        for (float l : logits) if (l > max_logit) max_logit = l;
        double sum_exp = 0.0;
        std::vector<TokenProb> probs(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            probs[i].id = i;
            probs[i].val = expf(logits[i] - max_logit);
            sum_exp += probs[i].val;
        }
        for (int i = 0; i < vocab_size; ++i) probs[i].val /= sum_exp;
        std::sort(probs.begin(), probs.end(),
                  [](const TokenProb& a, const TokenProb& b) { return a.val > b.val; });

        int n_probs = vocab_size;
        if (min_p > 0.0f) {
            float thr = probs[0].val * min_p;
            for (int i = 1; i < n_probs; ++i)
                if (probs[i].val < thr) { n_probs = i; break; }
        }
        if (top_p < 1.0f) {
            double cs = 0.0;
            for (int i = 0; i < n_probs; ++i) {
                cs += probs[i].val;
                if (cs >= top_p) { n_probs = i + 1; break; }
            }
        }
        if (top_k > 0 && top_k < n_probs) n_probs = top_k;

        double new_sum = 0.0;
        for (int i = 0; i < n_probs; ++i) new_sum += probs[i].val;
        float r = ((float)rand() / (float)RAND_MAX) * new_sum;
        double cdf = 0.0;
        for (int i = 0; i < n_probs; ++i) {
            cdf += probs[i].val;
            if (r < cdf) return probs[i].id;
        }
        return probs[n_probs - 1].id;
    }

    // -----------------------------------------------------------------------
    // BitMambaModel
    // -----------------------------------------------------------------------

    BitMambaModel::BitMambaModel(const std::string& path,
                                 int repeat_start,
                                 int repeat_end,
                                 int repeat_count) {
        init_lut();
        load_from_bin(path);
        current_x.resize(config.d_model);
        next_x.resize(config.d_model);
        build_execution_path(repeat_start, repeat_end, repeat_count);
    }

    // -----------------------------------------------------------------------
    // build_execution_path
    //
    // Constructs execution_path (the ordered list of physical layer indices
    // to call during forward_step) and allocates one MambaState per slot.
    //
    // Example — 32 layers, repeat_start=10, repeat_end=15:
    //   Normal path  : 0,1,...,9,10,11,12,13,14,15,16,...,31   (32 steps)
    //   After insert : 0,1,...,9,10,11,12,13,14,15,          <- first pass
    //                           10,11,12,13,14,15,            <- repeated slice
    //                                        16,...,31        (38 steps total)
    //
    // With repeat_start == -1 (default): path == [0..n_layers-1].
    // -----------------------------------------------------------------------
    void BitMambaModel::build_execution_path(int repeat_start, int repeat_end,
                                             int repeat_count) {
        execution_path.clear();
        layer_states.clear();

        // Build the base path: 0 → n_layers-1
        for (int i = 0; i < config.n_layers; ++i)
            execution_path.push_back(i);

        // Insert the repeated slice repeat_count times right after repeat_end
        if (repeat_start >= 0 && repeat_end >= repeat_start
                && repeat_end < config.n_layers && repeat_count > 0) {
            auto insert_pos = execution_path.begin() + repeat_end + 1;
            for (int r = 0; r < repeat_count; ++r)
                for (int i = repeat_start; i <= repeat_end; ++i)
                    insert_pos = execution_path.insert(insert_pos, i) + 1;

            std::cerr << "[RYS] Execution path size: " << execution_path.size()
                      << " steps (layers " << repeat_start << "–"
                      << repeat_end << " ×" << (repeat_count + 1) << ")\n";
        }

        // Allocate one independent MambaState per execution slot.
        layer_states.resize(execution_path.size());
        for (int i = 0; i < (int)execution_path.size(); ++i) {
            int li = execution_path[i];
            layer_states[i].init(layers[li].d_inner, layers[li].d_conv);
        }
    }

    // -----------------------------------------------------------------------
    // load_from_bin
    // -----------------------------------------------------------------------
    void BitMambaModel::load_from_bin(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "❌ Error: Cannot open file.\n";
            exit(1);
        }
        int magic;
        f.read((char*)&magic, sizeof(int));
        if (magic != 0x42495432) {
            std::cerr << "❌ Error: Wrong format (need Packed .bin)\n";
            exit(1);
        }
        f.read((char*)&config.vocab_size, sizeof(int));
        f.read((char*)&config.d_model,    sizeof(int));
        f.read((char*)&config.n_layers,   sizeof(int));
        f.read((char*)&config.n_heads,    sizeof(int));

        auto read_tensor = [&](Tensor& t) {
            int type;
            f.read((char*)&type, sizeof(int));
            if (type == 2) {
                t.is_bitnet = true;
                f.read((char*)&t.rows,  sizeof(int));
                f.read((char*)&t.cols,  sizeof(int));
                f.read((char*)&t.scale, sizeof(float));
                int packed_cols = (t.cols + 3) / 4;
                int total_bytes = t.rows * packed_cols;
                t.packed_data.resize(total_bytes + 64); // padding for SIMD
                f.read((char*)t.packed_data.data(), total_bytes);
            } else {
                t.is_bitnet = false;
                int ndim;
                f.read((char*)&ndim, sizeof(int));
                int total_size = 1;
                t.rows = 0; t.cols = 0;
                for (int i = 0; i < ndim; i++) {
                    int d;
                    f.read((char*)&d, sizeof(int));
                    total_size *= d;
                    if (i == 0) t.rows = d;
                    if (i == 1) t.cols = d;
                }
                t.data.resize(total_size);
                f.read((char*)t.data.data(), total_size * sizeof(float));
            }
        };

        read_tensor(embed);
        layers.resize(config.n_layers);
        for (int i = 0; i < config.n_layers; i++) {
            // init_cache stores dims only; no state allocated here.
            layers[i].init_cache(config.d_model, config.n_heads);
            read_tensor(layers[i].in_proj_norm); read_tensor(layers[i].in_proj);
            read_tensor(layers[i].conv1d_w);     read_tensor(layers[i].conv1d_b);
            read_tensor(layers[i].dt_bias);      read_tensor(layers[i].A_log);
            read_tensor(layers[i].D);
            read_tensor(layers[i].out_proj_norm); read_tensor(layers[i].out_proj);
        }
        read_tensor(norm_f);
        read_tensor(lm_head_norm);
        read_tensor(lm_head);
        f.close();
    }

    // -----------------------------------------------------------------------
    // forward_step — iterates execution_path, each slot uses its own state.
    // -----------------------------------------------------------------------
    int BitMambaModel::forward_step(int token,
                                    const std::vector<int>& history,
                                    float penalty,
                                    float temp,
                                    float min_p,
                                    float top_p,
                                    int top_k) {
        std::memcpy(current_x.data(),
                    &embed.data[token * config.d_model],
                    config.d_model * sizeof(float));

        // Core loop: use execution_path so repeated layer indices are handled
        // transparently — each slot i has its own layer_states[i].
        // LoRA slots are looked up by PHYSICAL layer index so a repeated slice
        // (RYS) reuses the same adapter weights consistently.
        for (int i = 0; i < (int)execution_path.size(); ++i) {
            int li = execution_path[i];
            const LoraSlot* lin  = lora.get(li, LORA_TARGET_IN);
            const LoraSlot* lout = lora.get(li, LORA_TARGET_OUT);
            layers[li].step(current_x, next_x, layer_states[i],
                            lin, lout, lora.scale);
            for (int j = 0; j < config.d_model; ++j)
                current_x[j] += next_x[j];
        }

        std::vector<float> final_feat(config.d_model);
        rms_norm(current_x, norm_f, final_feat);

        std::vector<float> logits(config.vocab_size);
        bitlinear_forward(final_feat, lm_head, lm_head_norm, logits);

        // Repetition penalty
        std::vector<int> protected_tokens = {50256, 198, 628, 13};
        for (int past_token : history) {
            bool is_prot = false;
            for (int p : protected_tokens) if (p == past_token) { is_prot = true; break; }
            if (!is_prot && past_token < config.vocab_size) {
                if (logits[past_token] > 0) logits[past_token] /= penalty;
                else logits[past_token] *= penalty;
            }
        }

        return sample_advanced(logits, temp, min_p, top_p, top_k);
    }

    // -----------------------------------------------------------------------
    // prefill_step — embed + layer loop only. ~8-12% faster than forward_step
    // because it skips the lm_head matmul, repetition penalty, and sampling
    // (all of which produce a token id that prefill throws away anyway).
    // -----------------------------------------------------------------------
    void BitMambaModel::prefill_step(int token) {
        std::memcpy(current_x.data(),
                    &embed.data[token * config.d_model],
                    config.d_model * sizeof(float));

        for (int i = 0; i < (int)execution_path.size(); ++i) {
            int li = execution_path[i];
            const LoraSlot* lin  = lora.get(li, LORA_TARGET_IN);
            const LoraSlot* lout = lora.get(li, LORA_TARGET_OUT);
            layers[li].step(current_x, next_x, layer_states[i],
                            lin, lout, lora.scale);
            for (int j = 0; j < config.d_model; ++j)
                current_x[j] += next_x[j];
        }
    }

    // -----------------------------------------------------------------------
    // prefill_sequence — layer-major batched prefill.
    //
    // For each layer l, runs prefill_block on all T tokens. The block does the
    // in_proj BitLinear in a single batched call (weights read once), then the
    // sequential conv+SSM scan per token, then the out_proj BitLinear batched.
    // After each layer the residual stream is updated in place.
    // -----------------------------------------------------------------------
    void BitMambaModel::prefill_sequence(const std::vector<int>& tokens) {
        const int T = (int)tokens.size();
        if (T == 0) return;

        const int d_model = config.d_model;

        // Ensure the four scratch buffers are at least [T × …] in capacity.
        // First-call (or growth) reallocates; subsequent prefills reuse the
        // same memory so glibc malloc doesn't lock the arena per layer.
        if (T > prefill_capacity_T) {
            int d_in_proj_max = 0, d_inner_max = 0;
            for (const auto& l : layers) {
                if (l.in_proj.rows > d_in_proj_max) d_in_proj_max = l.in_proj.rows;
                if (l.d_inner       > d_inner_max)   d_inner_max  = l.d_inner;
            }
            prefill_X_buf   .assign((size_t)T * d_model,       0.0f);
            prefill_OUT_buf .assign((size_t)T * d_model,       0.0f);
            prefill_proj_buf.assign((size_t)T * d_in_proj_max, 0.0f);
            prefill_y_buf   .assign((size_t)T * d_inner_max,   0.0f);
            prefill_capacity_T = T;
        }
        float* X   = prefill_X_buf.data();
        float* OUT = prefill_OUT_buf.data();

        // Gather embeddings into the [T × d_model] X buffer.
        for (int t = 0; t < T; ++t) {
            std::memcpy(X + (size_t)t * d_model,
                        &embed.data[tokens[t] * d_model],
                        d_model * sizeof(float));
        }

        // Iterate execution_path (so RYS is honored): each slot has its own state.
        for (int i = 0; i < (int)execution_path.size(); ++i) {
            int li = execution_path[i];
            const LoraSlot* lin  = lora.get(li, LORA_TARGET_IN);
            const LoraSlot* lout = lora.get(li, LORA_TARGET_OUT);
            layers[li].prefill_block(X, T, OUT,
                                     prefill_proj_buf.data(),
                                     prefill_y_buf.data(),
                                     layer_states[i],
                                     lin, lout, lora.scale);
            // Residual add: X += OUT  (in-place, parallel across tokens)
            #pragma omp parallel for
            for (int t = 0; t < T; ++t) {
                float* xt = X   + (size_t)t * d_model;
                float* ot = OUT + (size_t)t * d_model;
                int j = 0;
                for (; j <= d_model - 8; j += 8) {
                    __m256 vx = _mm256_loadu_ps(xt + j);
                    __m256 vo = _mm256_loadu_ps(ot + j);
                    _mm256_storeu_ps(xt + j, _mm256_add_ps(vx, vo));
                }
                for (; j < d_model; ++j) xt[j] += ot[j];
            }
        }

        // Keep current_x aligned with the LAST token's hidden state so the
        // following per-token generation call has the correct starting context.
        // (The recurrent state in layer_states is already correct.)
        std::memcpy(current_x.data(),
                    X + (size_t)(T - 1) * d_model,
                    d_model * sizeof(float));
    }

    // -----------------------------------------------------------------------
    // compute_log_prob
    // Runs a forward pass on input_token (using current layer_states),
    // then computes softmax and returns:
    //   {rank (0-indexed), log_prob}  of target_token.
    // A rank of 0 means the target is the argmax prediction.
    // log_prob is always <= 0  (higher = more probable).
    // -----------------------------------------------------------------------
    std::pair<int, float> BitMambaModel::compute_log_prob(int input_token,
                                                          int target_token) {
        // --- Forward pass (same structure as forward_step, no penalty) ---
        std::memcpy(current_x.data(),
                    &embed.data[input_token * config.d_model],
                    config.d_model * sizeof(float));

        for (int i = 0; i < (int)execution_path.size(); ++i) {
            int li = execution_path[i];
            const LoraSlot* lin  = lora.get(li, LORA_TARGET_IN);
            const LoraSlot* lout = lora.get(li, LORA_TARGET_OUT);
            layers[li].step(current_x, next_x, layer_states[i],
                            lin, lout, lora.scale);
            for (int j = 0; j < config.d_model; ++j)
                current_x[j] += next_x[j];
        }

        std::vector<float> final_feat(config.d_model);
        rms_norm(current_x, norm_f, final_feat);

        std::vector<float> logits(config.vocab_size);
        bitlinear_forward(final_feat, lm_head, lm_head_norm, logits);

        // --- Numerically stable softmax ---
        float max_l = *std::max_element(logits.begin(), logits.end());
        double sum_exp = 0.0;
        std::vector<float> probs(config.vocab_size);
        for (int i = 0; i < config.vocab_size; ++i) {
            probs[i] = expf(logits[i] - max_l);
            sum_exp += probs[i];
        }
        for (int i = 0; i < config.vocab_size; ++i) probs[i] /= (float)sum_exp;

        // --- Log-prob and rank of target_token ---
        float target_prob = probs[target_token];
        float log_prob    = logf(target_prob + 1e-10f);  // avoid log(0)

        int rank = 0;
        for (int i = 0; i < config.vocab_size; ++i)
            if (i != target_token && probs[i] > target_prob) ++rank;

        return {rank, log_prob};
    }

    // -----------------------------------------------------------------------
    // load_lora — load a .lora.bin file and validate shapes against this model.
    // -----------------------------------------------------------------------
    void BitMambaModel::load_lora(const std::string& path) {
        load_lora_bin(path, lora);

        if (lora.n_layers != config.n_layers) {
            std::cerr << "❌ Error: LoRA n_layers=" << lora.n_layers
                      << " does not match model n_layers=" << config.n_layers << "\n";
            std::exit(1);
        }
        // Verify per-slot shapes match base projection dimensions.
        for (int li = 0; li < config.n_layers; ++li) {
            const LoraSlot& sin  = lora.slots[li][LORA_TARGET_IN];
            const LoraSlot& sout = lora.slots[li][LORA_TARGET_OUT];
            if (sin.out_features != layers[li].in_proj.rows ||
                sin.in_features  != layers[li].in_proj.cols) {
                std::cerr << "❌ Error: LoRA W_in shape mismatch at layer " << li
                          << " (lora=" << sin.out_features << "x" << sin.in_features
                          << ", base=" << layers[li].in_proj.rows << "x"
                          << layers[li].in_proj.cols << ")\n";
                std::exit(1);
            }
            if (sout.out_features != layers[li].out_proj.rows ||
                sout.in_features  != layers[li].out_proj.cols) {
                std::cerr << "❌ Error: LoRA W_out shape mismatch at layer " << li
                          << " (lora=" << sout.out_features << "x" << sout.in_features
                          << ", base=" << layers[li].out_proj.rows << "x"
                          << layers[li].out_proj.cols << ")\n";
                std::exit(1);
            }
        }
        std::cerr << "[LoRA] Shape check passed for " << config.n_layers
                  << " layers (W_in + W_out)\n";
    }

    // -----------------------------------------------------------------------
    // reset_states — zeroes all conv_state and ssm_state buffers.
    // -----------------------------------------------------------------------
    void BitMambaModel::reset_states() {
        for (auto& s : layer_states) {
            std::fill(s.conv_state.begin(), s.conv_state.end(), 0.0f);
            std::fill(s.ssm_state.begin(), s.ssm_state.end(), 0.0f);
        }
    }

} // namespace bitmamba
