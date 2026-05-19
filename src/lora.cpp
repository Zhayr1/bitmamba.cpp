#include "bitmamba/lora.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <immintrin.h>

namespace bitmamba {

    static constexpr uint32_t MAGIC_LOR2  = 0x4C4F5232; // "LOR2" little-endian
    static constexpr uint32_t LORA_VERSION = 1;

    // ---------------------------------------------------------------------------
    // load_lora_bin
    // Reads the binary format produced by scripts/export_lora_bin.py.
    // ---------------------------------------------------------------------------
    void load_lora_bin(const std::string& path, LoraAdapter& adapter) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "❌ Error: cannot open LoRA file: " << path << "\n";
            std::exit(1);
        }

        uint32_t magic, version, rank_u, n_layers_u, n_targets_u;
        float alpha;
        f.read((char*)&magic,       sizeof(uint32_t));
        f.read((char*)&version,     sizeof(uint32_t));
        f.read((char*)&rank_u,      sizeof(uint32_t));
        f.read((char*)&alpha,       sizeof(float));
        f.read((char*)&n_layers_u,  sizeof(uint32_t));
        f.read((char*)&n_targets_u, sizeof(uint32_t));

        if (magic != MAGIC_LOR2) {
            std::cerr << "❌ Error: bad LoRA magic 0x" << std::hex << magic
                      << " (expected 0x" << MAGIC_LOR2 << ")\n";
            std::exit(1);
        }
        if (version != LORA_VERSION) {
            std::cerr << "❌ Error: unsupported LoRA version " << version << "\n";
            std::exit(1);
        }
        if (n_targets_u != LORA_N_TARGETS) {
            std::cerr << "❌ Error: LoRA file has n_targets=" << n_targets_u
                      << " but this build expects " << LORA_N_TARGETS << " (W_in, W_out)\n";
            std::exit(1);
        }

        adapter.rank     = (int)rank_u;
        adapter.alpha    = alpha;
        adapter.scale    = alpha / (float)rank_u;
        adapter.n_layers = (int)n_layers_u;
        adapter.slots.resize(adapter.n_layers);

        for (int li = 0; li < adapter.n_layers; ++li) {
            for (int t = 0; t < LORA_N_TARGETS; ++t) {
                uint32_t layer_idx, target_idx, out_features, in_features, rank_check;
                f.read((char*)&layer_idx,    sizeof(uint32_t));
                f.read((char*)&target_idx,   sizeof(uint32_t));
                f.read((char*)&out_features, sizeof(uint32_t));
                f.read((char*)&in_features,  sizeof(uint32_t));
                f.read((char*)&rank_check,   sizeof(uint32_t));

                if ((int)layer_idx != li || (int)target_idx != t) {
                    std::cerr << "❌ Error: LoRA slot ordering mismatch at layer "
                              << li << " target " << t << " (got layer=" << layer_idx
                              << " target=" << target_idx << ")\n";
                    std::exit(1);
                }
                if ((int)rank_check != adapter.rank) {
                    std::cerr << "❌ Error: LoRA rank mismatch at slot " << li << "/" << t
                              << " (got " << rank_check << ", header says " << adapter.rank << ")\n";
                    std::exit(1);
                }

                LoraSlot& slot = adapter.slots[li][t];
                slot.rank         = adapter.rank;
                slot.in_features  = (int)in_features;
                slot.out_features = (int)out_features;
                slot.A.resize((size_t)slot.rank * slot.in_features);
                slot.B.resize((size_t)slot.out_features * slot.rank);
                f.read((char*)slot.A.data(), slot.A.size() * sizeof(float));
                f.read((char*)slot.B.data(), slot.B.size() * sizeof(float));
            }
        }

        if (!f.good()) {
            std::cerr << "❌ Error: LoRA file truncated or read failure\n";
            std::exit(1);
        }
        f.close();

        std::cerr << "[LoRA] Loaded " << adapter.n_layers << " layers × "
                  << LORA_N_TARGETS << " targets, rank=" << adapter.rank
                  << ", alpha=" << adapter.alpha
                  << ", scale=" << adapter.scale << "\n";
    }

    // ---------------------------------------------------------------------------
    // apply_lora_delta
    //   delta = scale · (B · A · x_norm)
    //   out  += delta
    // A is [rank, in_features]  row-major
    // B is [out_features, rank] row-major
    // Two FP32 mat-vec products, both with `rank` (typically 16) as one dim.
    // ---------------------------------------------------------------------------
    void apply_lora_delta(const float* x_norm,
                          const LoraSlot& slot,
                          float scale,
                          float* out) {
        const int rank         = slot.rank;
        const int in_features  = slot.in_features;
        const int out_features = slot.out_features;
        const float* Adata = slot.A.data();
        const float* Bdata = slot.B.data();

        // Step 1: a_proj = A · x_norm   shape [rank]
        // For each row r, dot product of A[r, :] and x_norm[:].
        std::vector<float> a_proj((size_t)rank, 0.0f);
        for (int r = 0; r < rank; ++r) {
            const float* row = Adata + (size_t)r * in_features;
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i <= in_features - 8; i += 8) {
                __m256 va = _mm256_loadu_ps(row + i);
                __m256 vx = _mm256_loadu_ps(x_norm + i);
                acc = _mm256_fmadd_ps(va, vx, acc);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
            for (; i < in_features; ++i) s += row[i] * x_norm[i];
            a_proj[r] = s;
        }

        // Step 2: out[o] += scale · (B[o, :] · a_proj)
        //   B[o, :] is contiguous (row-major), length = rank.
        // Parallelize over output rows.
        #pragma omp parallel for
        for (int o = 0; o < out_features; ++o) {
            const float* row = Bdata + (size_t)o * rank;
            float s = 0.0f;
            // Typical rank=16: small loop, often fully unrolled by the compiler.
            for (int r = 0; r < rank; ++r) s += row[r] * a_proj[r];
            out[o] += scale * s;
        }
    }

    // ---------------------------------------------------------------------------
    // apply_lora_delta_batched
    // Batched version of apply_lora_delta. Reads X_norm [T × in_features] and
    // adds scale·(B·A·X_norm[t]) to OUT[t] for every t.
    //
    // Implementation strategy:
    //   A_proj[T × rank]    = X_norm[T × in] · A^T[in × rank]
    //   OUT[T × out]       += scale · A_proj[T × rank] · B^T[rank × out]
    //
    // A is small (rank * in_features ≤ 128KB for the 1B model) so it stays hot
    // in L1/L2 across all T iterations of the inner loop. The B matmul is the
    // bigger one but each row is contiguous in memory (length = rank) which is
    // friendly to SIMD.
    // ---------------------------------------------------------------------------
    void apply_lora_delta_batched(const float* X_norm,
                                  int T,
                                  const LoraSlot& slot,
                                  float scale,
                                  float* OUT) {
        const int rank         = slot.rank;
        const int in_features  = slot.in_features;
        const int out_features = slot.out_features;
        const float* Adata = slot.A.data();
        const float* Bdata = slot.B.data();

        // Stage 1: A_proj = X_norm · A^T  → [T × rank]
        std::vector<float> A_proj((size_t)T * rank, 0.0f);
        #pragma omp parallel for collapse(2)
        for (int t = 0; t < T; ++t) {
            for (int r = 0; r < rank; ++r) {
                const float* row    = Adata + (size_t)r * in_features;   // A[r, :]
                const float* xn_t   = X_norm + (size_t)t * in_features;  // X_norm[t, :]
                __m256 acc = _mm256_setzero_ps();
                int i = 0;
                for (; i <= in_features - 8; i += 8) {
                    __m256 va = _mm256_loadu_ps(row + i);
                    __m256 vx = _mm256_loadu_ps(xn_t + i);
                    acc = _mm256_fmadd_ps(va, vx, acc);
                }
                float tmp[8]; _mm256_storeu_ps(tmp, acc);
                float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                for (; i < in_features; ++i) s += row[i] * xn_t[i];
                A_proj[(size_t)t * rank + r] = s;
            }
        }

        // Stage 2: OUT += scale · A_proj · B^T  → [T × out_features]
        // For each output row o, B[o, :] is rank floats (contiguous).
        // Loop order: parallel over (t, o) — independent writes, no contention.
        #pragma omp parallel for collapse(2)
        for (int t = 0; t < T; ++t) {
            for (int o = 0; o < out_features; ++o) {
                const float* b_row  = Bdata + (size_t)o * rank;
                const float* ap_row = A_proj.data() + (size_t)t * rank;
                float s = 0.0f;
                for (int r = 0; r < rank; ++r) s += b_row[r] * ap_row[r];
                OUT[(size_t)t * out_features + o] += scale * s;
            }
        }
    }

} // namespace bitmamba
