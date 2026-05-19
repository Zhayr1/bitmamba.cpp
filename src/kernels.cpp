#include "bitmamba/kernels.h"
#include "bitmamba/quantization.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace bitmamba {

    void rms_norm(const std::vector<float>& x, const Tensor& weight, std::vector<float>& out) {
        float sum_sq = 0.0f;
        int size = x.size();
        
        __m256 sum_vec = _mm256_setzero_ps();
        int i = 0;
        for (; i <= size - 8; i += 8) {
            __m256 v = _mm256_loadu_ps(&x[i]);
            sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
        }
        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        for(int k=0; k<8; ++k) sum_sq += temp[k];
        for (; i < size; ++i) sum_sq += x[i] * x[i];

        float rms = 1.0f / sqrtf(sum_sq / size + 1e-6f);
        
        __m256 rms_vec = _mm256_set1_ps(rms);
        for (i = 0; i <= size - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(&x[i]);
            __m256 vw = _mm256_loadu_ps(&weight.data[i]);
            _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_mul_ps(vx, rms_vec), vw));
        }
        for (; i < size; ++i) out[i] = x[i] * rms * weight.data[i];
    }

    void bitlinear_forward(const std::vector<float>& x,
                           const Tensor& w,
                           const Tensor& norm_w,
                           std::vector<float>& out,
                           const LoraSlot* lora,
                           float lora_scale) {
        int n = x.size();
        std::vector<float> x_norm(n);
        rms_norm(x, norm_w, x_norm);

        float max_abs = 0.0f;
        for (float v : x_norm) max_abs = std::max(max_abs, std::abs(v));
        float scale_x = 127.0f / (max_abs + 1e-5f);
        
        std::vector<int8_t> x_quant(n + 32, 0); 
        
        __m256 scale_v = _mm256_set1_ps(scale_x);
        __m256 min_v = _mm256_set1_ps(-128.0f);
        __m256 max_v = _mm256_set1_ps(127.0f);
        
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 v = _mm256_loadu_ps(&x_norm[i]);
            v = _mm256_mul_ps(v, scale_v);
            v = _mm256_max_ps(min_v, _mm256_min_ps(v, max_v)); 
            float buf[8]; _mm256_storeu_ps(buf, v);
            for(int k=0; k<8; ++k) x_quant[i+k] = (int8_t)roundf(buf[k]);
        }
        for (; i < n; ++i) {
            float val = x_norm[i] * scale_x;
            x_quant[i] = (int8_t)(val > 127.f ? 127.f : (val < -128.f ? -128.f : val)); 
        }

        __m256i ones_16 = _mm256_set1_epi16(1);
        
        int cols = w.cols;
        int packed_stride = (cols + 3) / 4; 
        const uint8_t* packed_ptr = w.packed_data.data();

        #pragma omp parallel for
        for (int r = 0; r < w.rows; ++r) {
            __m256i acc_vec = _mm256_setzero_si256();
            int row_offset = r * packed_stride;
            
            int c = 0;
            for (; c <= cols - 32; c += 32) {
                // LUT OPTIMIZATION:
                // Read 8 packed bytes (32 weights)
                const uint8_t* p = packed_ptr + row_offset + c/4;
                
                // Unpack using Table (much faster than bit shifts)
                alignas(32) int8_t w_temp[32];
                uint32_t* w_ptr32 = (uint32_t*)w_temp;
                
                // Manual unroll for speed
                w_ptr32[0] = UNPACK_LUT[p[0]];
                w_ptr32[1] = UNPACK_LUT[p[1]];
                w_ptr32[2] = UNPACK_LUT[p[2]];
                w_ptr32[3] = UNPACK_LUT[p[3]];
                w_ptr32[4] = UNPACK_LUT[p[4]];
                w_ptr32[5] = UNPACK_LUT[p[5]];
                w_ptr32[6] = UNPACK_LUT[p[6]];
                w_ptr32[7] = UNPACK_LUT[p[7]];
                
                __m256i w_vec = _mm256_load_si256((__m256i*)w_temp);
                __m256i x_vec = _mm256_loadu_si256((__m256i*)&x_quant[c]);
                
                __m256i prod = _mm256_sign_epi8(x_vec, w_vec);
                
                __m256i prod_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(prod));
                __m256i prod_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(prod, 1));
                
                acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(prod_lo, ones_16));
                acc_vec = _mm256_add_epi32(acc_vec, _mm256_madd_epi16(prod_hi, ones_16));
            }
            
            int32_t temp_acc[8];
            _mm256_storeu_si256((__m256i*)temp_acc, acc_vec);
            int32_t total = 0;
            for(int k=0; k<8; ++k) total += temp_acc[k];
            
            for (; c < cols; ++c) {
                int byte_idx = c / 4;
                int bit_shift = (c % 4) * 2;
                int8_t w_val = ((packed_ptr[row_offset + byte_idx] >> bit_shift) & 0x03) - 1;
                if (w_val) total += (w_val == 1) ? x_quant[c] : -x_quant[c];
            }
            
            out[r] = (float)total / (scale_x * w.scale);
        }

        // Optional LoRA delta: out += lora_scale · (B · A · x_norm)
        // Uses the same pre-quantization x_norm that fed the bitlinear above.
        if (lora) {
            apply_lora_delta(x_norm.data(), *lora, lora_scale, out.data());
        }
    }

    // ---------------------------------------------------------------------------
    // bitlinear_forward_batched
    // Same math as bitlinear_forward, but processes T tokens with weights read
    // exactly once. The outer loop is the weight row r; the inner loop is the
    // token t. For each (r, c-chunk) pair the LUT-unpacked weight vector is
    // reused across all T tokens — that's where the speedup comes from.
    // ---------------------------------------------------------------------------
    void bitlinear_forward_batched(const float* X,
                                   int T,
                                   const Tensor& w,
                                   const Tensor& norm_w,
                                   float* OUT,
                                   const LoraSlot* lora,
                                   float lora_scale) {
        const int n      = w.cols;
        const int n_rows = w.rows;

        // Per-token: RMSNorm + max-abs scale + int8 quantization. Keep X_norm too
        // for an optional LoRA pass at the end.
        std::vector<float>  X_norm((size_t)T * n);
        std::vector<int8_t> X_quant((size_t)T * (n + 32), 0);   // +32 SIMD tail pad
        std::vector<float>  scale_x_per_t(T);

        for (int t = 0; t < T; ++t) {
            const float* xt = X + (size_t)t * n;
            float* xn = X_norm.data() + (size_t)t * n;

            // RMSNorm: reuse the same scalar/AVX2 pattern as rms_norm() but inlined
            // here to avoid wrapping std::vector for each token.
            float sum_sq = 0.0f;
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i <= n - 8; i += 8) {
                __m256 v = _mm256_loadu_ps(xt + i);
                acc = _mm256_fmadd_ps(v, v, acc);
            }
            float tmp[8]; _mm256_storeu_ps(tmp, acc);
            for (int k = 0; k < 8; ++k) sum_sq += tmp[k];
            for (; i < n; ++i) sum_sq += xt[i] * xt[i];

            float rms = 1.0f / sqrtf(sum_sq / n + 1e-6f);
            __m256 rms_v = _mm256_set1_ps(rms);
            i = 0;
            for (; i <= n - 8; i += 8) {
                __m256 vx = _mm256_loadu_ps(xt + i);
                __m256 vw = _mm256_loadu_ps(&norm_w.data[i]);
                _mm256_storeu_ps(xn + i, _mm256_mul_ps(_mm256_mul_ps(vx, rms_v), vw));
            }
            for (; i < n; ++i) xn[i] = xt[i] * rms * norm_w.data[i];

            // max-abs for int8 scale
            float max_abs = 0.0f;
            for (int j = 0; j < n; ++j) max_abs = std::max(max_abs, std::fabs(xn[j]));
            float sx = 127.0f / (max_abs + 1e-5f);
            scale_x_per_t[t] = sx;

            // Quantize
            int8_t* xq = X_quant.data() + (size_t)t * (n + 32);
            __m256 sv = _mm256_set1_ps(sx);
            __m256 mn = _mm256_set1_ps(-128.0f);
            __m256 mx = _mm256_set1_ps(127.0f);
            i = 0;
            for (; i <= n - 8; i += 8) {
                __m256 v = _mm256_loadu_ps(xn + i);
                v = _mm256_mul_ps(v, sv);
                v = _mm256_max_ps(mn, _mm256_min_ps(v, mx));
                float buf[8]; _mm256_storeu_ps(buf, v);
                for (int k = 0; k < 8; ++k) xq[i + k] = (int8_t)roundf(buf[k]);
            }
            for (; i < n; ++i) {
                float val = xn[i] * sx;
                xq[i] = (int8_t)(val > 127.f ? 127.f : (val < -128.f ? -128.f : val));
            }
        }

        const __m256i ones_16    = _mm256_set1_epi16(1);
        const int     packed_stride = (n + 3) / 4;
        const uint8_t* packed_ptr   = w.packed_data.data();
        const int     row_pad       = n + 32;

        // Outer loop: one weight row at a time, parallel across threads.
        // Inner loop over tokens reuses the unpacked weight vectors.
        #pragma omp parallel for
        for (int r = 0; r < n_rows; ++r) {
            const int row_offset = r * packed_stride;

            // Accumulator per token. Up to 8 vectorized lanes each; reduced at the end.
            // For small T this lives in OMP-thread-local stack.
            std::vector<__m256i> acc((size_t)T, _mm256_setzero_si256());

            int c = 0;
            for (; c <= n - 32; c += 32) {
                const uint8_t* p = packed_ptr + row_offset + c / 4;
                alignas(32) int8_t w_temp[32];
                uint32_t* w_ptr32 = (uint32_t*)w_temp;
                w_ptr32[0] = UNPACK_LUT[p[0]];
                w_ptr32[1] = UNPACK_LUT[p[1]];
                w_ptr32[2] = UNPACK_LUT[p[2]];
                w_ptr32[3] = UNPACK_LUT[p[3]];
                w_ptr32[4] = UNPACK_LUT[p[4]];
                w_ptr32[5] = UNPACK_LUT[p[5]];
                w_ptr32[6] = UNPACK_LUT[p[6]];
                w_ptr32[7] = UNPACK_LUT[p[7]];
                __m256i w_vec = _mm256_load_si256((__m256i*)w_temp);

                // For every token, dot this 32-wide weight chunk with the token's x_quant chunk.
                for (int t = 0; t < T; ++t) {
                    const int8_t* xq = X_quant.data() + (size_t)t * row_pad + c;
                    __m256i x_vec = _mm256_loadu_si256((const __m256i*)xq);
                    __m256i prod  = _mm256_sign_epi8(x_vec, w_vec);
                    __m256i p_lo  = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(prod));
                    __m256i p_hi  = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(prod, 1));
                    acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(p_lo, ones_16));
                    acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(p_hi, ones_16));
                }
            }

            // Reduce SIMD accumulators + scalar tail per token, write OUT[t][r].
            for (int t = 0; t < T; ++t) {
                int32_t tmp[8];
                _mm256_storeu_si256((__m256i*)tmp, acc[t]);
                int32_t total = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];

                const int8_t* xq = X_quant.data() + (size_t)t * row_pad;
                for (int cc = c; cc < n; ++cc) {
                    int byte_idx  = cc / 4;
                    int bit_shift = (cc % 4) * 2;
                    int8_t w_val = ((packed_ptr[row_offset + byte_idx] >> bit_shift) & 0x03) - 1;
                    if (w_val) total += (w_val == 1) ? xq[cc] : -xq[cc];
                }

                OUT[(size_t)t * n_rows + r] = (float)total / (scale_x_per_t[t] * w.scale);
            }
        }

        // Optional LoRA delta in batched form.
        if (lora) {
            apply_lora_delta_batched(X_norm.data(), T, *lora, lora_scale, OUT);
        }
    }

}
