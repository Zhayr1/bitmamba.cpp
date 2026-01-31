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

    void bitlinear_forward(const std::vector<float>& x, const Tensor& w, const Tensor& norm_w, std::vector<float>& out) {
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
    }

}
