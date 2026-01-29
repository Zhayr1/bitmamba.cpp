#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <string>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <random>
#include <immintrin.h>
#include <sstream>
#include "tokenizer.h"
// --- MONITORING FUNCTIONS ---

// Get current RAM usage of the process in MB (Linux)
double get_memory_usage_mb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            // VmRSS is the resident memory (physical RAM used)
            std::istringstream iss(line.substr(6));
            long kb;
            iss >> kb;
            return kb / 1024.0; // Convertir a MB
        }
    }
    return 0.0;
}

// Structure for inference statistics
struct InferenceStats {
    int total_tokens;
    double total_time_ms;
    double peak_memory_mb;
    double initial_memory_mb;
    
    InferenceStats() : total_tokens(0), total_time_ms(0), peak_memory_mb(0), initial_memory_mb(0) {}
    
    double tokens_per_second() const {
        if (total_time_ms <= 0) return 0;
        return (total_tokens * 1000.0) / total_time_ms;
    }
    
    void print_summary() const {
        std::cerr << "\n=== INFERENCE STATISTICS ===" << std::endl;
        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "Generated tokens: " << total_tokens << std::endl;
        std::cerr << "Total time: " << total_time_ms << " ms" << std::endl;
        std::cerr << "Speed: " << tokens_per_second() << " tokens/sec" << std::endl;
        std::cerr << "Initial RAM: " << initial_memory_mb << " MB" << std::endl;
        std::cerr << "Peak RAM: " << peak_memory_mb << " MB" << std::endl;
        std::cerr << "RAM used (inference): " << (peak_memory_mb - initial_memory_mb) << " MB" << std::endl;
        std::cerr << "===================================" << std::endl;
    }
};

// --- CONFIGURATION ---
struct Config {
    int vocab_size, d_model, n_layers, n_heads;
};

struct Tensor {
    std::vector<float> data;
    std::vector<uint8_t> packed_data; 
    int rows, cols;
    bool is_bitnet;
    float scale;
    Tensor() : rows(0), cols(0), is_bitnet(false), scale(1.0f) {}
};

struct TokenProb { int id; float val; };

// --- LOOKUP TABLE (LUT) FOR SPEED ---
// Maps 1 packed byte (4 2-bit weights) -> 4 unpacked bytes (int8)
// Stores the result as uint32_t to load 4 weights at once.
static uint32_t UNPACK_LUT[256];

void init_lut() {
    for (int i = 0; i < 256; ++i) {
        int8_t w[4];
        // Unpack logically
        // 00->-1, 01->0, 10->1, 11->1
        w[0] = (int8_t)((i & 0x03) - 1);
        w[1] = (int8_t)(((i >> 2) & 0x03) - 1);
        w[2] = (int8_t)(((i >> 4) & 0x03) - 1);
        w[3] = (int8_t)(((i >> 6) & 0x03) - 1);
        
        // Pack into a 32-bit integer (Little Endian: w0 is the least significant byte)
        uint32_t val = 0;
        uint8_t* p = (uint8_t*)&val;
        p[0] = (uint8_t)w[0];
        p[1] = (uint8_t)w[1];
        p[2] = (uint8_t)w[2];
        p[3] = (uint8_t)w[3];
        UNPACK_LUT[i] = val;
    }
    // std::cout << "⚡ LUT Initialized.\n";
}

// --- MATH KERNELS AVX2 ---

inline float silu(float x) { return x / (1.0f + expf(-x)); }
inline float softplus(float x) { return logf(1.0f + expf(x)); }

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

// ----------------------------------------------------------------------
// BITLINEAR FORWARD PACKED AVX2 + LUT (MAX SPEED)
// ----------------------------------------------------------------------
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

// --- SAMPLER ---
int sample_advanced(std::vector<float>& logits, float temp, float min_p, float top_p, int top_k) {
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

// --- MODEL CLASSES ---
class BitMambaBlock {
public:
    Tensor in_proj_norm, in_proj, conv1d_w, conv1d_b, dt_bias, A_log, D, out_proj_norm, out_proj;
    std::vector<float> conv_state, ssm_state;
    int d_model, d_inner, n_heads, head_dim, d_conv; 

    void init_cache(int d_m, int heads, int expand=2, int conv_k=4) {
        d_model = d_m; n_heads = heads; d_inner = d_model * expand; head_dim = d_inner / n_heads; d_conv = conv_k;
        conv_state.assign(d_inner * (d_conv - 1), 0.0f); ssm_state.assign(d_inner, 0.0f);
    }
    void step(const std::vector<float>& u, std::vector<float>& out_buffer) {
        std::vector<float> proj_out(in_proj.rows);
        bitlinear_forward(u, in_proj, in_proj_norm, proj_out);
        float* ptr_z = proj_out.data(); float* ptr_x = proj_out.data() + d_inner;
        float* ptr_B = proj_out.data() + 2 * d_inner; float* ptr_C = proj_out.data() + 2 * d_inner + n_heads;
        float* ptr_dt = proj_out.data() + 2 * d_inner + 2 * n_heads;
        std::vector<float> y(d_inner);

        #pragma omp parallel for
        for (int h = 0; h < n_heads; ++h) {
            float dt_val = softplus(ptr_dt[h] + dt_bias.data[h]);
            float A_val = -expf(A_log.data[h]); float decay = expf(A_val * dt_val);
            float B_val = ptr_B[h]; float C_val = ptr_C[h];
            for (int d = 0; d < head_dim; ++d) {
                int idx = h * head_dim + d; 
                float input_x = ptr_x[idx]; float conv_res = conv1d_b.data[idx];
                for (int k = 0; k < d_conv - 1; ++k) conv_res += conv_state[idx * (d_conv - 1) + k] * conv1d_w.data[idx * d_conv + k];
                conv_res += input_x * conv1d_w.data[idx * d_conv + (d_conv - 1)];
                for (int k = 0; k < d_conv - 2; ++k) conv_state[idx * (d_conv - 1) + k] = conv_state[idx * (d_conv - 1) + k + 1];
                conv_state[idx * (d_conv - 1) + (d_conv - 2)] = input_x;
                float x_act = silu(conv_res); float u_ssm = x_act * B_val * dt_val;
                ssm_state[idx] = ssm_state[idx] * decay + u_ssm; 
                y[idx] = (ssm_state[idx] * C_val + x_act * D.data[idx]) * silu(ptr_z[idx]);
            }
        }
        bitlinear_forward(y, out_proj, out_proj_norm, out_buffer);
    }
};

class BitMambaModel {
public:
    Config config;
    Tensor embed, norm_f, lm_head_norm, lm_head;
    std::vector<BitMambaBlock> layers;
    std::vector<float> current_x, next_x;

    BitMambaModel(const std::string& path) {
        init_lut(); // <--- INITIALIZE LUT
        load_from_bin(path);
        current_x.resize(config.d_model);
        next_x.resize(config.d_model);
    }
    
    void load_from_bin(const std::string& path) {
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

    int forward_step(int token, const std::vector<int>& history, float penalty, 
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
};

// Load tokenizer from binary vocabulary file
static gten::GPT2Tokenizer load_tokenizer(const std::string& vocab_file_path) {
    std::ifstream vin(vocab_file_path, std::ios::binary);
    if (!vin.is_open()) {
        std::cerr << "Error: tokenizer.bin not found: " 
                  << vocab_file_path << "\n";
        exit(1);
    }
    return gten::GPT2Tokenizer{vin};
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model.bin> <input> <mode> [temp] [penalty] [min_p] [top_p] [top_k] [max_tokens]" << std::endl;
        std::cerr << "\nParameters:" << std::endl;
        std::cerr << "  model.bin   - Path to model file" << std::endl;
        std::cerr << "  input       - Input text (tokenizer mode) or token IDs (raw mode)" << std::endl;
        std::cerr << "  mode        - 'tokenizer' (text input/output) or 'raw' (token IDs input/output)" << std::endl;
        std::cerr << "  temp        - Temperature (default: 0.8)" << std::endl;
        std::cerr << "  penalty     - Repetition Penalty (default: 1.15)" << std::endl;
        std::cerr << "  min_p       - Min-P sampling (default: 0.05)" << std::endl;
        std::cerr << "  top_p       - Top-P/nucleus sampling (default: 0.90)" << std::endl;
        std::cerr << "  top_k       - Top-K sampling (default: 40)" << std::endl;
        std::cerr << "  max_tokens  - Max tokens to generate (default: 400)" << std::endl;
        std::cerr << "\nExamples:" << std::endl;
        std::cerr << "  Tokenizer mode: ./bitmamba model.bin \"Hello, I am\" tokenizer 0.7 1.1" << std::endl;
        std::cerr << "  Raw mode:       ./bitmamba model.bin \"15496 11 314 716\" raw 0.7 1.1" << std::endl;
        return 1;
    }

    // Validate mode argument
    std::string mode = argv[3];
    if (mode != "tokenizer" && mode != "raw") {
        std::cerr << "Error: Invalid mode '" << mode << "'" << std::endl;
        std::cerr << "Mode must be either 'tokenizer' or 'raw'" << std::endl;
        std::cerr << "  tokenizer - Text input/output (uses GPT-2 tokenizer)" << std::endl;
        std::cerr << "  raw       - Token IDs input/output (numeric)" << std::endl;
        return 1;
    }
    bool use_tokenizer = (mode == "tokenizer");

    float temp = 0.8f; float penalty = 1.15f; float min_p = 0.05f; float top_p = 0.90f; int top_k = 40;
    int max_tokens = 400;
    if (argc > 4) temp = std::stof(argv[4]);
    if (argc > 5) penalty = std::stof(argv[5]);
    if (argc > 6) min_p = std::stof(argv[6]);
    if (argc > 7) top_p = std::stof(argv[7]);
    if (argc > 8) top_k = std::stoi(argv[8]);
    if (argc > 9) max_tokens = std::stoi(argv[9]);

    // Measure RAM before loading model
    double ram_before_model = get_memory_usage_mb();
    std::cerr << "[INFO] RAM before loading model: " << std::fixed << std::setprecision(2) << ram_before_model << " MB" << std::endl;

    BitMambaModel model(argv[1]);
    
    double ram_after_model = get_memory_usage_mb();
    std::cerr << "[INFO] RAM after loading model: " << ram_after_model << " MB (model: " << (ram_after_model - ram_before_model) << " MB)" << std::endl;
    
    // Initialize tokenizer only if needed
    gten::GPT2Tokenizer tokenizer;
    if (use_tokenizer) {
        tokenizer = load_tokenizer("tokenizer.bin");
    }

    // Parse input based on mode
    std::vector<int32_t> prompt_ids;
    std::string input_str = argv[2];
    
    if (use_tokenizer) {
        // Tokenizer mode: encode text to tokens
        prompt_ids = tokenizer.encode(input_str);
        std::cerr << "[INFO] Input Text: \"" << input_str << "\"" << std::endl;
    } else {
        // Raw mode: parse space-separated token IDs
        std::string delimiter = " ";
        size_t pos = 0;
        try {
            while ((pos = input_str.find(delimiter)) != std::string::npos) {
                std::string t = input_str.substr(0, pos);
                if (!t.empty()) prompt_ids.push_back(std::stoi(t));
                input_str.erase(0, pos + delimiter.length());
            }
            if (!input_str.empty()) prompt_ids.push_back(std::stoi(input_str));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: Invalid input for raw mode. Expected space-separated token IDs (numbers)." << std::endl;
            std::cerr << "Example: \"15496 11 314 716\"" << std::endl;
            std::cerr << "If you want to use text input, use 'tokenizer' mode instead." << std::endl;
            return 1;
        }
    }

    std::cerr << "[INFO] Input Tokens (" << prompt_ids.size() << "): ";
    for (int id : prompt_ids) std::cerr << id << " ";
    std::cerr << std::endl;

    // Initialize stats
    InferenceStats stats;
    stats.initial_memory_mb = get_memory_usage_mb();
    stats.peak_memory_mb = stats.initial_memory_mb;

    // Process prompt (prefill)
    std::cerr << "[INFO] Processing prompt..." << std::endl;
    auto prefill_start = std::chrono::high_resolution_clock::now();
    
    int current = prompt_ids[0];
    std::vector<int> history;
    for (size_t i = 0; i < prompt_ids.size() - 1; ++i) {
        model.forward_step(prompt_ids[i], history, 1.0f, 0.0f, 0.0f, 1.0f, 0); 
        history.push_back(prompt_ids[i]);
    }
    current = prompt_ids.back();
    history.push_back(current);
    
    auto prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_time = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
    std::cerr << "[INFO] Prefill completed in " << std::fixed << std::setprecision(2) << prefill_time << " ms (" 
              << prompt_ids.size() << " tokens)" << std::endl;

    // Generation
    std::cerr << "[INFO] Generating tokens..." << std::endl;
    
    srand(time(0));
    auto gen_start = std::chrono::high_resolution_clock::now();
    
    // Vector to accumulate all generated tokens
    std::vector<int> generated_tokens;
    
    for(int i = 0; i < max_tokens; ++i) {
        auto token_start = std::chrono::high_resolution_clock::now();
        
        int next = model.forward_step(current, history, penalty, temp, min_p, top_p, top_k);
        
        auto token_end = std::chrono::high_resolution_clock::now();
        double token_time = std::chrono::duration<double, std::milli>(token_end - token_start).count();
        
        // Update stats
        stats.total_tokens++;
        stats.total_time_ms += token_time;
        double current_mem = get_memory_usage_mb();
        if (current_mem > stats.peak_memory_mb) stats.peak_memory_mb = current_mem;
        
        // Accumulate generated token
        generated_tokens.push_back(next);
        
        // Show usage every 10 tokens (to stderr, without overwriting)
        if (stats.total_tokens % 10 == 0) {
            std::cerr << "[STATS] " << stats.total_tokens << " tokens | " 
                      << std::fixed << std::setprecision(2) << stats.tokens_per_second() << " tok/s | "
                      << "RAM: " << current_mem << " MB" << std::endl;
        }
        
        current = next;
        history.push_back(next);
        if(history.size() > 256) history.erase(history.begin());
        
        // Stop tokens
        if (next == 50256 || next == 0) break; 
    }
    
    // Output based on mode
    if (use_tokenizer) {
        std::cout << "\n=== Generated Text ===" << std::endl;
        for (int token : generated_tokens) {
            std::cout << tokenizer.decode(token);
        }
        std::cout << "\n=== End Inference ===" << std::endl;
    } else {
        std::cout << "\n=== Generated Token IDs ===" << std::endl;
        for (int token : generated_tokens) {
            std::cout << token << " ";
        }
        std::cout << "\n=== End Inference ===" << std::endl;
    }
    
    // Print final summary
    stats.print_summary();
    
    return 0;
}
