#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../src/tokenizer.h"
#include "bitmamba/model.h"
#include "bitmamba/utils.h"

using namespace bitmamba;

// Load tokenizer from binary vocabulary file
static gten::GPT2Tokenizer load_tokenizer(const std::string &vocab_file_path) {
  std::ifstream vin(vocab_file_path, std::ios::binary);
  if (!vin.is_open()) {
    std::cerr << "Error: tokenizer.bin not found: " << vocab_file_path << "\n";
    exit(1);
  }
  return gten::GPT2Tokenizer{vin};
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model.bin> <input> <mode> [temp] [penalty] [min_p] [top_p] "
                 "[top_k] [max_tokens] [output_mode]"
              << std::endl;
    std::cerr << "\nParameters:" << std::endl;
    std::cerr << "  model.bin   - Path to model file" << std::endl;
    std::cerr
        << "  input       - Input text (tokenizer mode) or token IDs (raw mode)"
        << std::endl;
    std::cerr << "  mode        - 'tokenizer' (text input/output) or 'raw' "
                 "(token IDs input/output)"
              << std::endl;
    std::cerr << "  temp        - Temperature (default: 0.8)" << std::endl;
    std::cerr << "  penalty     - Repetition Penalty (default: 1.15)"
              << std::endl;
    std::cerr << "  min_p       - Min-P sampling (default: 0.05)" << std::endl;
    std::cerr << "  top_p       - Top-P/nucleus sampling (default: 0.90)"
              << std::endl;
    std::cerr << "  top_k       - Top-K sampling (default: 40)" << std::endl;
    std::cerr << "  max_tokens  - Max tokens to generate (default: 400)"
              << std::endl;
    std::cerr << "  output_mode - 'bench' (default) shows stats, 'clean' shows "
                 "only output"
              << std::endl;
    std::cerr << "\nExamples:" << std::endl;
    std::cerr << "  Tokenizer mode: ./bitmamba model.bin \"Hello, I am\" "
                 "tokenizer 0.7 1.1"
              << std::endl;
    std::cerr << "  Raw mode:       ./bitmamba model.bin \"15496 11 314 716\" "
                 "raw 0.7 1.1"
              << std::endl;
    return 1;
  }

  // Validate mode argument
  std::string mode = argv[3];
  if (mode != "tokenizer" && mode != "raw") {
    std::cerr << "Error: Invalid mode '" << mode << "'" << std::endl;
    std::cerr << "Mode must be either 'tokenizer' or 'raw'" << std::endl;
    std::cerr << "  tokenizer - Text input/output (uses GPT-2 tokenizer)"
              << std::endl;
    std::cerr << "  raw       - Token IDs input/output (numeric)" << std::endl;
    return 1;
  }
  bool use_tokenizer = (mode == "tokenizer");

  float temp = 0.8f;
  float penalty = 1.15f;
  float min_p = 0.05f;
  float top_p = 0.90f;
  int top_k = 40;
  int max_tokens = 400;
  if (argc > 4)
    temp = std::stof(argv[4]);
  if (argc > 5)
    penalty = std::stof(argv[5]);
  if (argc > 6)
    min_p = std::stof(argv[6]);
  if (argc > 7)
    top_p = std::stof(argv[7]);
  if (argc > 8)
    top_k = std::stoi(argv[8]);
  if (argc > 9)
    max_tokens = std::stoi(argv[9]);

  // Check for optional "clean" or "bench" mode argument at the end
  std::string output_mode = "bench";
  if (argc > 10) {
    output_mode = argv[10];
  }
  bool is_clean = (output_mode == "clean");

  // Measure RAM before loading model
  double ram_before_model = get_memory_usage_mb();
  if (!is_clean)
    std::cerr << "[INFO] RAM before loading model: " << std::fixed
              << std::setprecision(2) << ram_before_model << " MB" << std::endl;

  BitMambaModel model(argv[1]);

  double ram_after_model = get_memory_usage_mb();
  if (!is_clean)
    std::cerr << "[INFO] RAM after loading model: " << ram_after_model
              << " MB (model: " << (ram_after_model - ram_before_model)
              << " MB)" << std::endl;

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
    if (!is_clean)
      std::cerr << "[INFO] Input Text: \"" << input_str << "\"" << std::endl;
  } else {
    // Raw mode: parse space-separated token IDs
    std::string delimiter = " ";
    size_t pos = 0;
    try {
      while ((pos = input_str.find(delimiter)) != std::string::npos) {
        std::string t = input_str.substr(0, pos);
        if (!t.empty())
          prompt_ids.push_back(std::stoi(t));
        input_str.erase(0, pos + delimiter.length());
      }
      if (!input_str.empty())
        prompt_ids.push_back(std::stoi(input_str));
    } catch (const std::invalid_argument &e) {
      std::cerr << "Error: Invalid input for raw mode. Expected "
                   "space-separated token IDs (numbers)."
                << std::endl;
      std::cerr << "Example: \"15496 11 314 716\"" << std::endl;
      std::cerr
          << "If you want to use text input, use 'tokenizer' mode instead."
          << std::endl;
      return 1;
    }
  }

  if (!is_clean) {
    std::cerr << "[INFO] Input Tokens (" << prompt_ids.size() << "): ";
    for (int id : prompt_ids)
      std::cerr << id << " ";
    std::cerr << std::endl;
  }

  // Initialize stats
  InferenceStats stats;
  stats.initial_memory_mb = get_memory_usage_mb();
  stats.peak_memory_mb = stats.initial_memory_mb;

  // Process prompt (prefill)
  if (!is_clean)
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
  double prefill_time =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start)
          .count();
  if (!is_clean)
    std::cerr << "[INFO] Prefill completed in " << std::fixed
              << std::setprecision(2) << prefill_time << " ms ("
              << prompt_ids.size() << " tokens)" << std::endl;

  // Generation
  if (!is_clean)
    std::cerr << "[INFO] Generating tokens..." << std::endl;

  srand(time(0));

  // Vector to accumulate all generated tokens
  std::vector<int> generated_tokens;

  for (int i = 0; i < max_tokens; ++i) {
    auto token_start = std::chrono::high_resolution_clock::now();

    int next = model.forward_step(current, history, penalty, temp, min_p, top_p,
                                  top_k);

    auto token_end = std::chrono::high_resolution_clock::now();
    double token_time =
        std::chrono::duration<double, std::milli>(token_end - token_start)
            .count();

    // Update stats
    stats.total_tokens++;
    stats.total_time_ms += token_time;
    double current_mem = get_memory_usage_mb();
    if (current_mem > stats.peak_memory_mb)
      stats.peak_memory_mb = current_mem;

    // Accumulate generated token
    generated_tokens.push_back(next);

    // Show usage every 10 tokens (to stderr, without overwriting) if not clean
    if (!is_clean && stats.total_tokens % 10 == 0) {
      std::cerr << "[STATS] " << stats.total_tokens << " tokens | "
                << std::fixed << std::setprecision(2)
                << stats.tokens_per_second() << " tok/s | "
                << "RAM: " << current_mem << " MB" << std::endl;
    }

    current = next;
    history.push_back(next);
    if (history.size() > 256)
      history.erase(history.begin());

    // Stop tokens
    if (next == 50256 || next == 0)
      break;
  }

  // Output based on mode
  if (use_tokenizer) {
    if (!is_clean)
      std::cout << "\n=== Generated Text ===" << std::endl;
    for (int token : generated_tokens) {
      std::cout << tokenizer.decode(token);
    }
    if (!is_clean)
      std::cout << "\n=== End Inference ===" << std::endl;
    else
      std::cout << std::endl; // Ensure newline at end of clean output
  } else {
    if (!is_clean)
      std::cout << "\n=== Generated Token IDs ===" << std::endl;
    for (int token : generated_tokens) {
      std::cout << token << " ";
    }
    if (!is_clean)
      std::cout << "\n=== End Inference ===" << std::endl;
    else
      std::cout << std::endl;
  }

  // Print final summary only if not clean
  if (!is_clean) {
    stats.print_summary();
  }

  return 0;
}
