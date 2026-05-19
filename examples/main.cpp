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

#include <immintrin.h>

// ---------------------------------------------------------------------------
// Named-flag scanner
// Removes the flag (and its value) from argv and returns the integer/string value.
// ---------------------------------------------------------------------------
static int extract_int_flag(int& argc, char**& argv,
                             const std::string& flag, int default_val) {
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == flag) {
      int val = std::stoi(argv[i + 1]);
      // Remove flag + value from argv (shift left)
      for (int j = i; j < argc - 2; ++j) argv[j] = argv[j + 2];
      argc -= 2;
      return val;
    }
  }
  return default_val;
}

static bool extract_bool_flag(int& argc, char**& argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == flag) {
      for (int j = i; j < argc - 1; ++j) argv[j] = argv[j + 1];
      argc -= 1;
      return true;
    }
  }
  return false;
}

static std::string extract_string_flag(int& argc, char**& argv,
                                        const std::string& flag,
                                        const std::string& default_val) {
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == flag) {
      std::string val = argv[i + 1];
      for (int j = i; j < argc - 2; ++j) argv[j] = argv[j + 2];
      argc -= 2;
      return val;
    }
  }
  return default_val;
}

int main(int argc, char **argv) {
  // -------------------------------------------------------------------------
  // Parse named (optional) flags FIRST so they don't interfere with positional
  // argument indexing.
  // -------------------------------------------------------------------------
  bool chat_mode   = extract_bool_flag(argc, argv, "--chat");
  int repeat_start = extract_int_flag(argc, argv, "--repeat-start", -1);
  int repeat_end   = extract_int_flag(argc, argv, "--repeat-end",   -1);
  int repeat_count = extract_int_flag(argc, argv, "--repeat-count",  1);
  std::string tokens_file_path =
      extract_string_flag(argc, argv, "--tokens-file", "");
  int ppl_window_size =
      extract_int_flag(argc, argv, "--ppl-window", 0);
  std::string lora_path =
      extract_string_flag(argc, argv, "--lora", "");
  // Prefill mode: batched is the default (~2-3× faster on long prompts).
  // Pass --sequential-prefill to force the old per-token path for A/B testing.
  bool sequential_prefill = extract_bool_flag(argc, argv, "--sequential-prefill");

  // -------------------------------------------------------------------------
  // Positional argument validation
  // -------------------------------------------------------------------------
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " [--repeat-start N] [--repeat-end N]"
              << " <model.bin> <input> <mode>"
              << " [temp] [penalty] [min_p] [top_p] [top_k] [max_tokens] [output_mode]"
              << std::endl;
    std::cerr << "\nParameters:" << std::endl;
    std::cerr << "  model.bin    - Path to model file" << std::endl;
    std::cerr << "  input        - Input text (tokenizer mode) or token IDs (raw mode)" << std::endl;
    std::cerr << "  mode         - 'tokenizer' (text input/output) or 'raw' (token IDs)" << std::endl;
    std::cerr << "  temp         - Temperature (default: 0.8)" << std::endl;
    std::cerr << "  penalty      - Repetition Penalty (default: 1.15)" << std::endl;
    std::cerr << "  min_p        - Min-P sampling (default: 0.05)" << std::endl;
    std::cerr << "  top_p        - Top-P/nucleus sampling (default: 0.90)" << std::endl;
    std::cerr << "  top_k        - Top-K sampling (default: 40)" << std::endl;
    std::cerr << "  max_tokens   - Max tokens to generate (default: 400)" << std::endl;
    std::cerr << "  output_mode  - 'bench' (default) shows stats, 'clean' shows only output," << std::endl;
    std::cerr << "                 'score' runs prefill only and prints Top-1 token ID" << std::endl;
    std::cerr << "\nLLM Neuroanatomy / RYS flags (optional):" << std::endl;
    std::cerr << "  --repeat-start N  - First layer of the repeated slice (0-indexed)" << std::endl;
    std::cerr << "  --repeat-end   N  - Last layer of the repeated slice (inclusive)" << std::endl;
    std::cerr << "  --repeat-count N  - How many extra times to execute the slice (default: 1)" << std::endl;
    std::cerr << "\nLoRA adapter (optional):" << std::endl;
    std::cerr << "  --lora <path>     - Apply a .lora.bin adapter on top of the base weights" << std::endl;
    std::cerr << "\nPrefill mode (optional):" << std::endl;
    std::cerr << "  --sequential-prefill - Force per-token prefill (default is layer-major batched, ~2-3× faster)" << std::endl;
    std::cerr << "\nChat mode (interactive, ChatML template):" << std::endl;
    std::cerr << "  --chat            - Enable interactive chat loop" << std::endl;
    std::cerr << "\nExamples:" << std::endl;
    std::cerr << "  Tokenizer mode: ./bitmamba model.bin \"Hello, I am\" tokenizer 0.7 1.1" << std::endl;
    std::cerr << "  Raw mode:       ./bitmamba model.bin \"15496 11 314 716\" raw 0.7 1.1" << std::endl;
    std::cerr << "  RYS mode:       ./bitmamba --repeat-start 10 --repeat-end 15 model.bin \"Hello\" tokenizer" << std::endl;
    std::cerr << "  Chat mode:      ./bitmamba --chat model.bin \"\" tokenizer 0.7 1.1 0.05 0.9 40 300" << std::endl;
    std::cerr << "  Chat + RYS:     ./bitmamba --chat --repeat-start 17 --repeat-end 21 model.bin \"\" tokenizer" << std::endl;
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

  float temp       = 0.8f;
  float penalty    = 1.15f;
  float min_p      = 0.05f;
  float top_p      = 0.90f;
  int   top_k      = 40;
  int   max_tokens = 400;
  if (argc > 4)  temp       = std::stof(argv[4]);
  if (argc > 5)  penalty    = std::stof(argv[5]);
  if (argc > 6)  min_p      = std::stof(argv[6]);
  if (argc > 7)  top_p      = std::stof(argv[7]);
  if (argc > 8)  top_k      = std::stoi(argv[8]);
  if (argc > 9)  max_tokens = std::stoi(argv[9]);

  std::string output_mode = "bench";
  if (argc > 10) output_mode = argv[10];
  bool is_clean   = (output_mode == "clean");
  bool is_score   = (output_mode == "score");
  bool is_ppl     = (output_mode == "perplexity");
  bool is_logprob = (output_mode == "logprob");

  if (output_mode != "bench" && output_mode != "clean" &&
      !is_score && !is_ppl && !is_logprob) {
    std::cerr << "Error: Invalid <output_mode>. Use 'bench', 'clean', 'score', 'perplexity', or 'logprob'.\n";
    return 1;
  }

  // Validate RYS flags if provided
  if (repeat_start != -1 || repeat_end != -1) {
    if (repeat_start < 0 || repeat_end < repeat_start) {
      std::cerr << "Error: --repeat-start must be >= 0 and <= --repeat-end\n";
      return 1;
    }
    if (repeat_count < 1) {
      std::cerr << "Error: --repeat-count must be >= 1\n";
      return 1;
    }
    if (!is_clean) {
      std::cerr << "[RYS] Neuroanatomy mode: layers "
                << repeat_start << "–" << repeat_end
                << " ×" << (repeat_count + 1) << "\n";
    }
  }

  // Measure RAM before loading model
  double ram_before_model = get_memory_usage_mb();
  if (!is_clean)
    std::cerr << "[INFO] RAM before loading model: " << std::fixed
              << std::setprecision(2) << ram_before_model << " MB" << std::endl;

  // Load model — pass RYS parameters; defaults (-1, -1) = normal inference.
  BitMambaModel model(argv[1], repeat_start, repeat_end, repeat_count);

  // Optionally load a LoRA adapter and validate shape compatibility.
  if (!lora_path.empty()) {
    if (!is_clean)
      std::cerr << "[INFO] Loading LoRA adapter: " << lora_path << std::endl;
    model.load_lora(lora_path);
  }

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
    prompt_ids = tokenizer.encode(input_str);
    if (!is_clean)
      std::cerr << "[INFO] Input Text: \"" << input_str << "\"" << std::endl;
  } else if (!tokens_file_path.empty()) {
    // --tokens-file: load tokens from a file. .bin extension is treated as
    // raw int32 little-endian (matches wikitext2_test.bin / GPU eval format).
    // Anything else is parsed as ascii whitespace-separated integers.
    bool is_bin = tokens_file_path.size() >= 4 &&
                  tokens_file_path.substr(tokens_file_path.size() - 4) == ".bin";
    std::ifstream tf(tokens_file_path, is_bin ? std::ios::binary : std::ios::in);
    if (!tf.is_open()) {
      std::cerr << "Error: cannot open tokens file: " << tokens_file_path
                << std::endl;
      return 1;
    }
    if (is_bin) {
      tf.seekg(0, std::ios::end);
      std::streamsize sz = tf.tellg();
      tf.seekg(0, std::ios::beg);
      size_t n = (size_t)sz / 4;
      prompt_ids.resize(n);
      tf.read(reinterpret_cast<char *>(prompt_ids.data()), n * 4);
    } else {
      int v;
      while (tf >> v) prompt_ids.push_back(v);
    }
    if (!is_clean)
      std::cerr << "[INFO] Loaded " << prompt_ids.size() << " tokens from "
                << tokens_file_path << (is_bin ? " (binary)" : " (ascii)")
                << std::endl;
  } else {
    std::string delimiter = " ";
    size_t pos = 0;
    try {
      while ((pos = input_str.find(delimiter)) != std::string::npos) {
        std::string t = input_str.substr(0, pos);
        if (!t.empty()) prompt_ids.push_back(std::stoi(t));
        input_str.erase(0, pos + delimiter.length());
      }
      if (!input_str.empty()) prompt_ids.push_back(std::stoi(input_str));
    } catch (const std::invalid_argument &e) {
      std::cerr << "Error: Invalid input for raw mode. Expected "
                   "space-separated token IDs (numbers)." << std::endl;
      std::cerr << "Example: \"15496 11 314 716\"" << std::endl;
      std::cerr << "If you want to use text input, use 'tokenizer' mode instead."
                << std::endl;
      return 1;
    }
  }

  if (!is_clean) {
    std::cerr << "[INFO] Input Tokens (" << prompt_ids.size() << ")";
    // Avoid flooding stderr for long sequences (e.g. full wikitext PPL).
    if (prompt_ids.size() <= 2048) {
      std::cerr << ": ";
      for (int id : prompt_ids) std::cerr << id << " ";
    }
    std::cerr << std::endl;
  }

  // Initialize stats
  InferenceStats stats;
  stats.initial_memory_mb = get_memory_usage_mb();
  stats.peak_memory_mb    = stats.initial_memory_mb;

  // ── CHAT MODE ────────────────────────────────────────────────────────────
  // Interactive ChatML loop for SFT models.
  // input_str (argv[2]) is used as an optional system prompt.
  // SSM state persists between turns — Mamba remembers the conversation.
  if (chat_mode && use_tokenizer) {
    srand(time(0));
    std::string system_prompt = input_str;  // argv[2], empty string = no system prompt

    std::cerr << "\nBitMamba-2 Chat";
    if (repeat_start >= 0)
      std::cerr << " [RYS " << repeat_start << "-" << repeat_end << "]";
    std::cerr << " — type 'exit' to quit\n" << std::endl;

    std::vector<int> history;
    bool first_turn = true;

    while (true) {
      // 1. Read user input
      std::cerr << "You: ";
      std::string user_input;
      if (!std::getline(std::cin, user_input)) break;
      if (user_input == "exit" || user_input == "quit" || user_input == "salir") break;
      if (user_input.empty()) continue;

      // 2. Format with ChatML template
      std::string chat_text;
      if (first_turn && !system_prompt.empty()) {
        chat_text = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
      }
      chat_text += "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
      first_turn = false;

      // 3. Tokenize the new turn
      auto turn_ids = tokenizer.encode(chat_text);

      // 4. Prefill — feed tokens into SSM state without generating
      for (size_t i = 0; i < turn_ids.size() - 1; ++i) {
        model.forward_step(turn_ids[i], history, 1.0f, 0.0f, 0.0f, 1.0f, 0);
        history.push_back(turn_ids[i]);
      }
      int current = turn_ids.back();
      history.push_back(current);

      // 5. Generate until <|im_end|> or max_tokens
      std::string generated_text;
      std::string print_buffer;
      std::cout << "\nAssistant: " << std::flush;

      for (int i = 0; i < max_tokens; ++i) {
        int next = model.forward_step(current, history, penalty, temp, min_p, top_p, top_k);

        std::string token_text = tokenizer.decode(next);
        generated_text += token_text;
        print_buffer += token_text;

        // Check stop conditions
        if (generated_text.find("<|im_end|>") != std::string::npos ||
            generated_text.find("<|endoftext|>") != std::string::npos ||
            next == 50256) {
          // Print remaining buffer without the stop tag
          size_t stop_pos = print_buffer.find("<|im_end|>");
          if (stop_pos == std::string::npos)
            stop_pos = print_buffer.find("<|endoftext|>");
          if (stop_pos != std::string::npos)
            print_buffer = print_buffer.substr(0, stop_pos);
          std::cout << print_buffer;
          break;
        }

        // Buffer heuristic: avoid printing fragments of special tags
        bool maybe_tag = (print_buffer.find('<') != std::string::npos ||
                          print_buffer.find('|') != std::string::npos);
        if (!maybe_tag || print_buffer.size() > 20) {
          std::cout << print_buffer << std::flush;
          print_buffer.clear();
        }

        current = next;
        history.push_back(next);
        if (history.size() > 256) history.erase(history.begin());
      }

      std::cout << "\n" << std::endl;

      // 6. Close the assistant turn in SSM state
      auto end_ids = tokenizer.encode("<|im_end|>\n");
      for (auto id : end_ids) {
        model.forward_step(id, history, 1.0f, 0.0f, 0.0f, 1.0f, 0);
        history.push_back(id);
      }
    }
    return 0;
  }
  // ────────────────────────────────────────────────────────────────────────────

  // ── PERPLEXITY MODE ────────────────────────────────────────────────────────
  // Iterates through the prompt, accumulating the log-probability of the NEXT
  // real token. Does not generate any new tokens. Prints exactly one line.
  if (is_ppl) {
    if (prompt_ids.size() < 2) {
      std::cerr << "Error: Perplexity mode requires at least 2 tokens in the prompt.\n";
      return 1;
    }

    if (!is_clean)
      std::cerr << "[INFO] Calculating perplexity over " << prompt_ids.size() << " tokens..." << std::endl;
    if (ppl_window_size > 0 && !is_clean)
      std::cerr << "[INFO] PPL window reset every " << ppl_window_size
                << " tokens (matches mamba-3 / GPU windowed eval)" << std::endl;

    auto ppl_start = std::chrono::high_resolution_clock::now();
    double total_log_prob = 0.0;
    std::vector<int> history;

    // We evaluate the probability of prompt_ids[i+1] given the context up to prompt_ids[i]
    for (size_t i = 0; i < prompt_ids.size() - 1; ++i) {
      if (ppl_window_size > 0 && i > 0 &&
          (i % (size_t)ppl_window_size) == 0) {
        model.reset_states();
      }
      int current_tok = prompt_ids[i];
      int target_tok  = prompt_ids[i+1];
      
      // Calculate log-prob of the target token
      auto [rank, lp] = model.compute_log_prob(current_tok, target_tok);
      total_log_prob += lp;
      
      if (!is_clean && (i + 1) % 100 == 0) {
        std::cerr << "[INFO] Processed " << (i + 1) << "/" << (prompt_ids.size() - 1) 
                  << " tokens. Current PPL: " << exp(-total_log_prob / (i + 1)) << "\r";
      }
    }
    
    if (!is_clean) std::cerr << std::endl;

    double avg_nll = -total_log_prob / (prompt_ids.size() - 1);
    double perplexity = exp(avg_nll);
    
    auto ppl_end = std::chrono::high_resolution_clock::now();
    double ppl_time = std::chrono::duration<double, std::milli>(ppl_end - ppl_start).count();

    if (!is_clean) {
      std::cerr << "[INFO] Calculation finished in " << std::fixed
                << std::setprecision(2) << ppl_time << " ms" << std::endl;
    }
    
    // Final machine-readable output
    std::cout << "[PPL] Perplexity: " << std::fixed << std::setprecision(4) << perplexity << std::endl;
    return 0;
  }
  // ────────────────────────────────────────────────────────────────────────────

  // ── LOGPROB MODE ─────────────────────────────────────────────────────────
  // Scores a multi-token completion given a context.
  // The full token sequence is in prompt_ids. argv[11] specifies how many
  // tokens from the end are the "completion" to score.
  // Context tokens are prefilled, then each completion token is scored
  // via compute_log_prob (replicating lm_eval loglikelihood).
  //
  // Output: [LOGPROB] Tokens: N | TotalLP: -X.XXXX | AvgLP: -X.XXXX
  if (is_logprob) {
    if (argc <= 11) {
      std::cerr << "Error: logprob mode requires <completion_len> as extra argument.\n";
      std::cerr << "Usage: ... logprob <completion_token_count>\n";
      return 1;
    }
    int completion_len = std::stoi(argv[11]);
    int total_len = (int)prompt_ids.size();
    int context_len = total_len - completion_len;

    if (context_len < 1 || completion_len < 1) {
      std::cerr << "Error: Need at least 1 context token and 1 completion token.\n";
      return 1;
    }

    // Prefill context tokens (build up SSM state)
    std::vector<int> history;
    for (int i = 0; i < context_len - 1; ++i) {
      model.prefill_step(prompt_ids[i]);
      history.push_back(prompt_ids[i]);
    }

    // Score each completion token sequentially
    double total_lp = 0.0;
    int current = prompt_ids[context_len - 1];
    for (int i = context_len; i < total_len; ++i) {
      auto [rank, lp] = model.compute_log_prob(current, prompt_ids[i]);
      total_lp += lp;
      current = prompt_ids[i];
    }

    double avg_lp = total_lp / completion_len;
    std::cout << "[LOGPROB] Tokens: " << completion_len
              << " | TotalLP: " << std::fixed << std::setprecision(6) << total_lp
              << " | AvgLP: " << std::fixed << std::setprecision(6) << avg_lp
              << std::endl;
    return 0;
  }
  // ────────────────────────────────────────────────────────────────────────────

  // Process prompt (prefill) for bench/clean/score modes
  if (!is_clean)
    std::cerr << "[INFO] Processing prompt..." << std::endl;
  auto prefill_start = std::chrono::high_resolution_clock::now();

  int current = prompt_ids[0];
  std::vector<int> history;
  if (sequential_prefill) {
    // Per-token prefill (slower; kept for validation and edge cases).
    for (size_t i = 0; i < prompt_ids.size() - 1; ++i) {
      model.prefill_step(prompt_ids[i]);
      history.push_back(prompt_ids[i]);
    }
  } else {
    // Layer-major batched prefill (default): each layer's W_in/W_out runs
    // as a single GEMM over all T tokens, so weights are streamed from
    // DRAM once instead of T times.
    std::vector<int> ctx(prompt_ids.begin(), prompt_ids.end() - 1);
    model.prefill_sequence(ctx);
    history.insert(history.end(), ctx.begin(), ctx.end());
  }
  current = prompt_ids.back();
  history.push_back(current);

  auto prefill_end = std::chrono::high_resolution_clock::now();
  double prefill_time =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
  if (!is_clean) {
    std::cerr << "[INFO] Prefill completed in " << std::fixed
              << std::setprecision(2) << prefill_time << " ms ("
              << prompt_ids.size() << " tokens)" << std::endl;
  }

  // ── SCORE MODE ─────────────────────────────────────────────────────────────
  // Without target token:  prints top-1 token ID (argmax).
  // With target token ID (argv[11]): runs compute_log_prob and prints
  //   rank (0-indexed) + log-probability of the target token.
  //   This gives continuous gradient signal even when argmax is wrong.
  //
  // Output formats:
  //   [SCORE] Top-1 Token ID: <id>
  //   [SCORE] Target: <id> | Rank: <rank> | LogProb: <lp>
  if (is_score) {
    if (argc > 11) {
      // Log-prob mode: score against a specific target token
      int target_id = std::stoi(argv[11]);
      auto [rank, lp] = model.compute_log_prob(current, target_id);
      std::cout << "[SCORE] Target: " << target_id
                << " | Rank: "   << rank
                << " | LogProb: " << std::fixed << std::setprecision(6) << lp
                << std::endl;
    } else {
      // Argmax mode (backward-compatible)
      std::vector<int> empty_history;
      int top1 = model.forward_step(current, empty_history,
                                    1.0f, 0.0f, 0.0f, 1.0f, 0);
      std::cout << "[SCORE] Top-1 Token ID: " << top1 << std::endl;
    }
    return 0;
  }
  // ────────────────────────────────────────────────────────────────────────────

  // Generation
  if (!is_clean)
    std::cerr << "[INFO] Generating tokens..." << std::endl;

  srand(time(0));

  std::vector<int> generated_tokens;

  for (int i = 0; i < max_tokens; ++i) {
    auto token_start = std::chrono::high_resolution_clock::now();

    int next = model.forward_step(current, history, penalty, temp, min_p, top_p, top_k);

    auto token_end = std::chrono::high_resolution_clock::now();
    double token_time =
        std::chrono::duration<double, std::milli>(token_end - token_start).count();

    stats.total_tokens++;
    stats.total_time_ms += token_time;
    double current_mem = get_memory_usage_mb();
    if (current_mem > stats.peak_memory_mb) stats.peak_memory_mb = current_mem;

    generated_tokens.push_back(next);

    if (!is_clean && stats.total_tokens % 10 == 0) {
      std::cerr << "[STATS] " << stats.total_tokens << " tokens | "
                << std::fixed << std::setprecision(2)
                << stats.tokens_per_second() << " tok/s | "
                << "RAM: " << current_mem << " MB" << std::endl;
    }

    current = next;
    history.push_back(next);
    if (history.size() > 256) history.erase(history.begin());

    // Stop token
    if (next == 50256) break;
  }

  // Output
  if (use_tokenizer) {
    if (!is_clean) std::cout << "\n=== Generated Text ===" << std::endl;
    for (int token : generated_tokens) std::cout << tokenizer.decode(token);
    if (!is_clean) std::cout << "\n=== End Inference ===" << std::endl;
    else           std::cout << std::endl;
  } else {
    if (!is_clean) std::cout << "\n=== Generated Token IDs ===" << std::endl;
    for (int token : generated_tokens) std::cout << token << " ";
    if (!is_clean) std::cout << "\n=== End Inference ===" << std::endl;
    else           std::cout << std::endl;
  }

  if (!is_clean) stats.print_summary();

  return 0;
}
