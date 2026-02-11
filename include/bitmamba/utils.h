#pragma once
#include <iostream>
#include <iomanip>
#include <string>

namespace bitmamba {
    // Get current RAM usage of the process in MB (Linux)
    double get_memory_usage_mb();

    struct InferenceStats {
        int total_tokens;
        double total_time_ms;
        double peak_memory_mb;
        double initial_memory_mb;
        
        InferenceStats();
        
        double tokens_per_second() const;
        void print_summary() const;
    };
}
