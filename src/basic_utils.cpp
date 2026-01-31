#include "bitmamba/utils.h"
#include <fstream>
#include <sstream>

namespace bitmamba {

    double get_memory_usage_mb() {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                // VmRSS is the resident memory (physical RAM used)
                std::istringstream iss(line.substr(6));
                long kb;
                iss >> kb;
                return kb / 1024.0; // Convert to MB
            }
        }
        return 0.0;
    }

    InferenceStats::InferenceStats() : total_tokens(0), total_time_ms(0), peak_memory_mb(0), initial_memory_mb(0) {}

    double InferenceStats::tokens_per_second() const {
        if (total_time_ms <= 0) return 0;
        return (total_tokens * 1000.0) / total_time_ms;
    }

    void InferenceStats::print_summary() const {
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

}
