#pragma once
#include <cstdint>

namespace bitmamba {
    void init_lut();
    // Expose LUT if needed by kernels, or keep it private in kernels.cpp if possible.
    // The bitlinear_forward function in kernels.cpp uses it.
    extern uint32_t UNPACK_LUT[256]; 
}
