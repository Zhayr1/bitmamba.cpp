#include "bitmamba/quantization.h"
#include <iostream>

namespace bitmamba {

    uint32_t UNPACK_LUT[256];

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

}
