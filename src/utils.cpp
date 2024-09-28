#include "utils.h"
#include <algorithm>

// Convert a float in the range [0, 1] to an 8-bit unsigned integer in the range [0, 255]
uint8_t floatToByte(float value) {
    return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255);
}