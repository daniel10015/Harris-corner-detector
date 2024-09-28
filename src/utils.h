#pragma once
#include <cstdint>

#define E_APPROX 2.71828182845904523536028747135266249775724709369995
#define PI_APPROX 3.14159265358979323846264338327950288419716939937510

// Convert a float in the range [0, 1] to an 8-bit unsigned integer in the range [0, 255]
uint8_t floatToByte(float value);