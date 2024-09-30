#pragma once
#include <cstdint>
#include <algorithm>
#include <vector>

#define E_APPROX 2.71828182845904523536028747135266249775724709369995
#define PI_APPROX 3.14159265358979323846264338327950288419716939937510

enum class ELEM_OP
{
	ADD = 0,
	SUB,
	MULT,
	DIV,
};

// Convert a float in the range [0, 1] to an 8-bit unsigned integer in the range [0, 255]
uint8_t floatToByte(float value);

// Just so we can look more similar to cuda
void hostAllocate(float*& data_ptr, size_t size);

// Just so we can look more similar to cuda
void deleteHostAllocate(float*& data);

// doesn't expect for size=1, function won't work
void CreateNormalKernel(float* kernel, size_t size);

// fixed-size of 9
void CreateGyKernel(float* kernel);

// fixed-size of 9
void CreateGxKernel(float* kernel);

void NormalizeVec(std::vector<float>& vec, float lower, float upper);