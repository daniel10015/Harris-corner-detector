#pragma once
#include <cstdint>
#include <algorithm>

#define E_APPROX 2.71828182845904523536028747135266249775724709369995
#define PI_APPROX 3.14159265358979323846264338327950288419716939937510

// Convert a float in the range [0, 1] to an 8-bit unsigned integer in the range [0, 255]
uint8_t floatToByte(float value);

// Just so we can look more similar to cuda
void hostAllocate(float* data_ptr, size_t size)
{
	data_ptr = new float[size];
}

// Just so we can look more similar to cuda
void deleteHostAllocate(void* data)
{
	delete[] data;
}

// doesn't expect for size=1, function won't work
void CreateKernel(float* kernel, size_t size)
{
    int radius = size / 2;
    float sigma = std::max(float(radius / 2), 1.0);
    float sum = 0;

    // populate kernel 
    for (int x = -radius; x <= radius; x++)
    {
        for (int y = -radius; y <= radius; y++)
        {
            // compute guas distribution 
            double exponentNumerator = double(-1 * (x * x + y * y));
            double exponentDenominator = (2 * sigma * sigma);
            double eExpression = pow(E_APPROX, exponentNumerator / exponentDenominator);

            float kernelValue = (eExpression / (2 * PI_APPROX * sigma * sigma));

            kernel[(y + radius) * size + (x + radius)] = kernelValue;
            sum += kernelValue;
        }
    }
}