#include "utils.h"
#include <algorithm>
#ifdef _DEBUG
#include <iostream>
#endif /* _DEBUG */

// Convert a float in the range [0, 1] to an 8-bit unsigned integer in the range [0, 255]
uint8_t floatToByte(float value) {
    return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255);
}

void hostAllocate(float*& data_ptr, size_t size)
{
    data_ptr = new float[size];
}

// Just so we can look more similar to cuda
void deleteHostAllocate(float*& data)
{
    delete[] data;
}

// doesn't expect for size=1, function won't work
void CreateNormalKernel(float* Kernel, size_t size)
{
    int radius = size/2;
    float sigma = float(radius / 2);
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

            Kernel[(y + radius) * size + (x + radius)] = kernelValue;
            sum += kernelValue;
        }
    }
    return;
}

void CreateGyKernel(float* kernel)
{
    kernel[0] =  1;
    kernel[1] =  2;
    kernel[2] =  1;
    kernel[3] =  0;
    kernel[4] =  0;
    kernel[5] =  0;
    kernel[6] = -1;
    kernel[7] = -2;
    kernel[8] = -1;
}

void CreateGxKernel(float* kernel)
{
    kernel[0] =  1;
    kernel[1] =  0;
    kernel[2] = -1;
    kernel[3] =  2;
    kernel[4] =  0;
    kernel[5] = -2;
    kernel[6] =  1;
    kernel[7] =  0;
    kernel[8] = -1;
}

void NormalizeVec(std::vector<float>& vec, float lower, float upper)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        vec[i] = (vec[i] - lower) / (upper - lower);
    }
}