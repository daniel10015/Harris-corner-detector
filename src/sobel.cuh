#ifndef SOBEL_CUH
#define SOBEL_CUH

/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// later look into loading image into shared memory 
__global__ void Sobel(const float* kernel, const float* img, float* result, const int width, const int height, const int kernelSize)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	// bounds check
	if (i < height * width)
	{
		int radius = (kernelSize >> 2) + kernelSize % 1;
		int maxR = 2 * radius;
		float gd = 0.0;
		for (int x = 0; x <= maxR; x++)
		{
			for (int y = 0; y <= maxR; y++)
			{
				gd += kernel[y*kernelSize + x] * img[i+((y-radius)*width+x-radius)];
			}
		}
		result[i] = gd;
	}
}
*/

#endif /* SOBEL_CUH */