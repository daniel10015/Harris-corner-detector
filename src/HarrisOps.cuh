#ifndef HARRIS_OPS_CUH
#define HARRIS_OPS_CUH
#include "sobel.cuh"
#include "utils.h"
#include "Image.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void HarrisCornerDetection(GrayscaleImage& dstImg, GrayscaleImage& srcImg);

#endif /* HARRIS_OPS_CUH */