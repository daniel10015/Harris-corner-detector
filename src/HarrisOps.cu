#include "sobel.cuh"
#include "utils.h"
#include "Image.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "HarrisOps.cuh"

// later look into loading image into shared memory 
__global__ void Sobel(const float* kernel, const float* img, float* result, const size_t width, const size_t height, const size_t imgCutoff, const int radius)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int kernelWidth = 2*radius + 1;


    // translate from result image space to img image space
    const int i_row = i/(width-2*imgCutoff) + imgCutoff;
    // if (i < 400000 && i>399700)
    //     printf("i: %d, i_row: %d\n",i, i_row);
    const int i_col = i%(width-2*imgCutoff) + imgCutoff;
    if (i < (width-imgCutoff)*(height-imgCutoff)) {

        int maxR = 2 * radius; // since 0 -> 2*radius intead of -radius -> radius
        float gd = 0.0;
        for (int x = 0; x <= maxR; x++)
        {
            for (int y = 0; y <= maxR; y++)
            {
                gd += kernel[y * kernelWidth + x] * img[i_row*width + i_col + ((y - radius) * width + x - radius)];
            }
        }

        if (i < 400000 && i>399700) {
            printf("i: %d, img: %10.6lf\n", i, gd);
            //printf("i: %d, img: %10.6lf\n", i, img[i_row * width + i_col + ((0) * width + 0)]);
        }

        result[i] = gd;
    }
}

void HarrisCornerDetection(GrayscaleImage& dstImg, GrayscaleImage& srcImg)
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n";
        exit(1);
    }
    // sizes
    size_t radius = 3;
    size_t kernelSize = 2*radius+1;
    size_t imgCutoff = radius;
    size_t newWidth = srcImg.width - 2 * radius;
    size_t newHeight = srcImg.height - 2 * radius;
    size_t N = newWidth * newHeight;
    std::cout << "pixel count: " << N << std::endl;
    size_t NBytes = N * sizeof(float);
    constexpr const int THREADS_PER_BLOCK = 1 << 10;
    const int NUMBER_OF_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    float* h_Ix = nullptr, * h_Iy = nullptr, * d_Ix = nullptr, * d_Iy = nullptr; // derivative gradients per-pixel

    // allocate gradients
    hostAllocate(h_Ix, N); hostAllocate(h_Iy, N);
    cudaMalloc((void**)&d_Ix, NBytes); cudaMalloc((void**)&d_Iy, NBytes);

    // allocate kernels used in gaus calcs
    float* d_NormalKernel;
    float* d_GyKernel;
    float* d_GxKernel;
    cudaMalloc(&d_NormalKernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_GyKernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_GxKernel, kernelSize * kernelSize * sizeof(float));

    float* d_Img;
    cudaMalloc(&d_Img, srcImg.pixels.size() * sizeof(float));
    cudaMemcpy(d_Img, srcImg.pixels.data(), srcImg.pixels.size() * sizeof(float), cudaMemcpyHostToDevice);

    //std::cout << "source img pixel counnt: " << 

    // construct host kernel, copy to device kernels then release host kernel
    float* h_kernelTemp = nullptr;
    hostAllocate(h_kernelTemp, kernelSize * kernelSize);

    // allocate and copy to normal kernel
    CreateNormalKernel(h_kernelTemp, kernelSize);
    cudaMemcpy(d_NormalKernel, h_kernelTemp, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    const size_t imageDerivativeKernelSize = 9;
    const size_t imageDerivativeKernelRadius = 1;
    // allocate and copy to Gy kernel
    CreateGyKernel(h_kernelTemp);
    cudaMemcpy(d_GyKernel, h_kernelTemp, imageDerivativeKernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // allocate and copy to Gx kernel
    CreateGxKernel(h_kernelTemp);
    cudaMemcpy(d_GxKernel, h_kernelTemp, imageDerivativeKernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // release host kernel temp memory
    deleteHostAllocate(h_kernelTemp);

    std::cout << "number of blocks and threads per block: " << NUMBER_OF_BLOCKS << " / " << THREADS_PER_BLOCK << std::endl;

    std::cout << "width / height: " << srcImg.width << " / " << srcImg.width << std::endl;
    std::cout << "image cutoff: " << imgCutoff << std::endl;
    // compute Ix, Iy
    Sobel << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_GxKernel, d_Img, d_Ix, srcImg.width, srcImg.height, imgCutoff, imageDerivativeKernelRadius);
    Sobel << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_GyKernel, d_Img, d_Iy, srcImg.width, srcImg.height, imgCutoff, imageDerivativeKernelRadius);

    cudaDeviceSynchronize();

    dstImg.height = newHeight;
    dstImg.width = newWidth;
    dstImg.maxColorValue = 255;
    dstImg.pixels.resize(newHeight * newWidth);
    cudaMemcpy(dstImg.pixels.data(), d_Ix, dstImg.pixels.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float min = 900000000000000;
    float max = -900000000000000;
    for (size_t i = 0; i < dstImg.pixels.size(); i++)
    {
        if (dstImg.pixels[i] < min)
            min = dstImg.pixels[i];
        if (dstImg.pixels[i] > max)
            max = dstImg.pixels[i];
    }

    NormalizeVec(dstImg.pixels, min, max);

    // compute Ixx, Iyy, Ixy (note Iyx = Ixy)


    // Compute weighted sums of Ixx, Iyy, Ixy 

    // use matrix made up of Sxx, Sxy, Sxy, Syy to compute R = det(M) / trace(M)

    return;
}