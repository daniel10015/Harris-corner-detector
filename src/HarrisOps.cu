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
    if (i < (width-2*imgCutoff)*(height-2*imgCutoff)) 
    {

        int maxR = 2 * radius; // since 0 -> 2*radius intead of -radius -> radius
        float gd = 0.0;
        for (int y = 0; y <= maxR; y++)
        {
            for (int x = 0; x <= maxR; x++)
            {
                gd += kernel[y * kernelWidth + x] * img[i_row*width + i_col + ((y - radius) * width + x - radius)];

                if (i == 10 && (y==radius && x==radius ))
                {
                    for (int xx = -radius; xx <= radius; xx++)
                    {
                        for (int yy = -radius; yy <= radius; yy++)
                        {
                            float val = kernel[(yy + radius) * kernelWidth + (xx + radius)];
                            printf("%10.6lf | ", val);
                        }
                        printf("\n");
                    }
                    printf("\n\n");
                }
            }
        }
        result[i] = gd;
    }
}

__global__ void elementWiseOp(float* I_1, float* I_2, float* I_out, size_t size, int op)
{
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size)
    {
        if (op == 0) // element-wise add
        {
            I_out[i] = I_1[i] + I_2[i];
        }
        else if (op == 1) // element-wise sub
        {
            I_out[i] = I_1[i] - I_2[i];
        }
        else if (op == 2) // element-wise mult
        {
            I_out[i] = I_1[i] * I_2[i];
        }
        else if (op == 3) // element-wise div
        {
            I_out[i] = I_1[i] / I_2[i];
        }
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
    float min = 900000000000000;
    float max = -900000000000000;
    const size_t imageDerivativeKernelSize = 9;
    const size_t imageDerivativeKernelRadius = 1;
    int radius = 3;
    size_t kernelSize = 2*radius+1;
    size_t kernelLength = kernelSize * kernelSize;
    size_t imgCutoff = radius;
    size_t newWidth = srcImg.width - 2 * radius;
    size_t newHeight = srcImg.height - 2 * radius;
    size_t N = newWidth * newHeight;
    std::cout << "pixel count: " << N << std::endl;
    size_t NBytes = N * sizeof(float);
    size_t finalNBytes = sizeof(float) * (N - 2 * radius * newWidth - 2 * radius * newHeight);
    size_t finalN = finalNBytes / sizeof(float);
    constexpr const int THREADS_PER_BLOCK = 1 << 10;
    const int NUMBER_OF_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaError_t err;

    float* h_Ix   = nullptr, * h_Iy     = nullptr;
    float* d_Ix   = nullptr, * d_Iy     = nullptr; 
    float* d_Ixx  = nullptr, * d_Iyy    = nullptr,     * d_Ixy = nullptr;
    float* d_Sxx  = nullptr, * d_Sxy    = nullptr,     * d_Syy = nullptr;
    float* d_MDet = nullptr, * d_MTrace = nullptr,     * d_MR  = nullptr;
    float* d_MDetFirstTerm = nullptr, * d_MDetSecondTerm = nullptr;

    // allocate gradients
    hostAllocate(h_Ix, N); hostAllocate(h_Iy, N);

    cudaMalloc((void**)&d_Ix, NBytes);                 cudaMalloc((void**)&d_Iy, NBytes);
    cudaMalloc((void**)&d_Ixx, NBytes);                cudaMalloc((void**)&d_Iyy, NBytes);                cudaMalloc((void**)&d_Ixy, NBytes);
    cudaMalloc((void**)&d_Sxx, finalNBytes);           cudaMalloc((void**)&d_Syy, finalNBytes);           cudaMalloc((void**)&d_Sxy, finalNBytes);
    cudaMalloc((void**)&d_MDetFirstTerm, finalNBytes); cudaMalloc((void**)&d_MDetSecondTerm, finalNBytes);
    cudaMalloc((void**)&d_MDet, finalNBytes);          cudaMalloc((void**)&d_MTrace, finalNBytes);
    cudaMalloc((void**)&d_MR, finalNBytes);
    

    // allocate kernels used in gaus calcs
    float* d_NormalKernel = nullptr;
    float* d_GyKernel = nullptr;
    float* d_GxKernel = nullptr;
    cudaMalloc(&d_NormalKernel, kernelLength * sizeof(float));
    cudaMalloc(&d_GyKernel, imageDerivativeKernelSize * sizeof(float));
    cudaMalloc(&d_GxKernel, imageDerivativeKernelSize * sizeof(float));

    float* d_Img;
    cudaMalloc(&d_Img, srcImg.pixels.size() * sizeof(float));
    cudaMemcpy(d_Img, srcImg.pixels.data(), srcImg.pixels.size() * sizeof(float), cudaMemcpyHostToDevice);

    // construct host kernel, copy to device kernels then release host kernel
    float* h_kernelTemp = nullptr;
    hostAllocate(h_kernelTemp, kernelLength);

    // allocate and copy to normal kernel
    CreateNormalKernel(h_kernelTemp, kernelSize);
    err = cudaMemcpy(d_NormalKernel, h_kernelTemp, kernelLength * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("memcpy Error: %s\n", cudaGetErrorString(err)); exit(1); }

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

    // compute Ixx, Iyy, Ixy (note Iyx = Ixy)
    elementWiseOp << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_Ix, d_Ix, d_Ixx, N, (int)ELEM_OP::MULT);
    elementWiseOp << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_Iy, d_Iy, d_Iyy, N, (int)ELEM_OP::MULT);
    elementWiseOp << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_Ix, d_Iy, d_Ixy, N, (int)ELEM_OP::MULT);

    cudaDeviceSynchronize();
    // Compute weighted sums of Ixx, Iyy, Ixy 
    Sobel << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_NormalKernel, d_Ixx, d_Sxx, newWidth, newHeight, radius, radius);
    Sobel << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_NormalKernel, d_Iyy, d_Syy, newWidth, newHeight, radius, radius);
    Sobel << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_NormalKernel, d_Ixy, d_Sxy, newWidth, newHeight, radius, radius);
    
    /*
    // temp -----------
    // place output image into dstImg
    dstImg.height = newHeight-2*radius;
    dstImg.width = newWidth-2*radius;
    dstImg.maxColorValue = 255;

    if (dstImg.height * dstImg.width * sizeof(float) < finalNBytes)
    {
        std::cout << "WARNING: dim is < bytes\n";
        std::cout << "dim size: " << dstImg.height * dstImg.width * sizeof(float) << std::endl;
        std::cout << "byte size: " << finalNBytes << std::endl;
    }

    dstImg.pixels.resize(dstImg.height * dstImg.width);
    cudaMemcpy(dstImg.pixels.data(), d_Sxy, finalNBytes, cudaMemcpyDeviceToHost);

    // normalize
    min =  900000000000000;
    max = -900000000000000;
    for (size_t i = 0; i < dstImg.pixels.size(); i++)
    {
        if (dstImg.pixels[i] < min)
            min = dstImg.pixels[i];
        if (dstImg.pixels[i] > max)
            max = dstImg.pixels[i];
    }

    NormalizeVec(dstImg.pixels, min, max);

    return;
    // temp -----------
    */
    

    cudaDeviceSynchronize();
    // use matrix made up of Sxx, Sxy, Sxy, Syy to compute R = det(M) / trace(M)
    // determinant intermediate calculations
    elementWiseOp << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_Sxx, d_Syy, d_MDetFirstTerm, finalN, (int)ELEM_OP::MULT);
    elementWiseOp << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_Sxy, d_Sxy, d_MDetSecondTerm, finalN, (int)ELEM_OP::MULT);
    
    // compute trace
    elementWiseOp << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_Sxx, d_Syy, d_MTrace, finalN, (int)ELEM_OP::ADD);

    cudaDeviceSynchronize();

    // compute determinant with intermediate calculations
    elementWiseOp << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_MDetFirstTerm, d_MDetSecondTerm, d_MDet, finalN, (int)ELEM_OP::SUB);

    cudaDeviceSynchronize();
    // compute R = det(M) / trace(M)
    elementWiseOp << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_MDet, d_MTrace, d_MR, finalN, (int)ELEM_OP::DIV);
    
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)	printf("Gauss Error: %s\n", cudaGetErrorString(err));
    else std::cout << "Cuda Success!\n";

    
    // place output image into dstImg
    dstImg.height = newHeight - 2 * radius;
    dstImg.width = newWidth - 2 * radius;
    dstImg.maxColorValue = 255;
    if (dstImg.height * dstImg.width * sizeof(float) < finalNBytes)
    {
        std::cout << "WARNING: dim is < bytes\n";
        std::cout << "dim size: " << dstImg.height * dstImg.width * sizeof(float) << std::endl;
        std::cout << "byte size: " << finalNBytes << std::endl;
    }
    dstImg.pixels.resize(dstImg.height * dstImg.width);
    cudaMemcpy(dstImg.pixels.data(), d_MR, finalNBytes, cudaMemcpyDeviceToHost);

    // normalize
    min = 900000000000000;
    max = -900000000000000;
    for (size_t i = 0; i < dstImg.pixels.size(); i++)
    {
        if (dstImg.pixels[i] < min)
            min = dstImg.pixels[i];
        if (dstImg.pixels[i] > max)
            max = dstImg.pixels[i];
    }

    NormalizeVec(dstImg.pixels, min, max);
    
    return;
}