#include "sobel.cuh"

void HarrisCornerDetection(GrayscaleImage& dstImg, GrayscaleImage& srcImg)
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n";
        exit(1);
    }
    // sizes
    size_t kernelSize = 7;
    size_t radius = kernelSize / 2;
    size_t newWidth = srcImg.width - 2*radius;
    size_t newHeight = srcImg.height - 2 * radius;
    size_t N = newWidth * newHeight;
    size_t NBytes = N * sizeof(float);
    const int THREADS_PER_BLOCK = 1 << 10;
    const int NUMBER_OF_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    float* h_Ix, *h_Iy, *d_Ix, *d_Iy; // derivative gradients per-pixel

    // allocate gradients
    hostAllocate(h_Ix, N); hostAllocate(h_Iy, N);
    cudaMalloc((void**)&d_Ix, NBytes); cudaMalloc((void**)&d_Iy, NBytes);

    // allocate kernel used in gaus calcs
    float* d_Kernel;
    cudaMalloc(&d_Kernel, kernelSize * kernelSize * sizeof(float));
    // construct host kernel, copy to device kernel then release host kernel
    float* h_kernelTemp;
    hostAllocate(h_kernelTemp, kernelSize * kernelSize);
    CreateKernel(h_kernelTemp, kernelSize);
    cudaMemcpy(d_Kernel, h_kernelTemp, kernelSize * kernelSize, cudaMemcpyHostToDevice);
    deleteHostAllocate(h_kernelTemp);

    // compute Ix, Iy

    // compute Ixx, Iyy, Ixy (note Iyx = Ixy)

    // Compute weighted sums of Ixx, Iyy, Ixy 

    // use matrix made up of Sxx, Sxy, Sxy, Syy to compute R = det(M) / trace(M)

}