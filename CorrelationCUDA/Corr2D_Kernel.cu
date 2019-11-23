#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Debug stuff.
#define SHOW_VARIABLE(x) \
    std::cout << #x" = " << x << std::endl;

// CUDA related constants.
namespace CUDA_PARAMS
{
const int CUDA_MAX_THREADS_PER_BLOCK = 1024;
const int CUDA_THREADS_PER_WARP = 32;   
}

// PTA for PackedTensorAccessor
#define PTA_INDEX_TYPE uint32_t

// ========== Device functions. ==========

template <typename scalar_t> 
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t x)
{
    return 1.0 / ( 1.0 + exp(-x) );
}

// ========== Kernel functions. ==========

/*!
 * \param padding The length of the padding. Single side.
 * 
 * This kernel should be launched with block arrangement
 * width coverage * height coverage * baches
 * and thread arrangement
 * x * y ( width, height )
 */
template <typename scalar_t> 
__global__ void k_from_BCHW_2_BHWC_padded(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, PTA_INDEX_TYPE> input, 
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, PTA_INDEX_TYPE> output,
    int padding )
{
    const int idxX    = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxY    = blockIdx.y * blockDim.y + threadIdx.y;
    const int strideX = gridDim.x * blockDim.x;
    const int strideY = gridDim.y * blockDim.y;

    const int b = blockIdx.z;

    const int channels = input.size(1);
    const int height   = input.size(2);
    const int width    = input.size(3);

    scalar_t value = 0.0;

    for ( int c = 0; c < channels; c++ )
    {
        for ( int y = idxY; y < height; y += strideY )
        {
            for ( int x = idxX; x < width; x += strideX )
            {
                // Get the data.
                value = input[b][c][y][x];

                // Output the data.
                output[b][y+padding][x+padding][c] = value;
            }
        }
    }
}

/*!
 * \param padding The padding width, single size. Should be non-negative.
 * \param kernelSize The kernel size, whole size. Should be a positive odd number.
 * \param maxDisplacement The correlation neighborhood along the x (width) direction. Single side. Should be positive.
 * \param strideK The moving stride of the kernel. Positive.
 * \param strideD The moving stride within the neighborhood for correlation. Positive.
 */
template <typename scalar_t>
__global__ void k_corr_2d_forward( 
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, PTA_INDEX_TYPE> input0,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, PTA_INDEX_TYPE> input1,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, PTA_INDEX_TYPE> output,
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD)
{
    const int idxC    = threadIdx.x;
    const int strideC = blockDim.x;

    const int idxB    = blockIdx.z;
    const int idxXOut = blockIdx.x;
    const int idxYOut = blockIdx.y;

    const int B   = input0.size(0);
    const int H   = input0.size(1) - padding * 2;
    const int W   = input0.size(2) - padding * 2;
    const int inC = input0.size(3);

    // Kernel.
    const int k2 = kernelSize * kernelSize;
    const int nElements = k2 * inC;

    // Output dimensions.
    const int gridRadius = maxDisplacement / strideD;
    const int outC = gridRadius + 1;

    // Shared memory.
    extern __shared__ char sharedMemory[];
    scalar_t* kernel0 = (scalar_t*)sharedMemory;
    scalar_t* corrResults = kernel0 + nElements;

    // Starting indices.
    const int x0 = idxXOut * strideK + maxDisplacement;
    const int y0 = idxYOut * strideK;

    // Load the kernel data of input0 into the shared memory.
    for ( int j = 0; j < kernelSize; j++ ) // Height.
    {
        for ( int i = 0; i < kernelSize; i++ ) // Width.
        {
            int chStart = ( j*kernelSize + i ) * inC;
            for ( int c = idxC; c < inC; c += strideC )
            {
                kernel0[ chStart + c ] = input0[idxB][y0+j][x0+i][c];
            }
        }
    }

    __syncthreads();

    for ( int idxOutC = 0; idxOutC < outC; idxOutC++ )
    {
        corrResults[idxC] = 0.0; // Clear the shared memory.

        int y1 = y0;
        int x1 = x0 - gridRadius * strideD + idxOutC * strideD;

        for ( int j = 0; j < kernelSize; j++ )
        {
            for ( int i = 0; i < kernelSize; i++ )
            {
                int chStart = ( j*kernelSize + i ) * inC;
                for ( int c = idxC; c < inC; c += strideC )
                {
                    corrResults[idxC] += kernel0[ chStart + c ] * input0[idxB][y1+j][x1+i][c];
                }
            }
        }

        __syncthreads();

        if ( 0 == idxC )
        {
            scalar_t kernelSum = 0.0;

            for ( int i = 0; i < blockDim.x; i++ )
            {
                kernelSum += corrResults[i];
            }

            output[idxB][idxOutC][idxYOut][idxXOut] = kernelSum / static_cast<scalar_t>( nElements );
        }
    }

    // __syncthreads();

    // if ( 0 == idxC )
    // {
    //     scalar_t s = 0.0;

    //     for ( int j = 0; j < kernelSize; j++ ) // Height.
    //     {
    //         for ( int i = 0; i < kernelSize; i++ ) // Width.
    //         {
    //             int chStart = ( j*kernelSize + i ) * inC;
    //             for ( int c = 0; c < inC; c++ )
    //             {
    //                 s += kernel0[ chStart + c ];
    //             }
    //         }
    //     }

    //     output[idxB][0][idxYOut][idxXOut] = s;
    // }
}

template <typename scalar_t> 
__global__ void k_corr_2d_backward(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> grad,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> s,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> output )
{
    const int idxX    = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxY    = blockIdx.y * blockDim.y + threadIdx.y;
    const int idxZ    = blockIdx.z * blockDim.z + threadIdx.z;
    const int strideX = gridDim.x * blockDim.x;
    const int strideY = gridDim.y * blockDim.y;
    const int strideZ = gridDim.z * blockDim.z;

    const int b = s.size(0);
    const int h = s.size(1);
    const int w = s.size(2);

    for (int z = idxZ; z < b; z += strideZ )
    {
        for ( int y = idxY; y < h; y += strideY )
        {
            for ( int x = idxX; x < w; x += strideX )
            {
                output[z][y][x] = 
                    grad[z][y][x] * 
                    ( 1.0 - s[z][y][x] ) * s[z][y][x];
            }
        }
    }
}

// ========== Interface functions. ==========

torch::Tensor from_BCHW_2_BHWC_padded_cuda( torch::Tensor input, int padding )
{
    auto b = input.size(0);
    auto c = input.size(1);
    auto h = input.size(2);
    auto w = input.size(3);

    // Create a padded tensor.
    auto output = torch::zeros({b, h + padding*2, w + padding*2, c}, input.options());

    // Kernel launch specification.
    const int threadsX = 16;
    const int threadsY = 16;
    const dim3 blocks( ( w + threadsX - 1 ) / threadsX, ( h + threadsY - 1 ) / threadsY, b );
    const dim3 thrds( threadsX, threadsY, 1 );

    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << "cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    // Kernel launch.
    AT_DISPATCH_FLOATING_TYPES( input.type(), "test_from_BCHW_2_BHWC_padded_cuda", ( [&] {
            k_from_BCHW_2_BHWC_padded<scalar_t><<<blocks, thrds>>>( 
                input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, PTA_INDEX_TYPE>(),
                output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, PTA_INDEX_TYPE>(),
                padding );
        } ) );

    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << "cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    return output;
}

torch::Tensor corr_2d_forward_cuda( 
    torch::Tensor input0, torch::Tensor input1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    // Get the dimensions of the original input.
    const int B = input0.size(0);
    const int H = input0.size(2);
    const int W = input0.size(3);

    const int inC = input0.size(1);

    const int kernelRadius = ( kernelSize - 1 ) / 2;

    const int paddedInputH = H + padding*2;
    const int paddedInputW = W + padding*2;

    const auto outH = static_cast<int>( ceil( static_cast<float>(paddedInputH - kernelRadius * 2) / static_cast<float>(strideK) ) );
    const auto outW = static_cast<int>( ceil( static_cast<float>(paddedInputW - kernelRadius * 2 - maxDisplacement) / static_cast<float>(strideK) ) );
    
    const int gridRadius = maxDisplacement / strideD;
    
    const int outC = gridRadius + 1; // The output channels

    // Rearrange the inputs.
    auto r0 = from_BCHW_2_BHWC_padded_cuda(input0, padding);
    auto r1 = from_BCHW_2_BHWC_padded_cuda(input1, padding);

    // // Debug.
    // SHOW_VARIABLE(B);
    // SHOW_VARIABLE(H);
    // SHOW_VARIABLE(W);
    // SHOW_VARIABLE(inC);
    // SHOW_VARIABLE(outH);
    // SHOW_VARIABLE(outW);
    // SHOW_VARIABLE(gridRadius);
    // SHOW_VARIABLE(r0.size(0));
    // SHOW_VARIABLE(r0.size(1));
    // SHOW_VARIABLE(r0.size(2));
    // SHOW_VARIABLE(r0.size(3));
    // SHOW_VARIABLE(r1.size(0));
    // SHOW_VARIABLE(r1.size(1));
    // SHOW_VARIABLE(r1.size(2));
    // SHOW_VARIABLE(r1.size(3));

    // Create the output.
    auto output = torch::zeros( { B, outC, outH, outW }, input0.options() );

    // Kernel launch specification.
    const int threads = CUDA_PARAMS::CUDA_THREADS_PER_WARP;
    const dim3 blocks( outW, outH, B );
    const dim3 thrds( threads, 1, 1 );

    // Shared memory size.
    // The size of one kernel across all the input channels and 
    // additional space for saving the correlation results for
    // each thread in a block.
    const int sizeSharedMemory = kernelSize * kernelSize * inC + threads;

    // CUDA context check.
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << "corr_2d_forward_cuda: cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    // Kernel launch.
    AT_DISPATCH_FLOATING_TYPES( r0.type(), "corr_2d_forward_cuda", ( [&] {
        k_corr_2d_forward<scalar_t><<<blocks, thrds, sizeSharedMemory*sizeof(scalar_t)>>>( 
            r0.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, PTA_INDEX_TYPE>(),
            r1.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, PTA_INDEX_TYPE>(),
            output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, PTA_INDEX_TYPE>(),
            padding, kernelSize, maxDisplacement, strideK, strideD );
    } ) );

    // CUDA context check.
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << "corr_2d_forward_cuda: cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    return output;
}

std::vector<torch::Tensor> corr_2d_backward_cuda( torch::Tensor grad, torch::Tensor s )
{
    // Get the batch size.
    auto b = s.size(0);

    // Get the 2D tensor dimesnions.
    auto h = s.size(1);
    auto w = s.size(2);

    // The result.
    auto output = torch::zeros_like(s);

    const int threadsX = 2;
    const int threadsY = 2;

    // Kernal launch dimensions.
    const dim3 blocks( ( w + threadsX - 1 ) / threadsX, ( h + threadsY - 1 ) / threadsY, b );
    const dim3 thrds( threadsX, threadsY, 1 );

    // Kernal launch.
    AT_DISPATCH_FLOATING_TYPES( s.type(), "corr_2d_backward_cuda", ( [&] {
        k_corr_2d_backward<scalar_t><<<blocks, thrds>>>( 
            grad.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            s.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>() );
    } ) );

    return { output };
}
