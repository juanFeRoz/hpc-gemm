#include <algorithm>
#include <hip/hip_runtime.h>
#include <random>
#include <vector>

std::mt19937 make_rng(unsigned int seed = std::random_device{}())
{
    return std::mt19937(seed);
}

template <typename Container>
void fillWithRandom(Container &c, std::mt19937 &gen)
{
    using T = typename Container::value_type;
    std::uniform_real_distribution<T> dis(0.0f, 1.0f);
    std::generate(std::begin(c), std::end(c), [&]()
                  { return dis(gen); });
}

template <const int BM, const int BN, const int TM, const int TN>
__device__ float stencil_read(int width, int height, const float *input, int r, int c)
{
    if (r < 0 || r >= height || c < 0 || c >= width)
        return 0.0f;
    return input[r * width + c];
}

template <const int BM, const int BN, const int TM, const int TN>
__global__ void stencil_2d_kernel(int width, int height, const float *input, float *output)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;
    const uint thread_row = threadIdx.y;
    const uint thread_col = threadIdx.x;

    const int base_row = block_row * BM + thread_row * TM;
    const int base_col = block_col * BN + thread_col * TN;

    for (uint i = 0; i < TM; ++i)
    {
        for (uint j = 0; j < TN; ++j)
        {
            const int row = base_row + i;
            const int col = base_col + j;
            if (row < height && col < width)
            {
                float center = stencil_read<BM, BN, TM, TN>(width, height, input, row, col);
                float north = stencil_read<BM, BN, TM, TN>(width, height, input, row - 1, col);
                float south = stencil_read<BM, BN, TM, TN>(width, height, input, row + 1, col);
                float west = stencil_read<BM, BN, TM, TN>(width, height, input, row, col - 1);
                float east = stencil_read<BM, BN, TM, TN>(width, height, input, row, col + 1);

                output[row * width + col] = 0.5f * center + 0.125f * (north + south + west + east);
            }
        }
    }
}

#ifndef _BM
#define _BM 32
#endif
#ifndef _BN
#define _BN 32
#endif
#ifndef _TM
#define _TM 1
#endif
#ifndef _TN
#define _TN 1
#endif

extern "C" float run_kernel(int width, int height, int K, int BM_arg,
                            int BN_arg, int BK_arg, int TM_arg, int TN_arg,
                            unsigned int seed)
{
    if (BM_arg != _BM || BN_arg != _BN || TM_arg != _TM || TN_arg != _TN)
    {
        return -1.0f;
    }

    auto gen = make_rng(seed);
    std::vector<float> input(width * height);
    fillWithRandom(input, gen);
    float *d_input, *d_output;
    hipMalloc(&d_input, width * height * sizeof(float));
    hipMalloc(&d_output, width * height * sizeof(float));
    hipMemcpy(d_input, input.data(), width * height * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, width * height * sizeof(float));

    dim3 dimBlock(_BN / _TN, _BM / _TM);
    dim3 dimGrid((width + _BN - 1) / _BN, (height + _BM - 1) / _BM);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    stencil_2d_kernel<_BM, _BN, _TM, _TN><<<dimGrid, dimBlock>>>(width, height, d_input, d_output);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_input);
    hipFree(d_output);

    return elapsed_ms;
}
