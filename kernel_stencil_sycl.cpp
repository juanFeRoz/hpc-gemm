#include <algorithm>
#include <random>
#include <sycl/sycl.hpp>
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
void stencil_2d_kernel(sycl::nd_item<2> item, int width, int height,
                       const float *input, float *output)
{
    const uint block_row = item.get_group(0);
    const uint block_col = item.get_group(1);
    const uint thread_row = item.get_local_id(0);
    const uint thread_col = item.get_local_id(1);

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
                auto read = [&](int r, int c)
                {
                    if (r < 0 || r >= height || c < 0 || c >= width)
                        return 0.0f;
                    return input[r * width + c];
                };

                float center = read(row, col);
                float north = read(row - 1, col);
                float south = read(row + 1, col);
                float west = read(row, col - 1);
                float east = read(row, col + 1);

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

    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property::queue::enable_profiling{});

    auto gen = make_rng(seed);
    std::vector<float> input(width * height);
    std::vector<float> output(width * height, 0.0f);
    fillWithRandom(input, gen);

    float *d_input = sycl::malloc_device<float>(width * height, q);
    float *d_output = sycl::malloc_device<float>(width * height, q);

    q.memcpy(d_input, input.data(), width * height * sizeof(float));
    q.memset(d_output, 0, width * height * sizeof(float));
    q.wait();

    sycl::range<2> local_range(_BM / _TM, _BN / _TN);
    sycl::range<2> global_range(((height + _BM - 1) / _BM) * (_BM / _TM),
                                ((width + _BN - 1) / _BN) * (_BN / _TN));

    sycl::event e = q.submit([&](sycl::handler &h)
                             { h.parallel_for(sycl::nd_range<2>(global_range, local_range),
                                              [=](sycl::nd_item<2> item)
                                              {
                                                  stencil_2d_kernel<_BM, _BN, _TM, _TN>(item, width, height, d_input, d_output);
                                              }); });

    e.wait();
    auto t_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

    sycl::free(d_input, q);
    sycl::free(d_output, q);

    return (t_end - t_start) * 1e-6f;
}
