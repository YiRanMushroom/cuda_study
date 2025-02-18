#include <iostream>
#include <ywl_header/all.hpp>
#include "wappers.cuh"
#include "../cpp/multi_dim_span.hpp"

__global__ void test_expensive_on_device(const float *in1, const float *in2, float *out, const int length,
                                         const float add_para,
                                         const float mul_para) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < 1000; i++) {
        if (index < length) {
            out[index] = in1[index] * in2[index] * mul_para + add_para;
        }
    }
}

// write another version of test_expensive but on host
void test_expensive_on_host(const std::vector<float> &in1, const std::vector<float> &in2, std::vector<float> &out,
                            const float add_para, const float mul_para) {
    for (int i = 0; i < 1000; i++) {
        for (int i = 0; i < in1.size(); i++) {
            out[i] = in1[i] * in2[i] * mul_para + add_para;
        }
    }
}

void matrix_mul_host(const float *mat1, const float *mat2, const float *out, int m, int n, int p) {}

int main() {
    // constexpr int arraySize = 5;

    /*std::vector<float> input1_vec(1000000, 1.0f);
    std::vector<float> input2_vec(1000000, 2.0f);
    std::vector<float> output_vec(1000000);

    cuda::CudaArray<float> output;
    // scope
    {
        ywl::miscellaneous::scoped_timer timer;
        for (int i = 0; i < 100; i++) {
            cuda::CudaArray<float> input1 = cuda::CudaArray<float>::create_from_host(
                input1_vec.begin(), input1_vec.end());
            cuda::CudaArray<float> input2 = cuda::CudaArray<float>::create_from_host(
                input2_vec.begin(), input2_vec.end());
            invoke_same_size(test_expensive_on_device, input1, input2)
                    .invoke_with(1.0f, 2.0f).wait().yield_non_block(output_vec.begin(), output_vec.end());
        }
        std::cout << "Time elapsed on device: " << timer.to_string() << std::endl;
    }
    std::cout << "Device Output[999999] = " << output_vec[999999] << std::endl;*/

    auto md_accessor = make_md_accessor(3, 3, 3);

    // test md_accessor
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                std::cout << "md_accessor[" << i << "][" << j << "][" << k << "] = " << md_accessor(i, j, k) <<
                        std::endl;
            }
        }
    }

    /*// scope
    {
        ywl::miscellaneous::scoped_timer timer;
        for (int i = 0; i < 100; i++) {
            std::vector<float> output_vec(1000000);
            test_expensive_on_host(input1_vec, input2_vec, output_vec, 1.0f, 2.0f);
        }
        std::cout << "Time elapsed on host: " << timer.to_string() << std::endl;
    }

    std::cout << "Host Output[999999] = " << output_vec[999999] << std::endl;
    return 0;*/
}
