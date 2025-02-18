#include <iostream>
#include <ywl_header/all.hpp>
#include "wappers.cuh"
#include "../cpp/multi_dim_span.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
// #include <mkl.h>

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

void matrix_mul_host(const float *mat1, const float *mat2, float *out, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += mat1[i * n + k] * mat2[k * p + j];
            }
            out[i * p + j] = sum;
        }
    }
}

__global__ void matrix_mul_device(const float *mat1, const float *mat2, float *out, int m, int n, int p) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < m * p) {
        size_t i = idx / p;
        size_t j = idx % p;

        float sum = 0;
        for (int k = 0; k < n; k++) {
            sum += mat1[i * n + k] * mat2[k * p + j];
        }
        out[i * p + j] = sum;
    }
}

void matrix_mul_device_sim_host(const float *mat1, const float *mat2, float *out, int m, int n, int p,
                                int idx) {
    if (idx < m * p) {
        size_t i = idx / p;
        size_t j = idx % p;

        float sum = 0;
        for (int k = 0; k < n; k++) {
            sum += mat1[i * n + k] * mat2[k * p + j];
        }
        out[i * p + j] = sum;
    }
}

// Another algorithm
#define TILE_SIZE 16

__global__ void matrix_mul_device_second(const float *mat1, const float *mat2, float *out, int m, int n, int p) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    float sum = 0;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < m && (t * TILE_SIZE + threadIdx.x) < n) {
            shared_A[threadIdx.y][threadIdx.x] = mat1[row * n + t * TILE_SIZE + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < p && (t * TILE_SIZE + threadIdx.y) < n) {
            shared_B[threadIdx.y][threadIdx.x] = mat2[(t * TILE_SIZE + threadIdx.y) * p + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < p) {
        out[row * p + col] = sum;
    }
}

void matrix_mul_device_third(const float *A, const float *B, float *C, int m, int n, int p) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, m, n, &alpha, B, p, A, n, &beta, C, p);

    cublasDestroy(handle);
}

/*
void test_cpu() {
    const int m = 30000, n = 100, k = 50000;
    float *A = new float[m * k]{};
    float *B = new float[k * n]{};
    float *C = new float[m * n]{};

    ywl::miscellaneous::timer timer;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
    std::cout << "cpu time: " << timer.to_string() << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
}
*/

int main() {
    // test_cpu();
    // std::terminate();
    // scope
    {
        // input 30000 * 10000
        std::vector<float> input1(300000000, 1);
        // input2 10000 * 50000
        std::vector<float> input2(500000000, 2);

        // scope
        {
            // output 30000 * 50000
            std::vector<float> output(1500000000);

            std::cout << "start" << std::endl;
            auto cuda1 = cuda::CudaArray<float>::create_from_host(input1.begin(), input1.end());
            auto cuda2 = cuda::CudaArray<float>::create_from_host(input2.begin(), input2.end());
            auto cuda_out = cuda::CudaArray<float>::with_length(1500000000);
            std::cout << "cuda finish copy" << std::endl;

            matrix_mul_device_third(cuda1.get_as_buffer(), cuda2.get_as_buffer(),
                                    cuda_out.get_as_buffer(),
                                    30000,
                                    10000,
                                    50000);

            ywl::miscellaneous::scoped_timer timer;

            cuda_out.yield_block(output.begin(), output.end());

            std::cout << "third time: " << timer.to_string() << std::endl;
            // print last element
            std::cout << "output[29999][49999]: " << output[29999 * 5000 + 49999] << std::endl;
        }

        /*// scope
        {
            // output 30000 * 50000
            std::vector<float> output(1500000000);

            std::cout << "start" << std::endl;
            auto cuda1 = cuda::CudaArray<float>::create_from_host(input1.begin(), input1.end());
            auto cuda2 = cuda::CudaArray<float>::create_from_host(input2.begin(), input2.end());
            auto cuda_out = cuda::CudaArray<float>::with_length(1500000000);
            std::cout << "cuda finish copy" << std::endl;

            dim3 blockDim(TILE_SIZE, TILE_SIZE);
            /*dim3 gridDim((p + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
            matrix_mul_device<<<gridDim, blockDim>>>(d_mat1, d_mat2, d_out, m, n, p);#1#
            dim3 gridDim((50000 + TILE_SIZE - 1) / TILE_SIZE, (30000 + TILE_SIZE - 1) / TILE_SIZE);
            matrix_mul_device_second<<<gridDim, blockDim>>
                    >(cuda1.get_as_buffer(), cuda2.get_as_buffer(), cuda_out.get_as_buffer(), 30000, 4000, 50000);
            ywl::miscellaneous::scoped_timer timer;

            cuda_out.yield_block(output.begin(), output.end());

            std::cout << "second time: " << timer.to_string() << std::endl;
            // print last element
            std::cout << "output[29999][49999]: " << output[29999 * 5000 + 49999] << std::endl;
        }*/
    }


    /*
    // scope
    {
        // input 30000 * 4000
        std::vector<float> input1(120000000, 1);
        // input2 4000 * 50000
        std::vector<float> input2(200000000, 2);

        // scope
        {
            // output 30000 * 50000
            std::vector<float> output(1500000000);
            ywl::miscellaneous::scoped_timer timer;
            std::cout << "start" << std::endl;
            auto cuda1 = cuda::CudaArray<float>::create_from_host(input1.begin(), input1.end());
            auto cuda2 = cuda::CudaArray<float>::create_from_host(input2.begin(), input2.end());
            std::cout << "cuda finish copy" << std::endl;

            cuda::invoke_any_size(matrix_mul_device, cuda1, cuda2)
                    .output_length(30000 * 50000)
                    .invoke_with(30000, 4000, 50000)
                    .wait()
                    .yield_non_block(output.begin(), output.end());

            std::cout << "device time: " << timer.to_string() << std::endl;
            // print last element
            std::cout << "output[29999][49999]: " << output[29999 * 5000 + 49999] << std::endl;
        }
    }*/


    // scope
    /*{
        std::vector<float> input1(12000000, 1);
        // input2 4000 * 5000
        std::vector<float> input2(20000000, 2);
        // output 3000 * 5000
        std::vector<float> output(15000000);
        // scope
        {
            ywl::miscellaneous::scoped_timer timer;
            matrix_mul_host(input1.data(), input2.data(), output.data(), 3000, 4000, 5000);
            std::cout << "host time: " << timer.to_string() << std::endl;
            // print last element
            std::cout << "output[2999][4999]: " << output[2999 * 5000 + 4999] << std::endl;
        }
    }*/
}
