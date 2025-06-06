#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

__global__ void sumKernel(float* input, float* result, int size) {
    __shared__ float partialSum[1024];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    partialSum[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(result, partialSum[0]);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Использование: ./cuda_sum <кол-во_запусков> <размер_массива>\n";
        return 1;
    }

    int n_runs = std::atoi(argv[1]);
    int size = std::atoi(argv[2]);
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    float* h_array = new float[size];
    for (int i = 0; i < size; i++) {
        h_array[i] = 1.0f;
    }

    float *d_array, *d_result;
    cudaMalloc(&d_array, size * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    double total_time = 0.0;
    float h_result = 0.0f;

    for (int run = 0; run < n_runs; run++) {
        h_result = 0.0f;
        cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

        clock_t start = clock();

        sumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, d_result, size);
        cudaDeviceSynchronize();

        clock_t end = clock();
        total_time += 1000.0 * (end - start) / CLOCKS_PER_SEC; // в миллисекундах

        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    }

    std::cout << "Сумма элементов массива: " << h_result << "\n";
    std::cout << "Среднее время выполнения: " << total_time / n_runs << " мс\n";

    cudaFree(d_array);
    cudaFree(d_result);
    delete[] h_array;
    return 0;
}
