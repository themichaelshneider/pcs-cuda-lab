#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

__global__ void sum_reduction(int* input, long long* output, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Использование: ./sum_cuda <число запусков> <размер массива> <потоки на блок>\n";
        return 1;
    }

    int runs = std::atoi(argv[1]);
    int size = std::atoi(argv[2]);
    int threads_per_block = std::atoi(argv[3]);
    long long total_time = 0;

    for (int r = 0; r < runs; ++r) {
        int* h_array = new int[size];
        for (int i = 0; i < size; ++i)
            h_array[i] = rand() % 100;

        int* d_array;
        long long* d_partial_sums;
        cudaMalloc(&d_array, size * sizeof(int));
        cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

        int blocks = (size + threads_per_block - 1) / threads_per_block;
        cudaMalloc(&d_partial_sums, blocks * sizeof(long long));

        clock_t start = clock();

        sum_reduction<<<blocks, threads_per_block, threads_per_block * sizeof(int)>>>(d_array, d_partial_sums, size);
        cudaDeviceSynchronize(); // важно: дождаться завершения ядра

        long long* h_partial_sums = new long long[blocks];
        cudaMemcpy(h_partial_sums, d_partial_sums, blocks * sizeof(long long), cudaMemcpyDeviceToHost);

        long long total_sum = 0;
        for (int i = 0; i < blocks; ++i)
            total_sum += h_partial_sums[i];

        clock_t end = clock();
        total_time += (end - start);

        delete[] h_array;
        delete[] h_partial_sums;
        cudaFree(d_array);
        cudaFree(d_partial_sums);
    }

    double avg_time_ms = (double)total_time / runs / CLOCKS_PER_SEC * 1000;
    std::cout << "Среднее время выполнения: " << avg_time_ms << " мс\n";

    return 0;
}
