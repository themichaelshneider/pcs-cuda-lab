#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <iomanip>


__device__ void swap(int& a, int& b, bool dir) {
    if ((a > b) == dir) {
        int temp = a;
        a = b;
        b = temp;
    }
}

__global__ void bitonicSortKernel(int* dev_data, int size, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i && ixj < size && i < size) {
        bool ascending = (i & k) == 0;
        swap(dev_data[i], dev_data[ixj], ascending);
    }
}

void bitonicSort(int* data, int size, int threads) {
    int* dev_data;
    cudaMalloc((void**)&dev_data, size * sizeof(int));
    cudaMemcpy(dev_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int blocks = (size + threads - 1) / threads;
            bitonicSortKernel<<<blocks, threads>>>(dev_data, size, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(data, dev_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_data);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Использование: ./bitonic_sort <кол-во запусков> <размер массива> <кол-во потоков>\n";
        return 1;
    }

    int runs = atoi(argv[1]);
    int size = atoi(argv[2]);
    int threads = atoi(argv[3]);

    if ((size & (size - 1)) != 0) {
        std::cout << "Размер массива должен быть степенью двойки!\n";
        return 1;
    }

    srand(time(0));
    clock_t total = 0;

    for (int i = 0; i < runs; ++i) {
        int* arr = new int[size];
        for (int j = 0; j < size; ++j)
            arr[j] = rand() % 100;

        clock_t start = clock();
        bitonicSort(arr, size, threads);
        clock_t end = clock();

        total += (end - start);
        delete[] arr;
    }

  

    double avg_time_s = (double)total / runs / CLOCKS_PER_SEC;
    std::cout << "Среднее время параллельной Bitonic сортировки: " << std::fixed << std::setprecision(8) << avg_time_s << " с\n";

    return 0;
}
