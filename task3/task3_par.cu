#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iomanip>

using namespace std;

__global__ void addKernel(int* A, int* B, int* res, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) res[i] = A[i] + B[i];
}

__global__ void subKernel(int* A, int* B, int* res, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) res[i] = A[i] - B[i];
}

__global__ void mulKernel(int* A, int* B, int* res, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) res[i] = A[i] * B[i];
}

__global__ void divKernel(int* A, int* B, float* res, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) res[i] = (float)A[i] / B[i];
}

void generateArray(int* arr, int size) {
    for (int i = 0; i < size; ++i)
        arr[i] = rand() % 100 + 1;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cout << "Использование: ./cuda_operations <кол-во_запусков> <размер_массива> <кол-во_потоков>\n";
        return 1;
    }

    int runs = atoi(argv[1]);
    int size = atoi(argv[2]);
    int threads = atoi(argv[3]);

    srand(time(nullptr));

    int* h_A = new int[size];
    int* h_B = new int[size];
    int* h_res = new int[size];
    float* h_fres = new float[size];

    int *d_A, *d_B, *d_res;
    float* d_fres;

    cudaMalloc(&d_A, size * sizeof(int));
    cudaMalloc(&d_B, size * sizeof(int));
    cudaMalloc(&d_res, size * sizeof(int));
    cudaMalloc(&d_fres, size * sizeof(float));

    double total_add = 0.0, total_sub = 0.0, total_mul = 0.0, total_div = 0.0;

    for (int r = 0; r < runs; ++r) {
        generateArray(h_A, size);
        generateArray(h_B, size);

        cudaMemcpy(d_A, h_A, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size * sizeof(int), cudaMemcpyHostToDevice);

        int blocks = (size + threads - 1) / threads;

        clock_t start = clock();
        addKernel<<<blocks, threads>>>(d_A, d_B, d_res, size);
        cudaDeviceSynchronize();
        clock_t end = clock();
        total_add += double(end - start) / CLOCKS_PER_SEC;

        start = clock();
        subKernel<<<blocks, threads>>>(d_A, d_B, d_res, size);
        cudaDeviceSynchronize();
        end = clock();
        total_sub += double(end - start) / CLOCKS_PER_SEC;

        start = clock();
        mulKernel<<<blocks, threads>>>(d_A, d_B, d_res, size);
        cudaDeviceSynchronize();
        end = clock();
        total_mul += double(end - start) / CLOCKS_PER_SEC;

        start = clock();
        divKernel<<<blocks, threads>>>(d_A, d_B, d_fres, size);
        cudaDeviceSynchronize();
        end = clock();
        total_div += double(end - start) / CLOCKS_PER_SEC;
    }

    cout << "\nСреднее время за " << runs << " запусков:\n";
    cout << "Сложение (CUDA):      " << std::fixed << std::setprecision(8) << total_add / runs << " секунд\n";
    cout << "Вычитание (CUDA):     " << std::fixed << std::setprecision(8) << total_sub / runs << " секунд\n";
    cout << "Умножение (CUDA):     " << std::fixed << std::setprecision(8) << total_mul / runs << " секунд\n";
    cout << "Деление (CUDA):       " << std::fixed << std::setprecision(8) << total_div / runs << " секунд\n";

    delete[] h_A;
    delete[] h_B;
    delete[] h_res;
    delete[] h_fres;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_res);
    cudaFree(d_fres);


    return 0;
}
