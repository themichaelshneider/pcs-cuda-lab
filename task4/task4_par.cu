#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

__global__ void addKernel(int* A, int* B, int* R, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) R[i] = A[i] + B[i];
}

__global__ void subKernel(int* A, int* B, int* R, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) R[i] = A[i] - B[i];
}

__global__ void mulKernel(int* A, int* B, int* R, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) R[i] = A[i] * B[i];
}

__global__ void divKernel(int* A, int* B, float* R, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) R[i] = (float)A[i] / B[i];
}

void generateMatrix1D(int* arr, int size) {
    for (int i = 0; i < size; ++i)
        arr[i] = rand() % 100 + 1;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cout << "Использование: ./cuda_matrix_ops <запуски> <строки> <столбцы> <потоки>\n";
        return 1;
    }

    int runs = atoi(argv[1]);
    int rows = atoi(argv[2]);
    int cols = atoi(argv[3]);
    int threadsPerBlock = atoi(argv[4]);
    int size = rows * cols;

    srand(time(nullptr));

    int* h_A = new int[size];
    int* h_B = new int[size];
    int* h_R = new int[size];
    float* h_RF = new float[size];

    int *d_A, *d_B, *d_R;
    float* d_RF;

    cudaMalloc(&d_A, size * sizeof(int));
    cudaMalloc(&d_B, size * sizeof(int));
    cudaMalloc(&d_R, size * sizeof(int));
    cudaMalloc(&d_RF, size * sizeof(float));

    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    double t_add = 0.0, t_sub = 0.0, t_mul = 0.0, t_div = 0.0;

    for (int r = 0; r < runs; ++r) {
        generateMatrix1D(h_A, size);
        generateMatrix1D(h_B, size);

        cudaMemcpy(d_A, h_A, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size * sizeof(int), cudaMemcpyHostToDevice);

        clock_t start, end;

        start = clock();
        addKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_R, size);
        cudaDeviceSynchronize();
        end = clock();
        t_add += double(end - start) / CLOCKS_PER_SEC;

        start = clock();
        subKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_R, size);
        cudaDeviceSynchronize();
        end = clock();
        t_sub += double(end - start) / CLOCKS_PER_SEC;

        start = clock();
        mulKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_R, size);
        cudaDeviceSynchronize();
        end = clock();
        t_mul += double(end - start) / CLOCKS_PER_SEC;

        start = clock();
        divKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_RF, size);
        cudaDeviceSynchronize();
        end = clock();
        t_div += double(end - start) / CLOCKS_PER_SEC;
    }

    cout << "\nСреднее время выполнения операций за " << runs << " запусков:\n";
    cout << "Сложение (CUDA):   " << t_add / runs << " секунд\n";
    cout << "Вычитание (CUDA):  " << t_sub / runs << " секунд\n";
    cout << "Умножение (CUDA):  " << t_mul / runs << " секунд\n";
    cout << "Деление (CUDA):    " << t_div / runs << " секунд\n";

    delete[] h_A;
    delete[] h_B;
    delete[] h_R;
    delete[] h_RF;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);
    cudaFree(d_RF);

    return 0;
}
